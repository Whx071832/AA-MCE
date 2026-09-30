"""Validation-selected full-feature label-wise multimodal tagging.

The runner has two explicit stages. ``sweep`` never evaluates test labels and
selects among predeclared architectures/losses using validation mAP. ``final``
trains only the locked models across five seeds and evaluates the untouched
artist-disjoint test partition and the fixed missing-modality interventions.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, f1_score
from torch import nn

from .common import _bootstrap
from .common import environment_record, load_config, resolve_device, set_seed
from .robust_multimodal_tagging import (
    MODALITIES,
    SOURCE_NAMES,
    build_dataset,
    holm_by_condition,
    intervention,
    metric_values,
    summarise,
    targets,
    tune_thresholds,
)


CONDITIONS = ["observed", "no_audio", "no_lyrics", "no_visual", "random_one_missing", "random_two_missing"]


def load_full_modality(
    processed_root: Path,
    source_name: str,
    item_ids: list[str],
    train_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load every source coordinate and fit preprocessing on training artists."""
    root = processed_root / "features" / "onion" / source_name
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    dimension = int(metadata["dimension"])
    item_position = {track_id: position for position, track_id in enumerate(item_ids)}
    index = pd.read_parquet(root / "index.parquet", columns=["track_id", "shard", "row_offset"])
    index["position"] = index["track_id"].astype(str).map(item_position)
    index = index.dropna(subset=["position"]).copy()
    index["position"] = index["position"].astype(np.int64)
    values = np.zeros((len(item_ids), dimension), dtype=np.float32)
    present = np.zeros(len(item_ids), dtype=bool)
    raw_quality = np.zeros(len(item_ids), dtype=np.float32)
    for shard, group in index.groupby("shard", sort=True):
        source = np.load(root / f"part-{int(shard):05d}.npy", mmap_mode="r")
        offsets = group["row_offset"].to_numpy(dtype=np.int64)
        positions = group["position"].to_numpy(dtype=np.int64)
        block = np.nan_to_num(np.asarray(source[offsets], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        values[positions] = block
        raw_quality[positions] = np.linalg.norm(block, axis=1)
        present[positions] = True

    train_present = present[:train_count]
    observed_train = values[:train_count][train_present]
    mean = observed_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = observed_train.std(axis=0, dtype=np.float64).astype(np.float32)
    del observed_train
    # Chunked in-place transforms keep the 4,096-d visual matrix affordable.
    for start in range(0, len(values), 4096):
        stop = min(start + 4096, len(values))
        values[start:stop] -= mean
        values[start:stop] /= np.maximum(std, 1e-6)
        values[start:stop][~present[start:stop]] = 0.0
        norms = np.linalg.norm(values[start:stop], axis=1, keepdims=True)
        values[start:stop] /= np.maximum(norms, 1e-6)
    median_quality = float(np.median(raw_quality[:train_count][train_present]))
    reliability = np.zeros(len(values), dtype=np.float32)
    reliability[present] = np.clip(raw_quality[present] / max(median_quality, 1e-6), 0.2, 2.0)
    return values, present.astype(np.float32), reliability


def load_full_features(
    processed_root: Path,
    split_frames: dict[str, pd.DataFrame],
    include_test: bool,
) -> tuple[dict[str, list[np.ndarray]], dict[str, np.ndarray], dict[str, np.ndarray]]:
    names = ["train", "validation"] + (["test"] if include_test else [])
    selected_frames = {name: split_frames[name] for name in names}
    combined = pd.concat(selected_frames.values(), ignore_index=True)
    item_ids = combined["track_id"].astype(str).tolist()
    offsets: dict[str, np.ndarray] = {}
    start = 0
    for name, frame in selected_frames.items():
        offsets[name] = np.arange(start, start + len(frame), dtype=np.int64)
        start += len(frame)
    modality_values, availability, reliability = [], [], []
    for source_name in SOURCE_NAMES:
        values, present, quality = load_full_modality(processed_root, source_name, item_ids, len(split_frames["train"]))
        modality_values.append(values)
        availability.append(present)
        reliability.append(quality)
    features = {name: [values[rows] for values in modality_values] for name, rows in offsets.items()}
    available_by_split = {name: np.stack([values[rows] for values in availability], axis=1) for name, rows in offsets.items()}
    reliability_by_split = {name: np.stack([values[rows] for values in reliability], axis=1) for name, rows in offsets.items()}
    return features, available_by_split, reliability_by_split


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_negative: float, gamma_positive: float = 0.0, clip: float = 0.05) -> None:
        super().__init__()
        self.gamma_negative = gamma_negative
        self.gamma_positive = gamma_positive
        self.clip = clip

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probability = torch.sigmoid(logits)
        positive = probability
        negative = 1.0 - probability
        if self.clip > 0:
            negative = (negative + self.clip).clamp(max=1.0)
        loss = target * torch.log(positive.clamp_min(1e-8)) + (1.0 - target) * torch.log(negative.clamp_min(1e-8))
        probability_true = positive * target + negative * (1.0 - target)
        gamma = self.gamma_positive * target + self.gamma_negative * (1.0 - target)
        return -(loss * torch.pow(1.0 - probability_true, gamma)).mean()


class FullFeatureTagger(nn.Module):
    def __init__(
        self,
        input_dimensions: list[int],
        hidden_dimension: int,
        labels: int,
        fusion: str,
        modality_dropout: float,
        representation_dropout: float,
        label_refinement: bool,
    ) -> None:
        super().__init__()
        self.fusion = fusion
        self.modality_dropout = modality_dropout
        self.encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dimension, hidden_dimension),
                    nn.LayerNorm(hidden_dimension),
                    nn.GELU(),
                    nn.Dropout(representation_dropout),
                    nn.Linear(hidden_dimension, hidden_dimension),
                    nn.GELU(),
                )
                for dimension in input_dimensions
            ]
        )
        self.global_gates = nn.ModuleList([nn.Linear(hidden_dimension + 1, 1) for _ in MODALITIES])
        self.expert_heads = nn.ModuleList([nn.Linear(hidden_dimension, labels) for _ in MODALITIES])
        self.label_gates = nn.ModuleList([nn.Linear(hidden_dimension + 1, labels) for _ in MODALITIES])
        self.concat_head = nn.Sequential(
            nn.Linear(hidden_dimension * len(MODALITIES) + 2 * len(MODALITIES), hidden_dimension),
            nn.GELU(),
            nn.Dropout(representation_dropout),
            nn.Linear(hidden_dimension, labels),
        )
        self.global_head = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension), nn.GELU(), nn.Dropout(representation_dropout), nn.Linear(hidden_dimension, labels)
        )
        self.refinement = nn.Parameter(torch.zeros(labels, labels)) if label_refinement else None
        self.register_buffer("off_diagonal", 1.0 - torch.eye(labels))

    def drop_modalities(self, availability: torch.Tensor) -> torch.Tensor:
        if not self.training or self.modality_dropout <= 0:
            return availability
        result = availability * (torch.rand_like(availability) >= self.modality_dropout)
        missing_all = result.sum(dim=1) == 0
        if missing_all.any():
            original = availability[missing_all]
            scores = torch.rand_like(original).masked_fill(original == 0, -1.0)
            selected = scores.argmax(dim=1)
            repaired = result[missing_all]
            repaired[torch.arange(len(repaired), device=result.device), selected] = 1.0
            result[missing_all] = repaired
        return result

    def forward(
        self,
        features: list[torch.Tensor],
        availability: torch.Tensor,
        reliability: torch.Tensor,
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        available = self.drop_modalities(availability)
        hidden = torch.stack([encoder(features[index]) for index, encoder in enumerate(self.encoders)], dim=1)
        hidden = hidden * available[:, :, None]
        experts = torch.stack([head(hidden[:, index]) for index, head in enumerate(self.expert_heads)], dim=1)
        if self.fusion == "concat":
            logits = self.concat_head(
                torch.cat([hidden.reshape(len(available), -1), available, reliability * available], dim=1)
            )
            gates = available[:, :, None] / available.sum(dim=1, keepdim=True).clamp_min(1.0)[:, :, None]
        elif self.fusion == "global":
            scores = torch.cat(
                [self.global_gates[index](torch.cat([hidden[:, index], reliability[:, index, None]], dim=1)) for index in range(len(MODALITIES))],
                dim=1,
            )
            scores = scores + torch.log(reliability.clamp_min(0.05))
            scores = scores.masked_fill(available == 0, -1e4)
            scalar_gates = torch.softmax(scores, dim=1)
            fused = (hidden * scalar_gates[:, :, None]).sum(dim=1)
            logits = self.global_head(fused)
            gates = scalar_gates[:, :, None].expand(-1, -1, experts.shape[2])
        elif self.fusion == "label":
            scores = torch.stack(
                [self.label_gates[index](torch.cat([hidden[:, index], reliability[:, index, None]], dim=1)) for index in range(len(MODALITIES))],
                dim=1,
            )
            scores = scores + torch.log(reliability.clamp_min(0.05))[:, :, None]
            scores = scores.masked_fill(available[:, :, None] == 0, -1e4)
            gates = torch.softmax(scores, dim=1)
            logits = (experts * gates).sum(dim=1)
        else:
            raise ValueError(self.fusion)
        if self.refinement is not None:
            matrix = self.refinement * self.off_diagonal
            logits = logits + 0.10 * (torch.sigmoid(logits) @ matrix)
        return (logits, experts, gates) if return_details else logits


SPECS: dict[str, dict[str, Any]] = {
    "FullConcat-BCE": {"fusion": "concat", "loss": "bce", "auxiliary_weight": 0.0, "label_refinement": False, "modality_dropout": 0.0},
    "FullConcatMD10-BCE": {"fusion": "concat", "loss": "bce", "auxiliary_weight": 0.0, "label_refinement": False, "modality_dropout": 0.10},
    "FullConcatMD20-BCE": {"fusion": "concat", "loss": "bce", "auxiliary_weight": 0.0, "label_refinement": False, "modality_dropout": 0.20},
    "FullConcatMD-BCE": {"fusion": "concat", "loss": "bce", "auxiliary_weight": 0.0, "label_refinement": False, "modality_dropout": 0.30},
    "FullConcatMD10-ASL2": {"fusion": "concat", "loss": "asl2", "auxiliary_weight": 0.0, "label_refinement": False, "modality_dropout": 0.10},
    "FullConcatMD20-ASL2": {"fusion": "concat", "loss": "asl2", "auxiliary_weight": 0.0, "label_refinement": False, "modality_dropout": 0.20},
    "FullConcatMD10-H256-BCE": {"fusion": "concat", "loss": "bce", "auxiliary_weight": 0.0, "label_refinement": False, "modality_dropout": 0.10, "hidden_dimension": 256},
    "FullConcatMD10-H384-BCE": {"fusion": "concat", "loss": "bce", "auxiliary_weight": 0.0, "label_refinement": False, "modality_dropout": 0.10, "hidden_dimension": 384},
    "FullGlobalRAMT-BCE": {"fusion": "global", "loss": "bce", "auxiliary_weight": 0.0, "label_refinement": False},
    "LabelMoE-BCE": {"fusion": "label", "loss": "bce", "auxiliary_weight": 0.20, "label_refinement": False},
    "LabelMoE-ASL2": {"fusion": "label", "loss": "asl2", "auxiliary_weight": 0.10, "label_refinement": False},
    "LabelMoE-ASL4": {"fusion": "label", "loss": "asl4", "auxiliary_weight": 0.10, "label_refinement": False},
    "LabelMoE-ASL2-Corr": {"fusion": "label", "loss": "asl2", "auxiliary_weight": 0.10, "label_refinement": True},
}


def make_loss(name: str, target: np.ndarray, device: torch.device) -> nn.Module:
    if name == "bce":
        ratio = (len(target) - target.sum(axis=0)) / np.maximum(target.sum(axis=0), 1.0)
        return nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(np.sqrt(np.clip(ratio, 1.0, 25.0)), device=device))
    if name == "asl2":
        return AsymmetricLoss(gamma_negative=2.0)
    if name == "asl4":
        return AsymmetricLoss(gamma_negative=4.0)
    raise ValueError(name)


@torch.no_grad()
def predict(
    model: FullFeatureTagger,
    features: list[np.ndarray],
    availability: np.ndarray,
    reliability: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities, gate_values = [], []
    for start in range(0, len(availability), batch_size):
        stop = min(start + batch_size, len(availability))
        logits, _, gates = model(
            [torch.as_tensor(values[start:stop], device=device) for values in features],
            torch.as_tensor(availability[start:stop], device=device),
            torch.as_tensor(reliability[start:stop], device=device),
            return_details=True,
        )
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        gate_values.append(gates.cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(gate_values)


def train_model(
    name: str,
    seed: int,
    features: dict[str, list[np.ndarray]],
    availability: dict[str, np.ndarray],
    reliability: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    settings: dict[str, Any],
    device: torch.device,
) -> tuple[FullFeatureTagger, np.ndarray, pd.DataFrame]:
    specification = SPECS[name]
    set_seed(seed, deterministic=True)
    model = FullFeatureTagger(
        input_dimensions=[values.shape[1] for values in features["train"]],
        hidden_dimension=int(specification.get("hidden_dimension", settings["hidden_dimension"])),
        labels=target["train"].shape[1],
        fusion=specification["fusion"],
        modality_dropout=float(specification.get("modality_dropout", settings["modality_dropout"])),
        representation_dropout=float(settings["representation_dropout"]),
        label_refinement=bool(specification["label_refinement"]),
    ).to(device)
    criterion = make_loss(str(specification["loss"]), target["train"], device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"]))
    rng = np.random.default_rng(seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_map = -math.inf
    best_thresholds = np.full(target["train"].shape[1], 0.5, dtype=np.float32)
    stale = 0
    history_rows = []
    for epoch in range(1, int(settings["epochs"]) + 1):
        model.train()
        losses = []
        order = rng.permutation(len(target["train"]))
        for start in range(0, len(order), int(settings["batch_size"])):
            selected = order[start : start + int(settings["batch_size"])]
            batch_target = torch.as_tensor(target["train"][selected], device=device)
            batch_available = torch.as_tensor(availability["train"][selected], device=device)
            optimizer.zero_grad(set_to_none=True)
            logits, experts, _ = model(
                [torch.as_tensor(values[selected], device=device) for values in features["train"]],
                batch_available,
                torch.as_tensor(reliability["train"][selected], device=device),
                return_details=True,
            )
            loss = criterion(logits, batch_target)
            auxiliary_weight = float(specification["auxiliary_weight"])
            if auxiliary_weight > 0:
                auxiliary_losses = []
                for modality in range(len(MODALITIES)):
                    valid = batch_available[:, modality] > 0
                    if valid.any():
                        auxiliary_losses.append(criterion(experts[valid, modality], batch_target[valid]))
                loss = loss + auxiliary_weight * torch.stack(auxiliary_losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_probability, _ = predict(
            model,
            features["validation"],
            availability["validation"],
            reliability["validation"],
            device,
            int(settings["evaluation_batch_size"]),
        )
        validation_map = float(average_precision_score(target["validation"], validation_probability, average="macro"))
        history_rows.append({"model": name, "seed": seed, "epoch": epoch, "train_loss": float(np.mean(losses)), "validation_mAP": validation_map})
        if validation_map > best_map + float(settings["minimum_improvement"]):
            best_map = validation_map
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            best_thresholds = tune_thresholds(target["validation"], validation_probability)
            stale = 0
        else:
            stale += 1
        if stale >= int(settings["patience"]):
            break
    if best_state is None:
        raise RuntimeError(name)
    model.load_state_dict(best_state)
    return model, best_thresholds, pd.DataFrame(history_rows)


def validation_sweep(
    project_root: Path,
    config: dict[str, Any],
    output_root: Path,
    split_frames: dict[str, pd.DataFrame],
    labels: list[str],
    device: torch.device,
) -> None:
    settings = config["optimized_multimodal_tagging"]
    features, availability, reliability = load_full_features(
        project_root / config["project"]["processed_root"], split_frames, include_test=False
    )
    target = {name: targets(split_frames[name], len(labels)) for name in ("train", "validation")}
    rows, histories, prediction_frames = [], [], []
    for name in settings["sweep_models"]:
        model, threshold, history = train_model(
            str(name), int(settings["sweep_seed"]), features, availability, reliability, target, settings, device
        )
        probability, _ = predict(
            model, features["validation"], availability["validation"], reliability["validation"], device, int(settings["evaluation_batch_size"])
        )
        values = metric_values(target["validation"], probability, threshold)
        rows.append({"model": name, **values, "best_epoch": int(history.loc[history["validation_mAP"].idxmax(), "epoch"])})
        histories.append(history)
        prediction_frame = pd.DataFrame(probability, columns=[f"score_{label}" for label in labels])
        prediction_frame.insert(0, "track_id", split_frames["validation"]["track_id"].astype(str).to_numpy())
        prediction_frame.insert(0, "model", name)
        prediction_frames.append(prediction_frame)
        print(f"validation complete model={name} mAP={values['mAP']:.6f}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result = pd.DataFrame(rows).sort_values("mAP", ascending=False)
    result.to_csv(output_root / "validation_sweep.csv", index=False)
    pd.concat(histories, ignore_index=True).to_csv(output_root / "validation_training_history.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_parquet(output_root / "validation_predictions.parquet", index=False)
    (output_root / "selection.json").write_text(
        json.dumps(
            {
                "selection_metric": "validation mAP",
                "best_model": str(result.iloc[0]["model"]),
                "test_evaluated": False,
                "rule": "final model must be selected before running --stage final",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def paired_tests(
    target: np.ndarray,
    labels: list[str],
    proposed_name: str,
    probabilities: dict[str, dict[str, np.ndarray]],
    thresholds: dict[str, np.ndarray],
    replicates: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_label_rows, test_rows = [], []
    for condition in CONDITIONS:
        for model, condition_values in probabilities.items():
            probability = condition_values[condition]
            predicted = probability >= thresholds[model][None, :]
            for label_id, label in enumerate(labels):
                per_label_rows.append(
                    {
                        "model": model,
                        "condition": condition,
                        "label_id": label_id,
                        "label": label,
                        "AP": float(average_precision_score(target[:, label_id], probability[:, label_id])),
                        "F1": float(f1_score(target[:, label_id], predicted[:, label_id], zero_division=0)),
                    }
                )
        proposed_probability = probabilities[proposed_name][condition]
        proposed_prediction = proposed_probability >= thresholds[proposed_name][None, :]
        for baseline_name, condition_values in probabilities.items():
            if baseline_name == proposed_name:
                continue
            baseline_probability = condition_values[condition]
            baseline_prediction = baseline_probability >= thresholds[baseline_name][None, :]
            proposed_values = {
                "AP": np.asarray([average_precision_score(target[:, label], proposed_probability[:, label]) for label in range(len(labels))]),
                "F1": np.asarray([f1_score(target[:, label], proposed_prediction[:, label], zero_division=0) for label in range(len(labels))]),
            }
            baseline_values = {
                "AP": np.asarray([average_precision_score(target[:, label], baseline_probability[:, label]) for label in range(len(labels))]),
                "F1": np.asarray([f1_score(target[:, label], baseline_prediction[:, label], zero_division=0) for label in range(len(labels))]),
            }
            for metric in ("AP", "F1"):
                delta = proposed_values[metric] - baseline_values[metric]
                low, high, p_value = _bootstrap(delta, 20261001 + len(test_rows), replicates)
                test_rows.append(
                    {
                        "condition": condition,
                        "metric": metric,
                        "proposed": proposed_name,
                        "baseline": baseline_name,
                        "mean_delta": float(delta.mean()),
                        "ci95_low": low,
                        "ci95_high": high,
                        "p_value": p_value,
                    }
                )
    return pd.DataFrame(per_label_rows), holm_by_condition(pd.DataFrame(test_rows))


def final_experiment(
    project_root: Path,
    config: dict[str, Any],
    output_root: Path,
    split_frames: dict[str, pd.DataFrame],
    labels: list[str],
    device: torch.device,
) -> None:
    settings = config["optimized_multimodal_tagging"]
    proposed_name = str(settings["proposed_model"])
    final_models = [str(value) for value in settings["final_models"]]
    ensemble_settings = settings.get("ensemble")
    if proposed_name not in final_models and not ensemble_settings:
        raise ValueError("proposed_model must be trained or defined as an ensemble")
    features, availability, reliability = load_full_features(
        project_root / config["project"]["processed_root"], split_frames, include_test=True
    )
    target = {name: targets(split_frames[name], len(labels)) for name in ("train", "validation", "test")}
    metrics_rows, history_frames = [], []
    predictions: dict[str, dict[str, list[np.ndarray]]] = {
        model: {condition: [] for condition in CONDITIONS} for model in final_models
    }
    validation_predictions: dict[str, list[np.ndarray]] = {model: [] for model in final_models}
    threshold_values: dict[str, list[np.ndarray]] = {model: [] for model in final_models}
    checkpoint_root = output_root / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    for seed in [int(value) for value in settings["seeds"]]:
        for model_name in final_models:
            model, threshold, history = train_model(
                model_name, seed, features, availability, reliability, target, settings, device
            )
            history_frames.append(history)
            threshold_values[model_name].append(threshold)
            torch.save(model.state_dict(), checkpoint_root / f"{model_name}_{seed}.pt")
            validation_probability, _ = predict(
                model,
                features["validation"],
                availability["validation"],
                reliability["validation"],
                device,
                int(settings["evaluation_batch_size"]),
            )
            validation_predictions[model_name].append(validation_probability)
            for number, condition in enumerate(CONDITIONS):
                condition_availability = intervention(condition, availability["test"], 20260910 + number)
                probability, _ = predict(
                    model,
                    features["test"],
                    condition_availability,
                    reliability["test"],
                    device,
                    int(settings["evaluation_batch_size"]),
                )
                predictions[model_name][condition].append(probability)
                metrics_rows.append(
                    {
                        "model": model_name,
                        "seed": seed,
                        "condition": condition,
                        **metric_values(target["test"], probability, threshold),
                    }
                )
            print(f"final complete model={model_name} seed={seed}", flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if ensemble_settings:
        ensemble_name = str(ensemble_settings["name"])
        components = [str(value) for value in ensemble_settings["components"]]
        weights = np.asarray(ensemble_settings["weights"], dtype=np.float64)
        if ensemble_name != proposed_name or set(components) - set(final_models) or not np.isclose(weights.sum(), 1.0):
            raise ValueError("Invalid locked ensemble configuration")
        predictions[ensemble_name] = {condition: [] for condition in CONDITIONS}
        threshold_values[ensemble_name] = []
        for seed_index, seed in enumerate([int(value) for value in settings["seeds"]]):
            validation_probability = sum(
                float(weight) * validation_predictions[component][seed_index]
                for component, weight in zip(components, weights, strict=True)
            )
            threshold = tune_thresholds(target["validation"], validation_probability)
            threshold_values[ensemble_name].append(threshold)
            for condition in CONDITIONS:
                probability = sum(
                    float(weight) * predictions[component][condition][seed_index]
                    for component, weight in zip(components, weights, strict=True)
                )
                predictions[ensemble_name][condition].append(probability)
                metrics_rows.append(
                    {
                        "model": ensemble_name,
                        "seed": seed,
                        "condition": condition,
                        **metric_values(target["test"], probability, threshold),
                    }
                )
    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(output_root / "metrics_per_seed_condition.csv", index=False)
    summarise(metrics).to_csv(output_root / "metrics_summary.csv", index=False)
    pd.concat(history_frames, ignore_index=True).to_csv(output_root / "training_history.csv", index=False)
    threshold_rows = []
    for model_name, arrays in threshold_values.items():
        for seed, values in zip(settings["seeds"], arrays, strict=True):
            for label, value in zip(labels, values, strict=True):
                threshold_rows.append({"model": model_name, "seed": int(seed), "label": label, "threshold": float(value)})
    pd.DataFrame(threshold_rows).to_csv(output_root / "validation_thresholds.csv", index=False)

    mean_probabilities = {
        model: {condition: np.mean(values, axis=0) for condition, values in condition_values.items()}
        for model, condition_values in predictions.items()
    }
    mean_thresholds = {model: np.mean(values, axis=0) for model, values in threshold_values.items()}
    reference_root = project_root / str(settings["reference_run"])
    reference_predictions = pd.read_parquet(reference_root / "mean_test_predictions_all_conditions.parquet")
    reference_labels = json.loads((reference_root / "labels.json").read_text(encoding="utf-8"))["labels"]
    if reference_labels != labels:
        raise RuntimeError("Reference label order mismatch")
    score_columns = [f"score_{label}" for label in labels]
    for reference_model in settings["reference_models"]:
        name = str(reference_model)
        mean_probabilities[name] = {}
        for condition in CONDITIONS:
            group = reference_predictions.loc[
                (reference_predictions["model"] == name) & (reference_predictions["condition"] == condition)
            ]
            if not np.array_equal(group["track_id"].astype(str).to_numpy(), split_frames["test"]["track_id"].astype(str).to_numpy()):
                raise RuntimeError(f"Reference track order mismatch: {name}/{condition}")
            mean_probabilities[name][condition] = group[score_columns].to_numpy(dtype=np.float32)
        # Reference thresholds are unavailable per seed; recover the test-safe
        # ensemble threshold from the saved validation-selected seed thresholds
        # by recomputing reference F1 predictions is impossible.  The mean of
        # per-label thresholds is reconstructed from v3 per-label predictions
        # only for AP tests; F1 comparisons are excluded below if absent.
        mean_thresholds[name] = np.full(len(labels), 0.5, dtype=np.float32)

    # Only AP comparisons to frozen references are valid because v3 did not
    # persist its validation thresholds. New-model F1 comparisons remain valid.
    per_label, tests = paired_tests(
        target["test"], labels, proposed_name, mean_probabilities, mean_thresholds, int(settings["bootstrap_replicates"])
    )
    tests.loc[tests["baseline"].isin(settings["reference_models"]) & (tests["metric"] == "F1"), ["mean_delta", "ci95_low", "ci95_high", "p_value", "p_value_holm"]] = np.nan
    tests.loc[tests["baseline"].isin(settings["reference_models"]) & (tests["metric"] == "F1"), "note"] = "not tested: reference validation thresholds were not persisted"
    per_label.to_csv(output_root / "per_label_metrics.csv", index=False)
    tests.to_csv(output_root / "paired_label_bootstrap_holm.csv", index=False)
    prediction_frames = []
    for model, condition_values in mean_probabilities.items():
        for condition, probability in condition_values.items():
            frame = pd.DataFrame(probability, columns=score_columns)
            frame.insert(0, "track_id", split_frames["test"]["track_id"].astype(str).to_numpy())
            frame.insert(0, "condition", condition)
            frame.insert(0, "model", model)
            prediction_frames.append(frame)
    pd.concat(prediction_frames, ignore_index=True).to_parquet(output_root / "mean_test_predictions_all_conditions.parquet", index=False)


def run(project_root: Path, config: dict[str, Any], run_name: str, stage: str, overwrite: bool) -> Path:
    settings = config["optimized_multimodal_tagging"]
    output_root = project_root / config["project"]["outputs_root"] / "experiments" / run_name
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    device = resolve_device(config["runtime"]["device"])
    split_frames, labels, frequencies = build_dataset(
        project_root / config["project"]["processed_root"],
        int(settings["top_labels"]),
        int(settings["min_train_tag_frequency"]),
    )
    if stage == "sweep":
        validation_sweep(project_root, config, output_root, split_frames, labels, device)
    elif stage == "final":
        selection_path = project_root / str(settings["selection_run"]) / "selection.json"
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        ensemble_settings = settings.get("ensemble")
        if ensemble_settings:
            ensemble_selection = json.loads(
                (project_root / str(settings["selection_run"]) / "ensemble_selection.json").read_text(encoding="utf-8")
            )
            locked_components = [str(value) for value in ensemble_settings["components"]]
            locked_weights = [float(value) for value in ensemble_settings["weights"]]
            selected_components = [str(ensemble_selection["left"]), str(ensemble_selection["right"])]
            selected_weights = [float(ensemble_selection["left_weight"]), float(ensemble_selection["right_weight"])]
            if (
                str(settings["proposed_model"]) != str(ensemble_settings["name"])
                or locked_components != selected_components
                or not np.allclose(locked_weights, selected_weights)
                or str(ensemble_selection["kind"]) != "probability"
            ):
                raise RuntimeError("Locked ensemble does not match validation-only selection")
        elif str(selection["best_model"]) != str(settings["proposed_model"]):
            raise RuntimeError(f"Locked proposed model does not match validation winner: {selection['best_model']}")
        final_experiment(project_root, config, output_root, split_frames, labels, device)
    else:
        raise ValueError(stage)
    (output_root / "labels.json").write_text(
        json.dumps({"labels": labels, "train_frequencies": {label: frequencies[label] for label in labels}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    manifest = {
        "environment": environment_record(device),
        "stage": stage,
        "task": "artist-disjoint 50-label music tagging",
        "samples": {name: len(frame) for name, frame in split_frames.items()},
        "artists": {name: int(frame["artist_id"].nunique()) for name, frame in split_frames.items()},
        "features": "all 1034 audio, 1000 lyrics and 4096 visual coordinates; train-only normalisation",
        "test_evaluated": stage == "final",
    }
    (output_root / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    completion = {"status": "complete", "stage": stage, "elapsed_seconds": time.perf_counter() - started, "run_directory": str(output_root)}
    (output_root / "completion.json").write_text(json.dumps(completion, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(completion, ensure_ascii=False, indent=2))
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--stage", choices=["sweep", "final"], required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.project_root, load_config(args.project_root, args.config), args.run_name, args.stage, args.overwrite)


if __name__ == "__main__":
    main()
