"""Artist-disjoint robust multimodal music tagging.

This experiment deliberately treats the Onion annotations as user-generated
music *tags*, rather than genres.  Labels and all preprocessing statistics are
derived from the training artists only.  The proposed RAMT model combines a
reliability-aware gate with modality dropout and is evaluated both under the
observed test data and under predeclared missing-modality interventions.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import sparse
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch import nn

from .common import _bootstrap
from .common import environment_record, load_config, resolve_device, set_seed


MODALITIES = ("audio", "lyrics", "visual")
SOURCE_NAMES = ("audio_essentia", "lyrics_tfidf", "visual_resnet")

# Only spelling and singular/plural variants are merged.  Subjective semantic
# merges (for example, "love" and "love at first listen") are not performed.
TAG_ALIASES = {
    "favourite": "favorites",
    "favourites": "favorites",
    "favorite": "favorites",
    "favorite songs": "favorites",
    "female vocalist": "female vocalists",
    "male vocalist": "male vocalists",
}


def stable_int(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


def canonical_tags(values: Any) -> list[str]:
    if isinstance(values, np.ndarray):
        source = values.tolist()
    else:
        source = list(values or [])
    return sorted({TAG_ALIASES.get(str(tag).strip().lower(), str(tag).strip().lower()) for tag in source if str(tag).strip()})


def build_dataset(processed_root: Path, top_labels: int, minimum_frequency: int) -> tuple[dict[str, pd.DataFrame], list[str], dict[str, int]]:
    tracks = pd.read_parquet(processed_root / "entities" / "onion_tracks.parquet", columns=["track_id", "tags"])
    splits = pd.read_parquet(
        processed_root / "entities" / "onion_track_artist_disjoint_splits.parquet",
        columns=["track_id", "artist_id", "cold_start_split"],
    )
    frame = tracks.merge(splits, on="track_id", how="inner")
    frame["canonical_tags"] = frame["tags"].map(canonical_tags)
    train = frame.loc[frame["cold_start_split"] == "train"]
    frequencies: dict[str, int] = {}
    for tags in train["canonical_tags"]:
        for tag in tags:
            frequencies[tag] = frequencies.get(tag, 0) + 1
    labels = [
        tag
        for tag, count in sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))
        if count >= minimum_frequency
    ][:top_labels]
    index = {label: position for position, label in enumerate(labels)}
    frame["label_ids"] = frame["canonical_tags"].map(lambda tags: [index[tag] for tag in tags if tag in index])
    frame = frame.loc[frame["label_ids"].map(bool)].copy()
    result = {
        split: frame.loc[frame["cold_start_split"] == split].sort_values("track_id", kind="stable").reset_index(drop=True)
        for split in ("train", "validation", "test")
    }
    artists = {split: set(value["artist_id"].astype(str)) for split, value in result.items()}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = artists[left] & artists[right]
        if overlap:
            raise RuntimeError(f"Artist leakage between {left} and {right}: {len(overlap)}")
    return result, labels, frequencies


def targets(frame: pd.DataFrame, labels: int) -> np.ndarray:
    values = np.zeros((len(frame), labels), dtype=np.float32)
    for row, label_ids in enumerate(frame["label_ids"]):
        values[row, label_ids] = 1.0
    return values


def sparse_projection(source_dimension: int, output_dimension: int, name: str) -> sparse.csr_matrix:
    """Deterministic CountSketch projection using every source coordinate."""
    rng = np.random.default_rng(stable_int(f"RAMT/countsketch/{name}/{output_dimension}"))
    rows = np.arange(source_dimension, dtype=np.int32)
    columns = rng.integers(0, output_dimension, size=source_dimension, dtype=np.int32)
    signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=source_dimension)
    return sparse.csr_matrix((signs, (rows, columns)), shape=(source_dimension, output_dimension), dtype=np.float32)


def load_modality(
    processed_root: Path,
    source_name: str,
    item_ids: list[str],
    train_rows: np.ndarray,
    output_dimension: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = processed_root / "features" / "onion" / source_name
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    source_dimension = int(metadata["dimension"])
    projection = sparse_projection(source_dimension, output_dimension, source_name)
    item_position = {track_id: position for position, track_id in enumerate(item_ids)}
    index = pd.read_parquet(root / "index.parquet", columns=["track_id", "shard", "row_offset"])
    index["position"] = index["track_id"].astype(str).map(item_position)
    index = index.dropna(subset=["position"]).copy()
    index["position"] = index["position"].astype(np.int64)
    values = np.zeros((len(item_ids), output_dimension), dtype=np.float32)
    raw_quality = np.zeros(len(item_ids), dtype=np.float32)
    present = np.zeros(len(item_ids), dtype=bool)
    for shard, group in index.groupby("shard", sort=True):
        raw = np.load(root / f"part-{int(shard):05d}.npy", mmap_mode="r")
        offsets = group["row_offset"].to_numpy(dtype=np.int64)
        positions = group["position"].to_numpy(dtype=np.int64)
        block = np.nan_to_num(np.asarray(raw[offsets], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        raw_quality[positions] = np.linalg.norm(block, axis=1)
        values[positions] = np.asarray(block @ projection, dtype=np.float32)
        present[positions] = True

    # Fit normalisation on training artists only; apply it unchanged to val/test.
    train_present = train_rows[present[train_rows]]
    mean = values[train_present].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = values[train_present].std(axis=0, dtype=np.float64).astype(np.float32)
    values[present] = (values[present] - mean) / np.maximum(std, 1e-6)
    norms = np.linalg.norm(values[present], axis=1, keepdims=True)
    values[present] /= np.maximum(norms, 1e-6)
    median_quality = float(np.median(raw_quality[train_present]))
    reliability = np.zeros(len(item_ids), dtype=np.float32)
    reliability[present] = np.clip(raw_quality[present] / max(median_quality, 1e-6), 0.2, 2.0)
    return values, present.astype(np.float32), reliability


def load_features(
    processed_root: Path,
    split_frames: dict[str, pd.DataFrame],
    dimension: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    combined = pd.concat(split_frames.values(), ignore_index=True)
    item_ids = combined["track_id"].astype(str).tolist()
    offsets: dict[str, np.ndarray] = {}
    start = 0
    for split, frame in split_frames.items():
        offsets[split] = np.arange(start, start + len(frame), dtype=np.int64)
        start += len(frame)
    feature_values, availability, reliability = [], [], []
    for source_name in SOURCE_NAMES:
        values, present, quality = load_modality(processed_root, source_name, item_ids, offsets["train"], dimension)
        feature_values.append(values)
        availability.append(present)
        reliability.append(quality)
    return (
        np.stack(feature_values, axis=1),
        np.stack(availability, axis=1),
        np.stack(reliability, axis=1),
        offsets,
    )


class MultimodalTagger(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        labels: int,
        fusion: str,
        active_modalities: tuple[int, ...] = (0, 1, 2),
        modality_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.fusion = fusion
        self.active_modalities = active_modalities
        self.modality_dropout = modality_dropout
        self.encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dimension, hidden_dimension),
                    nn.LayerNorm(hidden_dimension),
                    nn.GELU(),
                    nn.Dropout(0.15),
                )
                for _ in MODALITIES
            ]
        )
        self.gates = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden_dimension + 1, hidden_dimension // 2), nn.GELU(), nn.Linear(hidden_dimension // 2, 1)) for _ in MODALITIES]
        )
        head_input = hidden_dimension * len(active_modalities) + 2 * len(active_modalities) if fusion == "concat" else hidden_dimension
        self.head = nn.Sequential(nn.Linear(head_input, hidden_dimension), nn.GELU(), nn.Dropout(0.15), nn.Linear(hidden_dimension, labels))

    def _drop_modalities(self, availability: torch.Tensor) -> torch.Tensor:
        if not self.training or self.modality_dropout <= 0:
            return availability
        active = availability.clone()
        random_mask = torch.rand_like(active) >= self.modality_dropout
        active = active * random_mask
        choices = torch.as_tensor(self.active_modalities, device=active.device)
        none_left = active[:, choices].sum(dim=1) == 0
        if none_left.any():
            original = availability[none_left][:, choices]
            # Choose an originally available modality, deterministically under
            # the seeded torch RNG, so every example retains at least one view.
            scores = torch.rand_like(original).masked_fill(original == 0, -1.0)
            selected = scores.argmax(dim=1)
            repaired = active[none_left]
            repaired[torch.arange(len(repaired), device=active.device), choices[selected]] = 1.0
            active[none_left] = repaired
        inactive = [index for index in range(len(MODALITIES)) if index not in self.active_modalities]
        if inactive:
            active[:, inactive] = 0.0
        return active

    def forward(
        self,
        features: torch.Tensor,
        availability: torch.Tensor,
        reliability: torch.Tensor,
        return_gates: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        available = self._drop_modalities(availability)
        encoded = torch.stack([encoder(features[:, index]) for index, encoder in enumerate(self.encoders)], dim=1)
        encoded = encoded * available[:, :, None]
        active = list(self.active_modalities)
        if self.fusion == "concat":
            fused = torch.cat(
                [encoded[:, active].reshape(len(features), -1), available[:, active], reliability[:, active] * available[:, active]],
                dim=1,
            )
            gate_weights = available[:, active] / available[:, active].sum(dim=1, keepdim=True).clamp_min(1.0)
        elif self.fusion == "mean":
            weights = available[:, active]
            gate_weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
            fused = (encoded[:, active] * gate_weights[:, :, None]).sum(dim=1)
        elif self.fusion == "reliability":
            scores = torch.cat(
                [self.gates[index](torch.cat([encoded[:, index], reliability[:, index, None]], dim=1)) for index in active],
                dim=1,
            )
            # Reliability supplies an auditable prior, while the learned term
            # can correct it using the content representation.
            scores = scores + torch.log(reliability[:, active].clamp_min(0.05))
            scores = scores.masked_fill(available[:, active] == 0, -1e4)
            gate_weights = torch.softmax(scores, dim=1)
            fused = (encoded[:, active] * gate_weights[:, :, None]).sum(dim=1)
        else:
            raise ValueError(f"Unknown fusion: {self.fusion}")
        logits = self.head(fused)
        return (logits, gate_weights) if return_gates else logits


MODEL_SPECS: dict[str, dict[str, Any]] = {
    "AudioOnly": {"fusion": "mean", "active_modalities": (0,)},
    "LyricsOnly": {"fusion": "mean", "active_modalities": (1,)},
    "VisualOnly": {"fusion": "mean", "active_modalities": (2,)},
    "UniformMean": {"fusion": "mean", "active_modalities": (0, 1, 2)},
    "EarlyConcat": {"fusion": "concat", "active_modalities": (0, 1, 2)},
    "ConcatModDrop": {"fusion": "concat", "active_modalities": (0, 1, 2), "modality_dropout": 0.30},
    "ReliabilityGate": {"fusion": "reliability", "active_modalities": (0, 1, 2)},
    "RAMT": {"fusion": "reliability", "active_modalities": (0, 1, 2), "modality_dropout": 0.30},
}


def batches(length: int, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    order = rng.permutation(length)
    return [order[start : start + batch_size] for start in range(0, length, batch_size)]


@torch.no_grad()
def predict(
    model: MultimodalTagger,
    features: np.ndarray,
    availability: np.ndarray,
    reliability: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities, gates = [], []
    for start in range(0, len(features), batch_size):
        selected = slice(start, start + batch_size)
        output, weights = model(
            torch.as_tensor(features[selected], device=device),
            torch.as_tensor(availability[selected], device=device),
            torch.as_tensor(reliability[selected], device=device),
            return_gates=True,
        )
        probabilities.append(torch.sigmoid(output).cpu().numpy())
        gates.append(weights.cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(gates)


def tune_thresholds(target: np.ndarray, probability: np.ndarray) -> np.ndarray:
    grid = np.arange(0.05, 0.81, 0.025, dtype=np.float32)
    thresholds = np.full(target.shape[1], 0.5, dtype=np.float32)
    for label in range(target.shape[1]):
        scores = [f1_score(target[:, label], probability[:, label] >= value, zero_division=0) for value in grid]
        thresholds[label] = grid[int(np.argmax(scores))]
    return thresholds


def metric_values(target: np.ndarray, probability: np.ndarray, thresholds: np.ndarray) -> dict[str, float]:
    predicted = probability >= thresholds[None, :]
    valid_auc = np.logical_and(target.sum(axis=0) > 0, target.sum(axis=0) < len(target))
    return {
        "mAP": float(average_precision_score(target, probability, average="macro")),
        "MacroF1": float(f1_score(target, predicted, average="macro", zero_division=0)),
        "MicroF1": float(f1_score(target, predicted, average="micro", zero_division=0)),
        "MacroROC_AUC": float(roc_auc_score(target[:, valid_auc], probability[:, valid_auc], average="macro")),
    }


def train_model(
    name: str,
    spec: dict[str, Any],
    seed: int,
    features: dict[str, np.ndarray],
    availability: dict[str, np.ndarray],
    reliability: dict[str, np.ndarray],
    targets_by_split: dict[str, np.ndarray],
    settings: dict[str, Any],
    device: torch.device,
) -> tuple[MultimodalTagger, np.ndarray, pd.DataFrame]:
    set_seed(seed, deterministic=True)
    model = MultimodalTagger(
        input_dimension=features["train"].shape[2],
        hidden_dimension=int(settings["hidden_dimension"]),
        labels=targets_by_split["train"].shape[1],
        **spec,
    ).to(device)
    train_target = targets_by_split["train"]
    ratio = (len(train_target) - train_target.sum(axis=0)) / np.maximum(train_target.sum(axis=0), 1.0)
    positive_weight = torch.as_tensor(np.sqrt(np.clip(ratio, 1.0, 25.0)), device=device)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"]))
    rng = np.random.default_rng(seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_map = -math.inf
    best_thresholds = np.full(train_target.shape[1], 0.5, dtype=np.float32)
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(settings["epochs"]) + 1):
        model.train()
        losses = []
        for selected in batches(len(train_target), int(settings["batch_size"]), rng):
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                torch.as_tensor(features["train"][selected], device=device),
                torch.as_tensor(availability["train"][selected], device=device),
                torch.as_tensor(reliability["train"][selected], device=device),
            )
            loss = loss_function(logits, torch.as_tensor(train_target[selected], device=device))
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
        validation_map = float(average_precision_score(targets_by_split["validation"], validation_probability, average="macro"))
        history.append({"model": name, "seed": seed, "epoch": epoch, "train_loss": float(np.mean(losses)), "validation_mAP": validation_map})
        if validation_map > best_map + float(settings["minimum_improvement"]):
            best_map = validation_map
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            best_thresholds = tune_thresholds(targets_by_split["validation"], validation_probability)
            stale = 0
        else:
            stale += 1
        if stale >= int(settings["patience"]):
            break
    if best_state is None:
        raise RuntimeError(f"No checkpoint created for {name}, seed {seed}")
    model.load_state_dict(best_state)
    return model, best_thresholds, pd.DataFrame(history)


def intervention(name: str, availability: np.ndarray, seed: int) -> np.ndarray:
    result = availability.copy()
    if name == "observed":
        return result
    if name.startswith("no_"):
        result[:, MODALITIES.index(name[3:])] = 0.0
        return result
    rng = np.random.default_rng(seed)
    if name == "random_one_missing":
        for row in range(len(result)):
            candidates = np.flatnonzero(result[row] > 0)
            if len(candidates) > 1:
                result[row, rng.choice(candidates)] = 0.0
        return result
    if name == "random_two_missing":
        for row in range(len(result)):
            candidates = np.flatnonzero(result[row] > 0)
            if len(candidates) > 1:
                keep = int(rng.choice(candidates))
                result[row] = 0.0
                result[row, keep] = 1.0
        return result
    raise ValueError(name)


def summarise(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in metrics.groupby(["model", "condition"], sort=False):
        for metric in ("mAP", "MacroF1", "MicroF1", "MacroROC_AUC"):
            values = group[metric].to_numpy(float)
            standard_deviation = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            rows.append(
                {
                    "model": keys[0],
                    "condition": keys[1],
                    "metric": metric,
                    "mean": float(values.mean()),
                    "std": standard_deviation,
                    "ci95_across_seeds": 1.96 * standard_deviation / math.sqrt(len(values)) if len(values) > 1 else 0.0,
                    "seeds": len(values),
                }
            )
    return pd.DataFrame(rows)


def paired_label_tests(
    y_test: np.ndarray,
    predictions: dict[str, list[np.ndarray]],
    thresholds: dict[str, list[np.ndarray]],
    replicates: int,
    condition: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mean_probabilities = {model: np.mean(values, axis=0) for model, values in predictions.items()}
    mean_thresholds = {model: np.mean(values, axis=0) for model, values in thresholds.items()}
    per_label_rows, test_rows = [], []
    proposed_probability = mean_probabilities["RAMT"]
    proposed_prediction = proposed_probability >= mean_thresholds["RAMT"][None, :]
    for model, probability in mean_probabilities.items():
        predicted = probability >= mean_thresholds[model][None, :]
        for label in range(y_test.shape[1]):
            per_label_rows.append(
                {
                    "model": model,
                    "condition": condition,
                    "label_id": label,
                    "AP": float(average_precision_score(y_test[:, label], probability[:, label])),
                    "F1": float(f1_score(y_test[:, label], predicted[:, label], zero_division=0)),
                }
            )
        if model == "RAMT":
            continue
        for metric in ("AP", "F1"):
            if metric == "AP":
                proposed = np.asarray([average_precision_score(y_test[:, label], proposed_probability[:, label]) for label in range(y_test.shape[1])])
                baseline = np.asarray([average_precision_score(y_test[:, label], probability[:, label]) for label in range(y_test.shape[1])])
            else:
                proposed = np.asarray([f1_score(y_test[:, label], proposed_prediction[:, label], zero_division=0) for label in range(y_test.shape[1])])
                baseline = np.asarray([f1_score(y_test[:, label], predicted[:, label], zero_division=0) for label in range(y_test.shape[1])])
            delta = proposed - baseline
            low, high, p_value = _bootstrap(delta, 20260901 + len(test_rows), replicates)
            test_rows.append(
                {
                    "proposed": "RAMT",
                    "baseline": model,
                    "condition": condition,
                    "metric": metric,
                    "unit": "label",
                    "labels": len(delta),
                    "mean_delta": float(delta.mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                    "p_value": p_value,
                }
            )
    return pd.DataFrame(per_label_rows), pd.DataFrame(test_rows)


def holm_by_condition(frame: pd.DataFrame) -> pd.DataFrame:
    """Holm adjustment within each predeclared condition/metric family."""
    result = frame.copy()
    for _, indices in result.groupby(["condition", "metric"], sort=False).groups.items():
        ordered = result.loc[indices, "p_value"].sort_values().index.tolist()
        previous = 0.0
        for rank, index in enumerate(ordered):
            previous = max(previous, min(1.0, (len(ordered) - rank) * float(result.at[index, "p_value"])))
            result.at[index, "p_value_holm"] = previous
    return result


def run(project_root: Path, config: dict[str, Any], run_name: str, overwrite: bool) -> Path:
    settings = config["robust_multimodal_tagging"]
    output_root = project_root / config["project"]["outputs_root"] / "experiments" / run_name
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(f"Experiment directory exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    processed_root = project_root / config["project"]["processed_root"]
    device = resolve_device(config["runtime"]["device"])
    split_frames, labels, frequencies = build_dataset(
        processed_root,
        int(settings["top_labels"]),
        int(settings["min_train_tag_frequency"]),
    )
    all_features, all_availability, all_reliability, offsets = load_features(
        processed_root,
        split_frames,
        int(settings["projection_dimension"]),
    )
    features = {split: all_features[rows] for split, rows in offsets.items()}
    availability = {split: all_availability[rows] for split, rows in offsets.items()}
    reliability = {split: all_reliability[rows] for split, rows in offsets.items()}
    targets_by_split = {split: targets(frame, len(labels)) for split, frame in split_frames.items()}
    conditions = ["observed", "no_audio", "no_lyrics", "no_visual", "random_one_missing", "random_two_missing"]
    metrics_rows: list[dict[str, Any]] = []
    history_rows: list[pd.DataFrame] = []
    gate_rows: list[dict[str, Any]] = []
    condition_predictions: dict[str, dict[str, list[np.ndarray]]] = {
        condition: {model: [] for model in MODEL_SPECS} for condition in conditions
    }
    selected_thresholds: dict[str, list[np.ndarray]] = {model: [] for model in MODEL_SPECS}
    for seed in [int(value) for value in settings["seeds"]]:
        for model_name, specification in MODEL_SPECS.items():
            model, thresholds, history = train_model(
                model_name,
                specification,
                seed,
                features,
                availability,
                reliability,
                targets_by_split,
                settings,
                device,
            )
            history_rows.append(history)
            selected_thresholds[model_name].append(thresholds)
            for condition_number, condition in enumerate(conditions):
                condition_availability = intervention(condition, availability["test"], seed=20260910 + condition_number)
                probability, gates = predict(
                    model,
                    features["test"],
                    condition_availability,
                    reliability["test"],
                    device,
                    int(settings["evaluation_batch_size"]),
                )
                condition_predictions[condition][model_name].append(probability)
                values = metric_values(targets_by_split["test"], probability, thresholds)
                metrics_rows.append({"model": model_name, "seed": seed, "condition": condition, **values})
                active_names = [MODALITIES[index] for index in specification["active_modalities"]]
                for gate_index, modality in enumerate(active_names):
                    valid = condition_availability[:, MODALITIES.index(modality)] > 0
                    gate_rows.append(
                        {
                            "model": model_name,
                            "seed": seed,
                            "condition": condition,
                            "modality": modality,
                            "mean_gate": float(gates[valid, gate_index].mean()) if valid.any() else float("nan"),
                            "available_tracks": int(valid.sum()),
                        }
                    )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"complete model={model_name} seed={seed}", flush=True)

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(output_root / "metrics_per_seed_condition.csv", index=False)
    summarise(metrics).to_csv(output_root / "metrics_summary.csv", index=False)
    pd.concat(history_rows, ignore_index=True).to_csv(output_root / "training_history.csv", index=False)
    pd.DataFrame(gate_rows).to_csv(output_root / "gate_weights.csv", index=False)
    per_label_frames, paired_test_frames = [], []
    for condition in conditions:
        per_label, paired_tests = paired_label_tests(
            targets_by_split["test"],
            condition_predictions[condition],
            selected_thresholds,
            int(settings["bootstrap_replicates"]),
            condition,
        )
        per_label_frames.append(per_label)
        paired_test_frames.append(paired_tests)
    per_label = pd.concat(per_label_frames, ignore_index=True)
    paired_tests = holm_by_condition(pd.concat(paired_test_frames, ignore_index=True))
    per_label["label"] = per_label["label_id"].map(dict(enumerate(labels)))
    per_label.to_csv(output_root / "per_label_metrics.csv", index=False)
    paired_tests.to_csv(output_root / "paired_label_bootstrap_holm.csv", index=False)
    mean_predictions = []
    for condition, model_predictions in condition_predictions.items():
        for model, probabilities in model_predictions.items():
            frame = pd.DataFrame(np.mean(probabilities, axis=0), columns=[f"score_{label}" for label in labels])
            frame.insert(0, "track_id", split_frames["test"]["track_id"].astype(str).to_numpy())
            frame.insert(0, "condition", condition)
            frame.insert(0, "model", model)
            mean_predictions.append(frame)
    pd.concat(mean_predictions, ignore_index=True).to_parquet(output_root / "mean_test_predictions_all_conditions.parquet", index=False)

    label_payload = {
        "labels": labels,
        "train_frequencies": {label: frequencies[label] for label in labels},
        "aliases": TAG_ALIASES,
        "selection_rule": "top training-artist frequencies after spelling/number canonicalisation",
    }
    (output_root / "labels.json").write_text(json.dumps(label_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "environment": environment_record(device),
        "task": "artist-disjoint multilabel user-generated music tagging",
        "split": "pre-existing Onion artist-disjoint train/validation/test",
        "samples": {split: len(frame) for split, frame in split_frames.items()},
        "artists": {split: int(frame["artist_id"].nunique()) for split, frame in split_frames.items()},
        "artist_intersections": {"train_validation": 0, "train_test": 0, "validation_test": 0},
        "labels": len(labels),
        "feature_projection": "deterministic CountSketch using every source coordinate; train-only normalisation",
        "modalities": list(MODALITIES),
        "coverage": {
            split: {modality: float(availability[split][:, index].mean()) for index, modality in enumerate(MODALITIES)}
            for split in split_frames
        },
        "conditions": conditions,
        "thresholds": "per-label F1 maximisation on validation only",
        "selection": "early stopping on validation mAP",
        "statistical_unit": "label; mean prediction across seeds; paired nonparametric bootstrap with Holm correction",
    }
    (output_root / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    completion = {"status": "complete", "elapsed_seconds": time.perf_counter() - started, "run_directory": str(output_root)}
    (output_root / "completion.json").write_text(json.dumps(completion, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(completion, ensure_ascii=False, indent=2))
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--run-name", default="robust_multimodal_tagging_v1")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.project_root, load_config(args.project_root, args.config), args.run_name, args.overwrite)


if __name__ == "__main__":
    main()
