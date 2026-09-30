"""E1-F: full-feature mechanism-ablation family on the frozen Onion protocol.

The compact-feature study (``robust_multimodal_tagging_v3``) trained the eight
mechanism-ablation architectures (three single modalities, uniform mean, early
concatenation, concatenation with modality dropout, reliability gate and RAMT)
on 256-d CountSketch features with a 128-d hidden layer, whereas the locked
multi-capacity ensemble (MCE), its two components and the two recent strong
baselines use every raw feature coordinate.  Comparing those two groups
confounds the fusion mechanism with the feature representation.

This runner removes that confound.  It retrains the same eight architectures
with the full 1,034/1,000/4,096-d features and the locked H192 recipe used on
FMA (``GenericTagger``), under exactly the artist-disjoint split, label set,
missing-modality interventions, validation-only early stopping / thresholding
and five seeds of every other campaign.  It then

* compares every retrained model per label against the frozen mean test
  predictions of MCE (paired label bootstrap, Holm within condition/metric);
* repeats the mechanism chain tests with RAMT as the proposed model
  (EarlyConcat -> ConcatModDrop -> RAMT; ReliabilityGate -> RAMT);
* records reliability-gate weights per condition for the mechanism figure;
* extends the E4 efficiency audit to the eight architectures.

Nothing of the frozen runs is modified; the new run lives in its own directory.
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

from .advanced_fusion_baselines import load_reference_predictions
from .common import _bootstrap
from .fusion_models import GenericTagger, parameter_count
from .optimized_multimodal_tagging import CONDITIONS, load_full_features
from .common import environment_record, load_config, resolve_device, set_seed
from .robust_multimodal_tagging import (
    MODALITIES,
    build_dataset,
    intervention,
    metric_values,
    summarise,
    targets,
    tune_thresholds,
)


MODEL_SPECS: dict[str, dict[str, Any]] = {
    "AudioOnly": {"fusion": "mean", "active_modalities": (0,)},
    "LyricsOnly": {"fusion": "mean", "active_modalities": (1,)},
    "VisualOnly": {"fusion": "mean", "active_modalities": (2,)},
    "UniformMean": {"fusion": "mean"},
    "EarlyConcat": {"fusion": "concat"},
    "ConcatModDrop": {"fusion": "concat", "modality_dropout": True},
    "ReliabilityGate": {"fusion": "reliability"},
    "RAMT": {"fusion": "reliability", "modality_dropout": True},
}


def build_model(name: str, input_dimensions: list[int], labels: int, settings: dict[str, Any]) -> GenericTagger:
    spec = MODEL_SPECS[name]
    return GenericTagger(
        input_dimensions,
        hidden_dimension=int(settings["hidden_dimension"]),
        labels=labels,
        fusion=str(spec["fusion"]),
        active_modalities=spec.get("active_modalities"),
        modality_dropout=float(settings["modality_dropout"]) if spec.get("modality_dropout") else 0.0,
        representation_dropout=float(settings["representation_dropout"]),
    )


@torch.no_grad()
def predict(
    model: nn.Module,
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
        logits, gates = model(
            [torch.as_tensor(values[start:stop], device=device) for values in features],
            torch.as_tensor(availability[start:stop], device=device),
            torch.as_tensor(reliability[start:stop], device=device),
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
) -> tuple[nn.Module, np.ndarray, pd.DataFrame]:
    set_seed(seed, deterministic=True)
    model = build_model(name, [values.shape[1] for values in features["train"]], target["train"].shape[1], settings).to(device)
    ratio = (len(target["train"]) - target["train"].sum(axis=0)) / np.maximum(target["train"].sum(axis=0), 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(np.sqrt(np.clip(ratio, 1.0, 25.0)), device=device))
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
            batch_features = [torch.as_tensor(values[selected], device=device) for values in features["train"]]
            batch_available = torch.as_tensor(availability["train"][selected], device=device)
            batch_reliability = torch.as_tensor(reliability["train"][selected], device=device)
            batch_target = torch.as_tensor(target["train"][selected], device=device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(batch_features, batch_available, batch_reliability)
            loss = criterion(logits, batch_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_probability, _ = predict(
            model, features["validation"], availability["validation"], reliability["validation"], device, int(settings["evaluation_batch_size"])
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


def holm_by_family(frame: pd.DataFrame) -> pd.DataFrame:
    """Holm adjustment within each (proposed, condition, metric) family."""
    result = frame.copy()
    result["p_value_holm"] = np.nan
    for _, indices in result.groupby(["proposed", "condition", "metric"], sort=False).groups.items():
        ordered = result.loc[indices, "p_value"].sort_values().index.tolist()
        previous = 0.0
        for rank, index in enumerate(ordered):
            previous = max(previous, min(1.0, (len(ordered) - rank) * float(result.at[index, "p_value"])))
            result.at[index, "p_value_holm"] = previous
    return result


def paired_tests(
    target: np.ndarray,
    labels: list[str],
    mean_probabilities: dict[str, dict[str, np.ndarray]],
    mean_thresholds: dict[str, np.ndarray],
    families: dict[str, list[str]],
    replicates: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-label metrics for every model and paired tests for each family.

    ``families`` maps a proposed model to the list of baselines it is tested
    against.  Every test uses the mean prediction across seeds and the mean
    validation-selected threshold across seeds.
    """
    per_label_rows, test_rows = [], []
    for condition in CONDITIONS:
        ap_cache: dict[str, np.ndarray] = {}
        f1_cache: dict[str, np.ndarray] = {}
        for model, condition_values in mean_probabilities.items():
            probability = condition_values[condition]
            predicted = probability >= mean_thresholds[model][None, :]
            ap_cache[model] = np.asarray([average_precision_score(target[:, label], probability[:, label]) for label in range(len(labels))])
            f1_cache[model] = np.asarray([f1_score(target[:, label], predicted[:, label], zero_division=0) for label in range(len(labels))])
            for label_id, label in enumerate(labels):
                per_label_rows.append(
                    {
                        "model": model,
                        "condition": condition,
                        "label_id": label_id,
                        "label": label,
                        "AP": float(ap_cache[model][label_id]),
                        "F1": float(f1_cache[model][label_id]),
                    }
                )
        for proposed, baselines in families.items():
            for baseline in baselines:
                for metric, cache in (("AP", ap_cache), ("F1", f1_cache)):
                    delta = cache[proposed] - cache[baseline]
                    low, high, p_value = _bootstrap(delta, 20260906 + len(test_rows), replicates)
                    test_rows.append(
                        {
                            "condition": condition,
                            "metric": metric,
                            "proposed": proposed,
                            "baseline": baseline,
                            "mean_delta": float(delta.mean()),
                            "ci95_low": low,
                            "ci95_high": high,
                            "p_value": p_value,
                        }
                    )
    return pd.DataFrame(per_label_rows), holm_by_family(pd.DataFrame(test_rows))


def robustness_aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    selected = summary.loc[summary["metric"].isin(["mAP", "MacroF1"])]
    pivot = selected.pivot_table(index=["model", "metric"], columns="condition", values="mean")
    missing = CONDITIONS[1:]
    rows = []
    for (model, metric), values in pivot.iterrows():
        rows.append(
            {
                "model": model,
                "metric": metric,
                "observed": values["observed"],
                "missing_mean": values[missing].mean(),
                "worst_case": values[missing].min(),
                "random_one_missing": values["random_one_missing"],
                "random_two_missing": values["random_two_missing"],
                "random_two_retention_percent": 100.0 * values["random_two_missing"] / values["observed"],
            }
        )
    return pd.DataFrame(rows)


def efficiency_audit(input_dimensions: list[int], labels: int, settings: dict[str, Any], device: torch.device) -> pd.DataFrame:
    batch = int(settings["efficiency_batch_size"])
    repeats = int(settings["efficiency_repeats"])
    torch.manual_seed(20260724)
    features = [torch.randn(batch, dimension, device=device) for dimension in input_dimensions]
    availability = torch.ones(batch, len(input_dimensions), device=device)
    reliability = torch.ones(batch, len(input_dimensions), device=device)
    rows = []
    for name in MODEL_SPECS:
        model = build_model(name, input_dimensions, labels, settings).to(device).eval()
        with torch.no_grad():
            for _ in range(5):
                model(features, availability, reliability)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings = []
            for _ in range(repeats):
                start = time.perf_counter()
                model(features, availability, reliability)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                timings.append((time.perf_counter() - start) * 1000.0)
        latency = float(np.median(timings))
        rows.append(
            {
                "model": name,
                "parameters": int(parameter_count(model)),
                "batch_size": batch,
                "median_latency_ms": latency,
                "throughput_tracks_per_s": batch / (latency / 1000.0),
                "device": str(device),
            }
        )
        model.cpu()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def run(project_root: Path, config: dict[str, Any], run_name: str, overwrite: bool) -> Path:
    settings = config["full_feature_ablation"]
    output_root = project_root / config["project"]["outputs_root"] / "experiments" / run_name
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    device = resolve_device(config["runtime"]["device"])
    processed_root = project_root / config["project"]["processed_root"]
    split_frames, labels, frequencies = build_dataset(processed_root, int(settings["top_labels"]), int(settings["min_train_tag_frequency"]))
    features, availability, reliability = load_full_features(processed_root, split_frames, include_test=True)
    target = {name: targets(frame, len(labels)) for name, frame in split_frames.items()}
    test_track_ids = split_frames["test"]["track_id"].astype(str).to_numpy()
    models = [str(value) for value in settings["models"]]
    unknown = set(models) - set(MODEL_SPECS)
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")

    metrics_rows, history_frames, gate_rows, threshold_rows = [], [], [], []
    predictions: dict[str, dict[str, list[np.ndarray]]] = {model: {condition: [] for condition in CONDITIONS} for model in models}
    threshold_values: dict[str, list[np.ndarray]] = {model: [] for model in models}
    for seed in [int(value) for value in settings["seeds"]]:
        for model_name in models:
            model, thresholds, history = train_model(model_name, seed, features, availability, reliability, target, settings, device)
            history_frames.append(history)
            threshold_values[model_name].append(thresholds)
            for label, value in zip(labels, thresholds, strict=True):
                threshold_rows.append({"model": model_name, "seed": seed, "label": label, "threshold": float(value)})
            active = MODEL_SPECS[model_name].get("active_modalities", tuple(range(len(MODALITIES))))
            for number, condition in enumerate(CONDITIONS):
                condition_availability = intervention(condition, availability["test"], 20260910 + number)
                probability, gates = predict(
                    model, features["test"], condition_availability, reliability["test"], device, int(settings["evaluation_batch_size"])
                )
                predictions[model_name][condition].append(probability)
                metrics_rows.append({"model": model_name, "seed": seed, "condition": condition, **metric_values(target["test"], probability, thresholds)})
                for index in active:
                    valid = condition_availability[:, index] > 0
                    gate_rows.append(
                        {
                            "model": model_name,
                            "seed": seed,
                            "condition": condition,
                            "modality": MODALITIES[index],
                            "mean_gate": float(gates[valid, index].mean()) if valid.any() else float("nan"),
                            "available_tracks": int(valid.sum()),
                        }
                    )
            print(f"complete model={model_name} seed={seed} best_validation_mAP={history['validation_mAP'].max():.5f}", flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(output_root / "metrics_per_seed_condition.csv", index=False)
    summary = summarise(metrics)
    summary.to_csv(output_root / "metrics_summary.csv", index=False)
    robustness_aggregate(summary).to_csv(output_root / "robustness_aggregate.csv", index=False)
    pd.concat(history_frames, ignore_index=True).to_csv(output_root / "training_history.csv", index=False)
    pd.DataFrame(gate_rows).to_csv(output_root / "gate_weights.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(output_root / "validation_thresholds.csv", index=False)

    mean_probabilities: dict[str, dict[str, np.ndarray]] = {
        model: {condition: np.mean(values, axis=0) for condition, values in condition_values.items()} for model, condition_values in predictions.items()
    }
    mean_thresholds: dict[str, np.ndarray] = {model: np.mean(values, axis=0) for model, values in threshold_values.items()}

    reference_root = project_root / str(settings["reference_run"])
    reference_models = [str(value) for value in settings["reference_models"]]
    mean_probabilities.update(load_reference_predictions(reference_root, reference_models, labels, test_track_ids))
    stored_thresholds = pd.read_csv(reference_root / "validation_thresholds.csv")
    for model in reference_models:
        pivot = stored_thresholds.loc[stored_thresholds["model"] == model].pivot_table(index="seed", columns="label", values="threshold")[labels]
        mean_thresholds[model] = pivot.to_numpy(dtype=np.float32).mean(axis=0)

    proposed = str(reference_models[0])
    mechanism_proposed = str(settings["mechanism_proposed"])
    families = {
        proposed: models + reference_models[1:],
        mechanism_proposed: [model for model in models if model != mechanism_proposed],
    }
    per_label, tests = paired_tests(target["test"], labels, mean_probabilities, mean_thresholds, families, int(settings["bootstrap_replicates"]))
    per_label.to_csv(output_root / "per_label_metrics.csv", index=False)
    tests.to_csv(output_root / "paired_label_bootstrap_holm.csv", index=False)

    prediction_frames = []
    for model in models:
        for condition in CONDITIONS:
            frame = pd.DataFrame(mean_probabilities[model][condition], columns=[f"score_{label}" for label in labels])
            frame.insert(0, "track_id", test_track_ids)
            frame.insert(0, "condition", condition)
            frame.insert(0, "model", model)
            prediction_frames.append(frame)
    pd.concat(prediction_frames, ignore_index=True).to_parquet(output_root / "mean_test_predictions_all_conditions.parquet", index=False)

    efficiency = efficiency_audit([values.shape[1] for values in features["train"]], len(labels), settings, device)
    efficiency.to_csv(output_root / "efficiency.csv", index=False)

    (output_root / "labels.json").write_text(
        json.dumps({"labels": labels, "train_frequencies": {label: frequencies[label] for label in labels}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    manifest = {
        "environment": environment_record(device),
        "task": "E1-F full-feature mechanism-ablation family on frozen artist-disjoint 50-label tagging",
        "samples": {name: len(frame) for name, frame in split_frames.items()},
        "artists": {name: int(frame["artist_id"].nunique()) for name, frame in split_frames.items()},
        "features": "all 1034 audio, 1000 lyrics and 4096 visual coordinates; train-only normalisation",
        "recipe": "GenericTagger H192, representation dropout 0.20, modality dropout 0.30 for ConcatModDrop/RAMT (locked recipe shared with FMA)",
        "coverage": {
            split: {modality: float(availability[split][:, index].mean()) for index, modality in enumerate(MODALITIES)} for split in split_frames
        },
        "models": models,
        "reference_models": reference_models,
        "test_families": {key: value for key, value in families.items()},
        "conditions": CONDITIONS,
        "thresholds": "per-label F1 maximisation on validation only",
        "selection": "early stopping on validation mAP",
        "statistical_unit": "label; mean prediction across seeds; paired bootstrap with Holm correction within (proposed, condition, metric)",
    }
    (output_root / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    completion = {"status": "complete", "elapsed_seconds": time.perf_counter() - started, "run_directory": str(output_root)}
    (output_root / "completion.json").write_text(json.dumps(completion, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(completion, ensure_ascii=False, indent=2))
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.project_root, load_config(args.project_root, args.config), args.run_name, args.overwrite)


if __name__ == "__main__":
    main()
