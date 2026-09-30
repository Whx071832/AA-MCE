"""E2: recent strong fusion baselines on the frozen Onion tagging protocol.

Trains the modality-token transformer (AttnFusion) and the multi-label
trusted evidential fusion (EvidFusion) under exactly the artist-disjoint
split, label set, missing-modality conditions and seeds of the earlier
campaigns, then compares them per label against the frozen mean test
predictions of the locked multi-capacity ensemble and its references.

The runner also performs the E4 efficiency audit: parameter counts and warmed
batch inference latency for every architecture of the final model family.
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
from .fusion_models import (
    AttnFusionTagger,
    EvidFusionTagger,
    evidential_loss,
    parameter_count,
)
from .optimized_multimodal_tagging import CONDITIONS, FullFeatureTagger, load_full_features
from .common import environment_record, load_config, resolve_device, set_seed
from .robust_multimodal_tagging import (
    build_dataset,
    holm_by_condition,
    intervention,
    metric_values,
    summarise,
    targets,
    tune_thresholds,
)


NEW_MODELS = ("AttnFusion", "EvidFusion")


def build_model(name: str, input_dimensions: list[int], labels: int, settings: dict[str, Any]) -> nn.Module:
    if name == "AttnFusion":
        return AttnFusionTagger(
            input_dimensions,
            hidden_dimension=int(settings["hidden_dimension"]),
            labels=labels,
            heads=int(settings["attention_heads"]),
            layers=int(settings["attention_layers"]),
            modality_dropout=float(settings["modality_dropout"]),
            representation_dropout=float(settings["representation_dropout"]),
        )
    if name == "EvidFusion":
        return EvidFusionTagger(
            input_dimensions,
            hidden_dimension=int(settings["hidden_dimension"]),
            labels=labels,
            modality_dropout=float(settings["modality_dropout"]),
            representation_dropout=float(settings["representation_dropout"]),
        )
    raise ValueError(name)


@torch.no_grad()
def predict(
    model: nn.Module,
    features: list[np.ndarray],
    availability: np.ndarray,
    reliability: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    probabilities = []
    for start in range(0, len(availability), batch_size):
        stop = min(start + batch_size, len(availability))
        logits, _ = model(
            [torch.as_tensor(values[start:stop], device=device) for values in features],
            torch.as_tensor(availability[start:stop], device=device),
            torch.as_tensor(reliability[start:stop], device=device),
        )
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probabilities)


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
    if name == "AttnFusion":
        ratio = (len(target["train"]) - target["train"].sum(axis=0)) / np.maximum(target["train"].sum(axis=0), 1.0)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(np.sqrt(np.clip(ratio, 1.0, 25.0)), device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"]))
    rng = np.random.default_rng(seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_map = -math.inf
    best_thresholds = np.full(target["train"].shape[1], 0.5, dtype=np.float32)
    stale = 0
    history_rows = []
    annealing_epochs = float(settings["evidence_annealing_epochs"])
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
            if name == "EvidFusion":
                combined, per_modality = model.opinions(batch_features, batch_available)
                loss = evidential_loss(
                    combined,
                    per_modality,
                    batch_available,
                    batch_target,
                    annealing=min(1.0, epoch / annealing_epochs),
                )
            else:
                logits, _ = model(batch_features, batch_available, batch_reliability)
                loss = criterion(logits, batch_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_probability = predict(
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


def load_reference_predictions(
    run_root: Path,
    models: list[str],
    labels: list[str],
    test_track_ids: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    frame = pd.read_parquet(run_root / "mean_test_predictions_all_conditions.parquet")
    stored_labels = json.loads((run_root / "labels.json").read_text(encoding="utf-8"))["labels"]
    if stored_labels != labels:
        raise RuntimeError(f"Label order mismatch against {run_root}")
    score_columns = [f"score_{label}" for label in labels]
    result: dict[str, dict[str, np.ndarray]] = {}
    for model in models:
        result[model] = {}
        for condition in CONDITIONS:
            group = frame.loc[(frame["model"] == model) & (frame["condition"] == condition)]
            if not np.array_equal(group["track_id"].astype(str).to_numpy(), test_track_ids):
                raise RuntimeError(f"Track order mismatch: {model}/{condition}")
            result[model][condition] = group[score_columns].to_numpy(dtype=np.float32)
    return result


def efficiency_audit(
    input_dimensions: list[int],
    labels: int,
    settings: dict[str, Any],
    device: torch.device,
) -> pd.DataFrame:
    """Parameters and warmed batch latency for the final model family."""
    batch = int(settings["efficiency_batch_size"])
    repeats = int(settings["efficiency_repeats"])
    torch.manual_seed(20260724)
    features = [torch.randn(batch, dimension, device=device) for dimension in input_dimensions]
    availability = torch.ones(batch, len(input_dimensions), device=device)
    reliability = torch.ones(batch, len(input_dimensions), device=device)

    def full_feature(hidden: int, dropout: float) -> nn.Module:
        return FullFeatureTagger(
            input_dimensions=input_dimensions,
            hidden_dimension=hidden,
            labels=labels,
            fusion="concat",
            modality_dropout=dropout,
            representation_dropout=float(settings["representation_dropout"]),
            label_refinement=False,
        )

    members: dict[str, list[nn.Module]] = {
        "FullConcatMD-BCE": [full_feature(192, 0.30)],
        "FullConcatMD10-H384-BCE": [full_feature(384, 0.10)],
        "OptimizedEnsemble": [full_feature(384, 0.10), full_feature(192, 0.30)],
        "AttnFusion": [build_model("AttnFusion", input_dimensions, labels, settings)],
        "EvidFusion": [build_model("EvidFusion", input_dimensions, labels, settings)],
    }
    rows = []
    for name, model_list in members.items():
        model_list = [model.to(device).eval() for model in model_list]
        with torch.no_grad():
            for _ in range(5):
                for model in model_list:
                    model(features, availability, reliability)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings = []
            for _ in range(repeats):
                start = time.perf_counter()
                for model in model_list:
                    model(features, availability, reliability)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                timings.append((time.perf_counter() - start) * 1000.0)
        latency = float(np.median(timings))
        rows.append(
            {
                "model": name,
                "parameters": int(sum(parameter_count(model) for model in model_list)),
                "batch_size": batch,
                "median_latency_ms": latency,
                "throughput_tracks_per_s": batch / (latency / 1000.0),
                "device": str(device),
            }
        )
        for model in model_list:
            model.cpu()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def paired_tests(
    target: np.ndarray,
    labels: list[str],
    mean_probabilities: dict[str, dict[str, np.ndarray]],
    mean_thresholds: dict[str, np.ndarray | None],
    proposed: str,
    replicates: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_label_rows, test_rows = [], []
    for condition in CONDITIONS:
        ap_cache: dict[str, np.ndarray] = {}
        f1_cache: dict[str, np.ndarray | None] = {}
        for model, condition_values in mean_probabilities.items():
            probability = condition_values[condition]
            ap_cache[model] = np.asarray(
                [average_precision_score(target[:, label], probability[:, label]) for label in range(len(labels))]
            )
            thresholds = mean_thresholds.get(model)
            if thresholds is not None:
                predicted = probability >= thresholds[None, :]
                f1_cache[model] = np.asarray(
                    [f1_score(target[:, label], predicted[:, label], zero_division=0) for label in range(len(labels))]
                )
            else:
                f1_cache[model] = None
            for label_id, label in enumerate(labels):
                per_label_rows.append(
                    {
                        "model": model,
                        "condition": condition,
                        "label_id": label_id,
                        "label": label,
                        "AP": float(ap_cache[model][label_id]),
                        "F1": float(f1_cache[model][label_id]) if f1_cache[model] is not None else np.nan,
                    }
                )
        for baseline in mean_probabilities:
            if baseline == proposed:
                continue
            for metric, cache in (("AP", ap_cache), ("F1", f1_cache)):
                if cache[proposed] is None or cache[baseline] is None:
                    continue
                delta = cache[proposed] - cache[baseline]
                low, high, p_value = _bootstrap(delta, 20260801 + len(test_rows), replicates)
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
    return pd.DataFrame(per_label_rows), holm_by_condition(pd.DataFrame(test_rows))


def run(project_root: Path, config: dict[str, Any], run_name: str, overwrite: bool) -> Path:
    settings = config["advanced_fusion_baselines"]
    output_root = project_root / config["project"]["outputs_root"] / "experiments" / run_name
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    device = resolve_device(config["runtime"]["device"])
    processed_root = project_root / config["project"]["processed_root"]
    split_frames, labels, frequencies = build_dataset(
        processed_root, int(settings["top_labels"]), int(settings["min_train_tag_frequency"])
    )
    features, availability, reliability = load_full_features(processed_root, split_frames, include_test=True)
    target = {name: targets(frame, len(labels)) for name, frame in split_frames.items()}
    test_track_ids = split_frames["test"]["track_id"].astype(str).to_numpy()

    metrics_rows, history_frames = [], []
    predictions: dict[str, dict[str, list[np.ndarray]]] = {model: {condition: [] for condition in CONDITIONS} for model in NEW_MODELS}
    threshold_values: dict[str, list[np.ndarray]] = {model: [] for model in NEW_MODELS}
    for seed in [int(value) for value in settings["seeds"]]:
        for model_name in NEW_MODELS:
            model, thresholds, history = train_model(
                model_name, seed, features, availability, reliability, target, settings, device
            )
            history_frames.append(history)
            threshold_values[model_name].append(thresholds)
            for number, condition in enumerate(CONDITIONS):
                condition_availability = intervention(condition, availability["test"], 20260910 + number)
                probability = predict(
                    model,
                    features["test"],
                    condition_availability,
                    reliability["test"],
                    device,
                    int(settings["evaluation_batch_size"]),
                )
                predictions[model_name][condition].append(probability)
                metrics_rows.append(
                    {"model": model_name, "seed": seed, "condition": condition, **metric_values(target["test"], probability, thresholds)}
                )
            print(f"complete model={model_name} seed={seed}", flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(output_root / "metrics_per_seed_condition.csv", index=False)
    summarise(metrics).to_csv(output_root / "metrics_summary.csv", index=False)
    pd.concat(history_frames, ignore_index=True).to_csv(output_root / "training_history.csv", index=False)

    mean_probabilities: dict[str, dict[str, np.ndarray]] = {
        model: {condition: np.mean(values, axis=0) for condition, values in condition_values.items()}
        for model, condition_values in predictions.items()
    }
    mean_thresholds: dict[str, np.ndarray | None] = {
        model: np.mean(values, axis=0) for model, values in threshold_values.items()
    }
    reference_root = project_root / str(settings["reference_run"])
    reference_models = [str(value) for value in settings["reference_models"]]
    mean_probabilities.update(load_reference_predictions(reference_root, reference_models, labels, test_track_ids))
    stored_thresholds = pd.read_csv(reference_root / "validation_thresholds.csv")
    for model in reference_models:
        group = stored_thresholds.loc[stored_thresholds["model"] == model]
        pivot = group.pivot_table(index="seed", columns="label", values="threshold")[labels]
        mean_thresholds[model] = pivot.to_numpy(dtype=np.float32).mean(axis=0)
    v3_root = project_root / str(settings["reference_v3_run"])
    v3_models = [str(value) for value in settings["reference_v3_models"]]
    mean_probabilities.update(load_reference_predictions(v3_root, v3_models, labels, test_track_ids))
    for model in v3_models:
        mean_thresholds[model] = None  # v3 validation thresholds were not persisted; AP-only tests.

    per_label, tests = paired_tests(
        target["test"],
        labels,
        mean_probabilities,
        mean_thresholds,
        proposed=str(settings["reference_models"][0]),
        replicates=int(settings["bootstrap_replicates"]),
    )
    per_label.to_csv(output_root / "per_label_metrics.csv", index=False)
    tests.to_csv(output_root / "paired_label_bootstrap_holm.csv", index=False)

    prediction_frames = []
    for model in NEW_MODELS:
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
        "task": "E2 strong fusion baselines on frozen artist-disjoint 50-label tagging",
        "samples": {name: len(frame) for name, frame in split_frames.items()},
        "models": list(NEW_MODELS),
        "reference_models": reference_models + v3_models,
        "statistical_unit": "label; mean prediction across seeds; paired bootstrap with Holm correction",
        "f1_tests": "only for models with persisted validation thresholds",
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
