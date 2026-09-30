"""E7: six-view extended-modality study on the frozen Onion tagging protocol.

Every sense of the three-modality study receives a second, architecturally
different precomputed view that ships with Music4All-Onion:

======  =====================================  =======  =========
sense   view                                   dims     source
======  =====================================  =======  =========
audio   Essentia descriptors (A1)              1,034    id_essentia
audio   MFCC i-vector, 1,024 GMM (A2)            400    id_ivec1024
lyrics  TF-IDF (L1)                            1,000    id_lyrics_tf-idf
lyrics  averaged word2vec (L2)                   300    id_lyrics_word2vec
visual  ResNet frame max+mean (V1)             4,096    id_resnet
visual  Inception-v3 frame max+mean (V2)       4,096    id_incp
======  =====================================  =======  =========

The split, label set, seeds, validation-only selection and thresholding are
identical to the three-view campaigns.  The method recipe (architectures,
dropout rates, MCE components and 0.6/0.4 weights) is transferred without
re-tuning, exactly as in the FMA study.  Seven availability conditions are
predeclared: observed, the three sense-level removals (both views of a sense),
one random view removed, half of the available views removed at random, and a
single random view kept.  Sense-level conditions are comparable with the
three-view protocol, so MCE is additionally compared per label with the frozen
three-view MCE predictions on those conditions.
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
from .full_feature_ablation import holm_by_family
from .fusion_models import AttnFusionTagger, EvidFusionTagger, GenericTagger, evidential_loss, parameter_count
from .optimized_multimodal_tagging import load_full_modality
from .common import environment_record, load_config, resolve_device, set_seed
from .robust_multimodal_tagging import build_dataset, metric_values, summarise, targets, tune_thresholds


CONDITIONS = ["observed", "no_audio", "no_lyrics", "no_visual", "random_one_view_missing", "random_half_views_missing", "keep_one_view"]
SHARED_CONDITIONS = {"observed": "observed", "no_audio": "no_audio", "no_lyrics": "no_lyrics", "no_visual": "no_visual"}
SINGLE_VIEW_MODELS = {
    "AudioEssentiaOnly": 0,
    "AudioIvecOnly": 1,
    "LyricsTfidfOnly": 2,
    "LyricsW2vOnly": 3,
    "VisualResnetOnly": 4,
    "VisualIncpOnly": 5,
}


def load_views(processed_root: Path, split_frames: dict[str, pd.DataFrame], views: list[dict[str, str]]) -> tuple[dict[str, list[np.ndarray]], dict[str, np.ndarray], dict[str, np.ndarray]]:
    names = ["train", "validation", "test"]
    combined = pd.concat([split_frames[name] for name in names], ignore_index=True)
    item_ids = combined["track_id"].astype(str).tolist()
    offsets: dict[str, np.ndarray] = {}
    start = 0
    for name in names:
        offsets[name] = np.arange(start, start + len(split_frames[name]), dtype=np.int64)
        start += len(split_frames[name])
    values_by_view, availability, reliability = [], [], []
    for view in views:
        values, present, quality = load_full_modality(processed_root, str(view["name"]), item_ids, len(split_frames["train"]))
        values_by_view.append(values)
        availability.append(present)
        reliability.append(quality)
        print(f"loaded view={view['name']} dimension={values.shape[1]} coverage={present.mean():.4f}", flush=True)
    features = {name: [values[rows] for values in values_by_view] for name, rows in offsets.items()}
    available = {name: np.stack([values[rows] for values in availability], axis=1) for name, rows in offsets.items()}
    reliable = {name: np.stack([values[rows] for values in reliability], axis=1) for name, rows in offsets.items()}
    return features, available, reliable


def build_model(name: str, input_dimensions: list[int], labels: int, settings: dict[str, Any]) -> nn.Module:
    hidden = int(settings["hidden_dimension"])
    dropout = float(settings["representation_dropout"])
    md = float(settings["modality_dropout"])
    new_md = float(settings["new_model_modality_dropout"])
    if name in SINGLE_VIEW_MODELS:
        return GenericTagger(input_dimensions, hidden_dimension=hidden, labels=labels, fusion="mean", active_modalities=(SINGLE_VIEW_MODELS[name],), representation_dropout=dropout)
    generic = {
        "UniformMean": {"fusion": "mean"},
        "EarlyConcat": {"fusion": "concat"},
        "ConcatModDrop": {"fusion": "concat", "modality_dropout": md},
        "ReliabilityGate": {"fusion": "reliability"},
        "RAMT": {"fusion": "reliability", "modality_dropout": md},
        "ConcatMD10-H384": {"fusion": "concat", "modality_dropout": float(settings["ensemble_component_dropout"]), "hidden": int(settings["ensemble_hidden_dimension"])},
    }
    if name in generic:
        spec = generic[name]
        return GenericTagger(
            input_dimensions,
            hidden_dimension=int(spec.get("hidden", hidden)),
            labels=labels,
            fusion=str(spec["fusion"]),
            modality_dropout=float(spec.get("modality_dropout", 0.0)),
            representation_dropout=dropout,
        )
    if name == "AttnFusion":
        return AttnFusionTagger(input_dimensions, hidden_dimension=hidden, labels=labels, heads=int(settings["attention_heads"]), layers=int(settings["attention_layers"]), modality_dropout=new_md, representation_dropout=dropout)
    if name == "EvidFusion":
        return EvidFusionTagger(input_dimensions, hidden_dimension=hidden, labels=labels, modality_dropout=new_md, representation_dropout=dropout)
    raise ValueError(name)


@torch.no_grad()
def predict(model: nn.Module, features: list[np.ndarray], availability: np.ndarray, reliability: np.ndarray, device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
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


def train_model(name: str, seed: int, features: dict[str, list[np.ndarray]], availability: dict[str, np.ndarray], reliability: dict[str, np.ndarray], target: dict[str, np.ndarray], settings: dict[str, Any], device: torch.device) -> tuple[nn.Module, np.ndarray, pd.DataFrame]:
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
                loss = evidential_loss(combined, per_modality, batch_available, batch_target, annealing=min(1.0, epoch / annealing_epochs))
            else:
                logits, _ = model(batch_features, batch_available, batch_reliability)
                loss = criterion(logits, batch_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_probability, _ = predict(model, features["validation"], availability["validation"], reliability["validation"], device, int(settings["evaluation_batch_size"]))
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


def intervention(name: str, availability: np.ndarray, senses: list[str], seed: int) -> np.ndarray:
    result = availability.copy()
    if name == "observed":
        return result
    if name.startswith("no_"):
        columns = [index for index, sense in enumerate(senses) if sense == name[3:]]
        result[:, columns] = 0.0
        return result
    rng = np.random.default_rng(seed)
    for row in range(len(result)):
        candidates = np.flatnonzero(result[row] > 0)
        if len(candidates) <= 1:
            continue
        if name == "random_one_view_missing":
            result[row, rng.choice(candidates)] = 0.0
        elif name == "random_half_views_missing":
            removed = rng.choice(candidates, size=len(candidates) // 2, replace=False)
            result[row, removed] = 0.0
        elif name == "keep_one_view":
            keep = int(rng.choice(candidates))
            result[row] = 0.0
            result[row, keep] = 1.0
        else:
            raise ValueError(name)
    return result


def efficiency_audit(names: list[str], input_dimensions: list[int], labels: int, settings: dict[str, Any], device: torch.device) -> pd.DataFrame:
    batch = int(settings["efficiency_batch_size"])
    repeats = int(settings["efficiency_repeats"])
    torch.manual_seed(20260724)
    features = [torch.randn(batch, dimension, device=device) for dimension in input_dimensions]
    availability = torch.ones(batch, len(input_dimensions), device=device)
    reliability = torch.ones(batch, len(input_dimensions), device=device)
    rows = []
    for name in names:
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
        rows.append({"model": name, "parameters": int(parameter_count(model)), "batch_size": batch, "median_latency_ms": latency, "throughput_tracks_per_s": batch / (latency / 1000.0), "device": str(device)})
        model.cpu()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def run(project_root: Path, config: dict[str, Any], run_name: str, overwrite: bool) -> Path:
    settings = config["extended_modalities"]
    output_root = project_root / config["project"]["outputs_root"] / "experiments" / run_name
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    device = resolve_device(config["runtime"]["device"])
    processed_root = project_root / config["project"]["processed_root"]
    views = [dict(view) for view in settings["views"]]
    senses = [str(view["sense"]) for view in views]
    split_frames, labels, frequencies = build_dataset(processed_root, int(settings["top_labels"]), int(settings["min_train_tag_frequency"]))
    features, availability, reliability = load_views(processed_root, split_frames, views)
    target = {name: targets(frame, len(labels)) for name, frame in split_frames.items()}
    test_track_ids = split_frames["test"]["track_id"].astype(str).to_numpy()
    models = [str(value) for value in settings["models"]]
    ensemble = settings["ensemble"]
    ensemble_name = str(ensemble["name"])
    components = [str(value) for value in ensemble["components"]]
    weights = np.asarray(ensemble["weights"], dtype=np.float64)
    if set(components) - set(models) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("Invalid locked ensemble configuration")

    metrics_rows, history_frames, gate_rows, threshold_rows = [], [], [], []
    predictions: dict[str, dict[str, list[np.ndarray]]] = {model: {c: [] for c in CONDITIONS} for model in models + [ensemble_name]}
    validation_predictions: dict[str, list[np.ndarray]] = {model: [] for model in models}
    threshold_values: dict[str, list[np.ndarray]] = {model: [] for model in models + [ensemble_name]}
    condition_masks = {condition: intervention(condition, availability["test"], senses, 20260910 + number) for number, condition in enumerate(CONDITIONS)}
    for seed in [int(value) for value in settings["seeds"]]:
        for model_name in models:
            model, thresholds, history = train_model(model_name, seed, features, availability, reliability, target, settings, device)
            history_frames.append(history)
            threshold_values[model_name].append(thresholds)
            for label, value in zip(labels, thresholds, strict=True):
                threshold_rows.append({"model": model_name, "seed": seed, "label": label, "threshold": float(value)})
            validation_probability, _ = predict(model, features["validation"], availability["validation"], reliability["validation"], device, int(settings["evaluation_batch_size"]))
            validation_predictions[model_name].append(validation_probability)
            for condition in CONDITIONS:
                probability, gates = predict(model, features["test"], condition_masks[condition], reliability["test"], device, int(settings["evaluation_batch_size"]))
                predictions[model_name][condition].append(probability)
                metrics_rows.append({"model": model_name, "seed": seed, "condition": condition, **metric_values(target["test"], probability, thresholds)})
                if model_name in ("ReliabilityGate", "RAMT", "AttnFusion", "UniformMean"):
                    for index, view in enumerate(views):
                        valid = condition_masks[condition][:, index] > 0
                        gate_rows.append({"model": model_name, "seed": seed, "condition": condition, "view": str(view["name"]), "sense": senses[index], "mean_gate": float(gates[valid, index].mean()) if valid.any() else float("nan"), "available_tracks": int(valid.sum())})
            print(f"complete model={model_name} seed={seed} best_validation_mAP={history['validation_mAP'].max():.5f}", flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        seed_index = len(threshold_values[ensemble_name])
        validation_probability = sum(float(weight) * validation_predictions[component][seed_index] for component, weight in zip(components, weights, strict=True))
        ensemble_thresholds = tune_thresholds(target["validation"], validation_probability)
        threshold_values[ensemble_name].append(ensemble_thresholds)
        for label, value in zip(labels, ensemble_thresholds, strict=True):
            threshold_rows.append({"model": ensemble_name, "seed": seed, "label": label, "threshold": float(value)})
        for condition in CONDITIONS:
            probability = sum(float(weight) * predictions[component][condition][seed_index] for component, weight in zip(components, weights, strict=True))
            predictions[ensemble_name][condition].append(probability)
            metrics_rows.append({"model": ensemble_name, "seed": seed, "condition": condition, **metric_values(target["test"], probability, ensemble_thresholds)})

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(output_root / "metrics_per_seed_condition.csv", index=False)
    summarise(metrics).to_csv(output_root / "metrics_summary.csv", index=False)
    pd.concat(history_frames, ignore_index=True).to_csv(output_root / "training_history.csv", index=False)
    pd.DataFrame(gate_rows).to_csv(output_root / "gate_weights.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(output_root / "validation_thresholds.csv", index=False)

    mean_probabilities = {model: {condition: np.mean(values, axis=0) for condition, values in condition_values.items()} for model, condition_values in predictions.items()}
    mean_thresholds = {model: np.mean(values, axis=0) for model, values in threshold_values.items()}

    # Frozen three-view MCE on the shared sense-level conditions.
    reference_root = project_root / str(settings["reference_run"])
    reference_model = str(settings["reference_model"])
    reference_name = f"{reference_model}@3view"
    frozen = load_reference_predictions(reference_root, [reference_model], labels, test_track_ids)[reference_model]
    stored = pd.read_csv(reference_root / "validation_thresholds.csv")
    pivot = stored.loc[stored["model"] == reference_model].pivot_table(index="seed", columns="label", values="threshold")[labels]
    reference_thresholds = pivot.to_numpy(dtype=np.float32).mean(axis=0)

    per_label_rows, test_rows = [], []
    for condition in CONDITIONS:
        ap_cache, f1_cache = {}, {}
        candidates = dict(mean_probabilities)
        thresholds_by_model = dict(mean_thresholds)
        if condition in SHARED_CONDITIONS:
            candidates[reference_name] = {condition: frozen[SHARED_CONDITIONS[condition]]}
            thresholds_by_model[reference_name] = reference_thresholds
        for model, condition_values in candidates.items():
            probability = condition_values[condition]
            predicted = probability >= thresholds_by_model[model][None, :]
            ap_cache[model] = np.asarray([average_precision_score(target["test"][:, label], probability[:, label]) for label in range(len(labels))])
            f1_cache[model] = np.asarray([f1_score(target["test"][:, label], predicted[:, label], zero_division=0) for label in range(len(labels))])
            for label_id, label in enumerate(labels):
                per_label_rows.append({"model": model, "condition": condition, "label_id": label_id, "label": label, "AP": float(ap_cache[model][label_id]), "F1": float(f1_cache[model][label_id])})
        for baseline in candidates:
            if baseline == ensemble_name:
                continue
            for metric, cache in (("AP", ap_cache), ("F1", f1_cache)):
                delta = cache[ensemble_name] - cache[baseline]
                low, high, p_value = _bootstrap(delta, 20261201 + len(test_rows), int(settings["bootstrap_replicates"]))
                test_rows.append({"condition": condition, "metric": metric, "proposed": ensemble_name, "baseline": baseline, "mean_delta": float(delta.mean()), "ci95_low": low, "ci95_high": high, "p_value": p_value})
    pd.DataFrame(per_label_rows).to_csv(output_root / "per_label_metrics.csv", index=False)
    holm_by_family(pd.DataFrame(test_rows)).to_csv(output_root / "paired_label_bootstrap_holm.csv", index=False)

    prediction_frames = []
    for model, condition_values in mean_probabilities.items():
        for condition, probability in condition_values.items():
            frame = pd.DataFrame(probability, columns=[f"score_{label}" for label in labels])
            frame.insert(0, "track_id", test_track_ids)
            frame.insert(0, "condition", condition)
            frame.insert(0, "model", model)
            prediction_frames.append(frame)
    pd.concat(prediction_frames, ignore_index=True).to_parquet(output_root / "mean_test_predictions_all_conditions.parquet", index=False)

    fusion_models = [name for name in models if name not in SINGLE_VIEW_MODELS]
    efficiency = efficiency_audit(fusion_models, [values.shape[1] for values in features["train"]], len(labels), settings, device)
    component_rows = efficiency.set_index("model").loc[components]
    efficiency = pd.concat(
        [efficiency, pd.DataFrame([{"model": ensemble_name, "parameters": int(component_rows["parameters"].sum()), "batch_size": int(component_rows["batch_size"].iloc[0]), "median_latency_ms": float(component_rows["median_latency_ms"].sum()), "throughput_tracks_per_s": float(component_rows["batch_size"].iloc[0] / (component_rows["median_latency_ms"].sum() / 1000.0)), "device": str(device)}])],
        ignore_index=True,
    )
    efficiency.to_csv(output_root / "efficiency.csv", index=False)

    (output_root / "labels.json").write_text(json.dumps({"labels": labels, "train_frequencies": {label: frequencies[label] for label in labels}}, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    manifest = {
        "environment": environment_record(device),
        "task": "E7 six-view extended-modality artist-disjoint 50-label tagging (locked recipe)",
        "samples": {name: len(frame) for name, frame in split_frames.items()},
        "artists": {name: int(frame["artist_id"].nunique()) for name, frame in split_frames.items()},
        "views": [{"name": str(view["name"]), "sense": str(view["sense"]), "dimension": int(features["train"][index].shape[1])} for index, view in enumerate(views)],
        "coverage": {split: {str(views[index]["name"]): float(availability[split][:, index].mean()) for index in range(len(views))} for split in split_frames},
        "conditions": CONDITIONS,
        "shared_conditions_with_three_view_protocol": list(SHARED_CONDITIONS),
        "models": models,
        "ensemble": ensemble,
        "reference": reference_name,
        "recipe": "architectures, dropout rates and 0.6/0.4 ensemble weights locked from the three-view Onion study; thresholds calibrated on validation only",
        "statistical_unit": "label; mean prediction across seeds; paired bootstrap with Holm within condition and metric",
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
