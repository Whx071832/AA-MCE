"""E8: availability-aware multi-capacity ensemble (AA-MCE) on the frozen Onion protocol.

The locked MCE mixes two full-feature concatenation taggers of different
capacity with fixed 0.6/0.4 weights and uses per-label thresholds calibrated
on the fully observed validation data.  Under the most severe availability
condition (one modality kept) its ranking advantage vanished and the gated
RAMT retained better Macro-F1.  AA-MCE exploits the fact that the set of
available modalities is *known at inference time*:

* **availability-conditioned capacity mixing** -- for each of the seven
  non-empty availability patterns the mixing weights over the candidate
  components (H384/MD10, H192/MD30 and the gated full-feature RAMT) are
  selected on validation tracks masked to that pattern;
* **availability-conditioned thresholds** -- per-label decision thresholds
  are calibrated on validation tracks masked to the same pattern.

Both selections use validation artists only; the locked 0.6/0.4/0.0 mixture is
kept unless the validation gain exceeds a predeclared margin.  Ablations
separate the two devices (thresholds only, weights only, two vs three
components), and the strongest baselines (RAMT, AttnFusion, EvidFusion) are
given the same availability-conditioned thresholds as controls so that F1
comparisons remain like-for-like.  The two MCE components are re-used from the
frozen checkpoints of ``optimized_multimodal_tagging_final_v1``; RAMT,
AttnFusion and EvidFusion are retrained with the recipes of E1-F/E2 and their
checkpoints are persisted.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, f1_score
from torch import nn

from .advanced_fusion_baselines import build_model as build_recent_model
from .advanced_fusion_baselines import load_reference_predictions
from .advanced_fusion_baselines import train_model as train_recent_model
from .common import _bootstrap
from .full_feature_ablation import build_model as build_generic_model
from .full_feature_ablation import holm_by_family
from .full_feature_ablation import train_model as train_generic_model
from .optimized_multimodal_tagging import CONDITIONS, FullFeatureTagger, load_full_features
from .common import environment_record, load_config, resolve_device, set_seed
from .robust_multimodal_tagging import MODALITIES, build_dataset, intervention, metric_values, summarise, targets, tune_thresholds


PATTERNS: list[tuple[int, int, int]] = [p for p in itertools.product((1, 0), repeat=3) if sum(p) > 0]  # 7 non-empty
COMPONENT_SPECS = {
    "FullConcatMD10-H384-BCE": {"hidden_dimension": 384, "modality_dropout": 0.10},
    "FullConcatMD-BCE": {"hidden_dimension": 192, "modality_dropout": 0.30},
}


def pattern_name(pattern: tuple[int, int, int]) -> str:
    return "+".join(MODALITIES[index] for index, flag in enumerate(pattern) if flag) or "none"


def pattern_index(availability: np.ndarray) -> np.ndarray:
    """Map each availability row to its index in ``PATTERNS`` (-1 if empty)."""
    lookup = {pattern: index for index, pattern in enumerate(PATTERNS)}
    rows = availability.astype(int)
    return np.asarray([lookup.get(tuple(int(v) for v in row), -1) for row in rows], dtype=np.int64)


@torch.no_grad()
def predict_any(model: nn.Module, features: list[np.ndarray], availability: np.ndarray, reliability: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    outputs = []
    for start in range(0, len(availability), batch_size):
        stop = min(start + batch_size, len(availability))
        batch = [torch.as_tensor(values[start:stop], device=device) for values in features]
        available = torch.as_tensor(availability[start:stop], device=device)
        reliable = torch.as_tensor(reliability[start:stop], device=device)
        result = model(batch, available, reliable)
        logits = result[0] if isinstance(result, tuple) else result
        outputs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(outputs)


def load_component(name: str, seed: int, input_dimensions: list[int], labels: int, settings: dict[str, Any], device: torch.device) -> nn.Module:
    spec = COMPONENT_SPECS[name]
    model = FullFeatureTagger(
        input_dimensions=input_dimensions,
        hidden_dimension=int(spec["hidden_dimension"]),
        labels=labels,
        fusion="concat",
        modality_dropout=float(spec["modality_dropout"]),
        representation_dropout=float(settings["representation_dropout"]),
        label_refinement=False,
    )
    path = Path(settings["component_run"]) / "checkpoints" / f"{name}_{seed}.pt"
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.to(device)


def simplex_grid(count: int, step: float) -> list[tuple[float, ...]]:
    units = int(round(1.0 / step))
    grid = []
    for combo in itertools.product(range(units + 1), repeat=count):
        if sum(combo) == units:
            grid.append(tuple(round(value / units, 6) for value in combo))
    return grid


def mix(probabilities: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    return sum(float(weight) * probabilities[name] for name, weight in weights.items() if weight > 0)


def thresholds_by_track(pattern_thresholds: dict[int, np.ndarray], track_patterns: np.ndarray, labels: int) -> np.ndarray:
    result = np.full((len(track_patterns), labels), 0.5, dtype=np.float32)
    for index, thresholds in pattern_thresholds.items():
        result[track_patterns == index] = thresholds[None, :]
    return result


def metric_values_tracked(target: np.ndarray, probability: np.ndarray, thresholds: np.ndarray) -> dict[str, float]:
    """Same metrics as ``metric_values`` but with a per-track threshold matrix."""
    from sklearn.metrics import roc_auc_score

    predicted = probability >= thresholds
    valid_auc = np.logical_and(target.sum(axis=0) > 0, target.sum(axis=0) < len(target))
    return {
        "mAP": float(average_precision_score(target, probability, average="macro")),
        "MacroF1": float(f1_score(target, predicted, average="macro", zero_division=0)),
        "MicroF1": float(f1_score(target, predicted, average="micro", zero_division=0)),
        "MacroROC_AUC": float(roc_auc_score(target[:, valid_auc], probability[:, valid_auc], average="macro")),
    }


def run(project_root: Path, config: dict[str, Any], run_name: str, overwrite: bool, section: str = "availability_aware_ensemble") -> Path:
    settings = dict(config[section])
    settings["component_run"] = str(project_root / settings["component_run"])
    reuse_root = project_root / str(settings["reuse_checkpoints_from"]) / "checkpoints" if settings.get("reuse_checkpoints_from") else None
    output_root = project_root / config["project"]["outputs_root"] / "experiments" / run_name
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root = output_root / "checkpoints"
    checkpoint_root.mkdir(exist_ok=True)
    started = time.perf_counter()
    device = resolve_device(config["runtime"]["device"])
    processed_root = project_root / config["project"]["processed_root"]
    split_frames, labels, frequencies = build_dataset(processed_root, int(settings["top_labels"]), int(settings["min_train_tag_frequency"]))
    features, availability, reliability = load_full_features(processed_root, split_frames, include_test=True)
    target = {name: targets(frame, len(labels)) for name, frame in split_frames.items()}
    input_dimensions = [values.shape[1] for values in features["train"]]
    test_track_ids = split_frames["test"]["track_id"].astype(str).to_numpy()
    seeds = [int(value) for value in settings["seeds"]]
    components = [str(value) for value in settings["candidate_components"]]
    controls = [str(value) for value in settings["control_models"]]
    recent_settings = dict(settings, modality_dropout=float(settings["new_model_modality_dropout"]))
    eval_batch = int(settings["evaluation_batch_size"])

    # Validation rows eligible for each pattern (all modalities of the pattern originally present).
    validation_available = availability["validation"]
    eligible = {index: np.flatnonzero((validation_available >= np.asarray(pattern, dtype=np.float32)).all(axis=1)) for index, pattern in enumerate(PATTERNS)}
    masked_validation = {index: np.tile(np.asarray(pattern, dtype=np.float32), (len(validation_available), 1)) for index, pattern in enumerate(PATTERNS)}

    # --- per-seed predictions ------------------------------------------------
    validation_predictions: dict[int, dict[int, dict[str, np.ndarray]]] = {}  # seed -> pattern -> model -> probs (eligible rows)
    test_predictions: dict[int, dict[str, dict[str, np.ndarray]]] = {}  # seed -> condition -> model -> probs
    test_patterns: dict[str, np.ndarray] = {}
    history_frames = []
    for seed in seeds:
        models: dict[str, nn.Module] = {}
        for name in components:
            if name in COMPONENT_SPECS:
                models[name] = load_component(name, seed, input_dimensions, len(labels), settings, device)
        trained_names = [name for name in components + controls if name not in COMPONENT_SPECS]
        for name in dict.fromkeys(trained_names):
            reusable = reuse_root / f"{name}_{seed}.pt" if reuse_root is not None else None
            if reusable is not None and reusable.exists():
                if name == "RAMT":
                    model = build_generic_model(name, input_dimensions, len(labels), settings)
                else:
                    model = build_recent_model(name, input_dimensions, len(labels), recent_settings)
                model.load_state_dict(torch.load(reusable, map_location="cpu", weights_only=True))
                model = model.to(device)
                print(f"reused model={name} seed={seed} from {reusable.parent.parent.name}", flush=True)
            else:
                if name == "RAMT":
                    model, _, history = train_generic_model(name, seed, features, availability, reliability, target, settings, device)
                else:
                    model, _, history = train_recent_model(name, seed, features, availability, reliability, target, recent_settings, device)
                history_frames.append(history)
                torch.save(model.state_dict(), checkpoint_root / f"{name}_{seed}.pt")
                print(f"trained model={name} seed={seed} best_validation_mAP={history['validation_mAP'].max():.5f}", flush=True)
            models[name] = model
        validation_predictions[seed] = {}
        for index in range(len(PATTERNS)):
            rows = eligible[index]
            validation_predictions[seed][index] = {
                name: predict_any(model, [values[rows] for values in features["validation"]], masked_validation[index][rows], reliability["validation"][rows], device, eval_batch)
                for name, model in models.items()
            }
        test_predictions[seed] = {}
        for number, condition in enumerate(CONDITIONS):
            condition_availability = intervention(condition, availability["test"], 20260910 + number)
            if condition not in test_patterns:
                # A handful of test tracks lack every modality under no_lyrics
                # (originally missing audio and visual); they fall back to the
                # fully observed pattern's weights and thresholds.
                indices = pattern_index(condition_availability)
                test_patterns[condition] = np.where(indices < 0, PATTERNS.index((1, 1, 1)), indices)
            test_predictions[seed][condition] = {
                name: predict_any(model, features["test"], condition_availability, reliability["test"], device, eval_batch) for name, model in models.items()
            }
        print(f"predictions complete seed={seed}", flush=True)
        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if history_frames:
        pd.concat(history_frames, ignore_index=True).to_csv(output_root / "training_history.csv", index=False)

    # --- validation-only selection of pattern weights ------------------------
    locked = {name: float(value) for name, value in zip(settings["locked_components"], settings["locked_weights"], strict=True)}
    margin = float(settings["selection_margin"])
    grid = simplex_grid(len(components), float(settings["weight_step"]))
    grid_rows, weight_rows = [], []
    pattern_weights: dict[str, dict[int, dict[str, float]]] = {"AA-MCE": {}, "AA-MCE-2": {}}
    for index, pattern in enumerate(PATTERNS):
        rows = eligible[index]
        pattern_target = target["validation"][rows]
        scores: dict[tuple[float, ...], float] = {}
        for combo in grid:
            weights = dict(zip(components, combo, strict=True))
            values = [float(average_precision_score(pattern_target, mix(validation_predictions[seed][index], weights), average="macro")) for seed in seeds]
            scores[combo] = float(np.mean(values))
            grid_rows.append({"pattern": pattern_name(pattern), **{f"w_{name}": weight for name, weight in weights.items()}, "validation_mAP": scores[combo]})
        locked_combo = tuple(locked.get(name, 0.0) for name in components)
        locked_score = scores[locked_combo]
        for variant, allowed in (("AA-MCE", lambda combo: True), ("AA-MCE-2", lambda combo: all(w == 0 for name, w in zip(components, combo, strict=True) if name not in locked))):
            best_combo = max((combo for combo in grid if allowed(combo)), key=lambda combo: scores[combo])
            chosen = best_combo if scores[best_combo] - locked_score > margin else locked_combo
            pattern_weights[variant][index] = dict(zip(components, chosen, strict=True))
            weight_rows.append(
                {
                    "variant": variant,
                    "pattern": pattern_name(pattern),
                    "eligible_validation_tracks": int(len(rows)),
                    **{f"w_{name}": weight for name, weight in pattern_weights[variant][index].items()},
                    "validation_mAP": scores[chosen],
                    "locked_validation_mAP": locked_score,
                    "best_grid_validation_mAP": scores[best_combo],
                    "deviates_from_locked": bool(chosen != locked_combo),
                }
            )
    pd.DataFrame(grid_rows).to_csv(output_root / "pattern_weight_grid.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(output_root / "pattern_weights.csv", index=False)

    # --- variants ------------------------------------------------------------
    def locked_weights_for(_: int) -> dict[str, float]:
        return {name: locked.get(name, 0.0) for name in components}

    variants: dict[str, dict[str, Any]] = {
        "MCE": {"weights": locked_weights_for, "pattern_thresholds": False},
        "MCE+AAT": {"weights": locked_weights_for, "pattern_thresholds": True},
        "AA-MCE-w": {"weights": lambda index: pattern_weights["AA-MCE"][index], "pattern_thresholds": False},
        "AA-MCE-2": {"weights": lambda index: pattern_weights["AA-MCE-2"][index], "pattern_thresholds": True},
        "AA-MCE": {"weights": lambda index: pattern_weights["AA-MCE"][index], "pattern_thresholds": True},
    }
    for name in controls:
        variants[name] = {"single": name, "pattern_thresholds": False}
        variants[f"{name}+AAT"] = {"single": name, "pattern_thresholds": True}

    observed_index = PATTERNS.index((1, 1, 1))

    def variant_probability(spec: dict[str, Any], predictions: dict[str, np.ndarray], index: int) -> np.ndarray:
        if "single" in spec:
            return predictions[spec["single"]]
        return mix(predictions, spec["weights"](index))

    metrics_rows, threshold_rows = [], []
    seed_thresholds: dict[str, dict[int, list[np.ndarray]]] = {name: {index: [] for index in range(len(PATTERNS))} for name in variants}
    seed_test: dict[str, dict[str, list[np.ndarray]]] = {name: {condition: [] for condition in CONDITIONS} for name in variants}
    for seed in seeds:
        for name, spec in variants.items():
            per_pattern: dict[int, np.ndarray] = {}
            for index in range(len(PATTERNS)):
                rows = eligible[index]
                probability = variant_probability(spec, validation_predictions[seed][index], index)
                per_pattern[index] = tune_thresholds(target["validation"][rows], probability)
                seed_thresholds[name][index].append(per_pattern[index])
                for label, value in zip(labels, per_pattern[index], strict=True):
                    threshold_rows.append({"variant": name, "seed": seed, "pattern": pattern_name(PATTERNS[index]), "label": label, "threshold": float(value)})
            for condition in CONDITIONS:
                track_pattern = test_patterns[condition]
                if spec["pattern_thresholds"]:
                    # Mixed probability per track follows the track's own pattern.
                    probability = np.zeros((len(track_pattern), len(labels)), dtype=np.float32)
                    for index in np.unique(track_pattern):
                        selected = track_pattern == index
                        probability[selected] = variant_probability(spec, {k: v[selected] for k, v in test_predictions[seed][condition].items()}, int(index))
                    thresholds = thresholds_by_track(per_pattern, track_pattern, len(labels))
                else:
                    probability = variant_probability(spec, test_predictions[seed][condition], observed_index)
                    thresholds = np.tile(per_pattern[observed_index][None, :], (len(track_pattern), 1))
                seed_test[name][condition].append(probability)
                metrics_rows.append({"model": name, "seed": seed, "condition": condition, **metric_values_tracked(target["test"], probability, thresholds)})
    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(output_root / "metrics_per_seed_condition.csv", index=False)
    summarise(metrics).to_csv(output_root / "metrics_summary.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(output_root / "pattern_thresholds.csv", index=False)

    # --- reproduction check against the frozen runs ---------------------------
    mean_test = {name: {condition: np.mean(values, axis=0) for condition, values in conditions.items()} for name, conditions in seed_test.items()}
    checks: dict[str, Any] = {}
    frozen_mce = load_reference_predictions(project_root / str(settings["reference_run"]), ["OptimizedEnsemble"], labels, test_track_ids)["OptimizedEnsemble"]
    checks["MCE_vs_frozen_max_abs_diff"] = float(max(np.abs(mean_test["MCE"][condition] - frozen_mce[condition]).max() for condition in CONDITIONS))
    recent_root = project_root / str(settings["recent_reference_run"])
    if recent_root.exists():
        frozen_recent = load_reference_predictions(recent_root, [name for name in controls if name in ("AttnFusion", "EvidFusion")], labels, test_track_ids)
        for name, conditions in frozen_recent.items():
            checks[f"{name}_vs_frozen_max_abs_diff"] = float(max(np.abs(mean_test[name][condition] - conditions[condition]).max() for condition in CONDITIONS))
    (output_root / "reproduction_check.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")

    # --- paired label tests --------------------------------------------------
    mean_thresholds = {name: {index: np.mean(values, axis=0) for index, values in patterns.items()} for name, patterns in seed_thresholds.items()}
    per_label_rows, test_rows = [], []
    families = {proposed: [name for name in variants if name != proposed] for proposed in settings["test_proposed"]}
    for condition in CONDITIONS:
        track_pattern = test_patterns[condition]
        ap_cache, f1_cache = {}, {}
        for name, spec in variants.items():
            probability = mean_test[name][condition]
            if spec["pattern_thresholds"]:
                thresholds = thresholds_by_track(mean_thresholds[name], track_pattern, len(labels))
            else:
                thresholds = np.tile(mean_thresholds[name][observed_index][None, :], (len(track_pattern), 1))
            predicted = probability >= thresholds
            ap_cache[name] = np.asarray([average_precision_score(target["test"][:, label], probability[:, label]) for label in range(len(labels))])
            f1_cache[name] = np.asarray([f1_score(target["test"][:, label], predicted[:, label], zero_division=0) for label in range(len(labels))])
            for label_id, label in enumerate(labels):
                per_label_rows.append({"model": name, "condition": condition, "label_id": label_id, "label": label, "AP": float(ap_cache[name][label_id]), "F1": float(f1_cache[name][label_id])})
        for proposed, baselines in families.items():
            for baseline in baselines:
                for metric, cache in (("AP", ap_cache), ("F1", f1_cache)):
                    delta = cache[proposed] - cache[baseline]
                    low, high, p_value = _bootstrap(delta, 20261101 + len(test_rows), int(settings["bootstrap_replicates"]))
                    test_rows.append({"condition": condition, "metric": metric, "proposed": proposed, "baseline": baseline, "mean_delta": float(delta.mean()), "ci95_low": low, "ci95_high": high, "p_value": p_value})
    pd.DataFrame(per_label_rows).to_csv(output_root / "per_label_metrics.csv", index=False)
    holm_by_family(pd.DataFrame(test_rows)).to_csv(output_root / "paired_label_bootstrap_holm.csv", index=False)

    prediction_frames = []
    for name in variants:
        for condition in CONDITIONS:
            frame = pd.DataFrame(mean_test[name][condition], columns=[f"score_{label}" for label in labels])
            frame.insert(0, "track_id", test_track_ids)
            frame.insert(0, "condition", condition)
            frame.insert(0, "model", name)
            prediction_frames.append(frame)
    pd.concat(prediction_frames, ignore_index=True).to_parquet(output_root / "mean_test_predictions_all_conditions.parquet", index=False)

    (output_root / "labels.json").write_text(json.dumps({"labels": labels, "train_frequencies": {label: frequencies[label] for label in labels}}, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    manifest = {
        "environment": environment_record(device),
        "task": "E8 availability-aware multi-capacity ensemble on frozen artist-disjoint 50-label tagging",
        "samples": {name: len(frame) for name, frame in split_frames.items()},
        "patterns": [pattern_name(pattern) for pattern in PATTERNS],
        "eligible_validation_tracks": {pattern_name(PATTERNS[index]): int(len(rows)) for index, rows in eligible.items()},
        "candidate_components": components,
        "controls": controls,
        "variants": list(variants),
        "selection": "pattern weights: validation mAP averaged over seeds, locked 0.6/0.4/0.0 kept unless gain > margin; thresholds: per-label F1 on validation rows masked to the pattern",
        "selection_margin": margin,
        "reproduction_check": checks,
        "statistical_unit": "label; mean prediction across seeds; paired bootstrap with Holm within (proposed, condition, metric)",
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
    parser.add_argument("--section", default="availability_aware_ensemble", help="configuration section (e.g. availability_aware_ensemble_5c)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.project_root, load_config(args.project_root, args.config), args.run_name, args.overwrite, section=args.section)


if __name__ == "__main__":
    main()
