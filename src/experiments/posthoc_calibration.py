"""Post-hoc probability calibration for the Onion model family (E5 follow-up).

The calibration audit showed that the evidential baseline is the best
calibrated model while MCE trades calibration for ranking accuracy.  This
runner asks whether a validation-fitted, per-label Platt scaling closes that
gap without touching the ranking (Platt scaling is monotone per label, so AP
is unchanged by construction).  Two calibration protocols are compared:

* ``cal-obs``: Platt parameters fitted on the fully observed validation data
  and applied under every test condition (the conventional protocol);
* ``cal-AA``: availability-aware calibration -- parameters fitted on
  validation tracks masked to each availability pattern and applied
  according to the pattern of each test track (the AA-MCE philosophy).

Every model receives the same treatment.  Models: the two MCE components
(frozen checkpoints), RAMT, AttnFusion and EvidFusion (E8 checkpoints), the
locked MCE mixture and the AA-MCE mixture (pattern weights from E8).
Calibration is fitted per seed; metrics are computed on the five-seed mean
of the calibrated probabilities, as in the calibration audit.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from .advanced_fusion_baselines import build_model as build_recent_model
from .availability_aware_ensemble import PATTERNS, load_component, pattern_index, pattern_name, predict_any
from .common import _bootstrap
from .full_feature_ablation import build_model as build_generic_model
from .optimized_multimodal_tagging import CONDITIONS, load_full_features
from .paper_analysis import brier_by_label, ece_by_label, holm, log_loss
from .common import environment_record, load_config, resolve_device
from .robust_multimodal_tagging import build_dataset, intervention, targets


EPSILON = 1e-6


def logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, EPSILON, 1.0 - EPSILON)
    return np.log(clipped) - np.log1p(-clipped)


def fit_platt(target: np.ndarray, probability: np.ndarray) -> np.ndarray:
    """Per-label (a, b) of sigmoid(a * logit(p) + b) fitted by logistic regression."""
    parameters = np.zeros((target.shape[1], 2), dtype=np.float64)
    features = logit(probability)
    for label in range(target.shape[1]):
        y = target[:, label]
        if y.sum() == 0 or y.sum() == len(y):
            parameters[label] = (1.0, 0.0)
            continue
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500)
        model.fit(features[:, label : label + 1], y)
        parameters[label] = (float(model.coef_[0, 0]), float(model.intercept_[0]))
    return parameters


def apply_platt(probability: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    z = logit(probability) * parameters[None, :, 0] + parameters[None, :, 1]
    return 1.0 / (1.0 + np.exp(-z))


def run(project_root: Path, config: dict[str, Any], run_name: str, replicates: int) -> Path:
    settings = dict(config["availability_aware_ensemble"])
    settings["component_run"] = str(project_root / settings["component_run"])
    aa_root = project_root / str(settings["aa_run"]) if settings.get("aa_run") else project_root / "outputs" / "experiments" / "availability_aware_ensemble_v1"
    output_root = project_root / config["project"]["outputs_root"] / "experiments" / run_name
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    device = resolve_device(config["runtime"]["device"])
    processed_root = project_root / config["project"]["processed_root"]
    split_frames, labels, _ = build_dataset(processed_root, int(settings["top_labels"]), int(settings["min_train_tag_frequency"]))
    features, availability, reliability = load_full_features(processed_root, split_frames, include_test=True)
    target = {name: targets(frame, len(labels)).astype(np.float64) for name, frame in split_frames.items()}
    input_dimensions = [values.shape[1] for values in features["train"]]
    seeds = [int(value) for value in settings["seeds"]]
    recent_settings = dict(settings, modality_dropout=float(settings["new_model_modality_dropout"]))
    eval_batch = int(settings["evaluation_batch_size"])
    base_models = ["FullConcatMD10-H384-BCE", "FullConcatMD-BCE", "RAMT", "AttnFusion", "EvidFusion"]
    locked = {name: float(value) for name, value in zip(settings["locked_components"], settings["locked_weights"], strict=True)}
    weights = pd.read_csv(aa_root / "pattern_weights.csv")
    weights = weights.loc[weights["variant"] == "AA-MCE"].set_index("pattern")
    components = [str(value) for value in settings["candidate_components"]]
    aa_weights = {index: {name: float(weights.loc[pattern_name(pattern), f"w_{name}"]) for name in components} for index, pattern in enumerate(PATTERNS)}
    observed_index = PATTERNS.index((1, 1, 1))

    validation_available = availability["validation"]
    eligible = {index: np.flatnonzero((validation_available >= np.asarray(pattern, dtype=np.float32)).all(axis=1)) for index, pattern in enumerate(PATTERNS)}
    masked = {index: np.tile(np.asarray(pattern, dtype=np.float32), (len(validation_available), 1)) for index, pattern in enumerate(PATTERNS)}
    condition_masks = {condition: intervention(condition, availability["test"], 20260910 + number) for number, condition in enumerate(CONDITIONS)}
    test_patterns = {}
    for condition, mask in condition_masks.items():
        indices = pattern_index(mask)
        test_patterns[condition] = np.where(indices < 0, observed_index, indices)

    def mixtures(predictions: dict[str, np.ndarray], index: int) -> dict[str, np.ndarray]:
        result = dict(predictions)
        result["MCE"] = sum(locked[name] * predictions[name] for name in locked)
        result["AA-MCE"] = sum(weight * predictions[name] for name, weight in aa_weights[index].items() if weight > 0)
        return result

    variants = ["raw", "cal-obs", "cal-AA"]
    model_names = base_models + ["MCE", "AA-MCE"]
    accumulated: dict[tuple[str, str, str], list[np.ndarray]] = {}
    for seed in seeds:
        models = {}
        for name in base_models:
            if name in ("FullConcatMD10-H384-BCE", "FullConcatMD-BCE"):
                models[name] = load_component(name, seed, input_dimensions, len(labels), settings, device)
            else:
                checkpoint = aa_root / "checkpoints" / f"{name}_{seed}.pt"
                model = build_generic_model(name, input_dimensions, len(labels), settings) if name == "RAMT" else build_recent_model(name, input_dimensions, len(labels), recent_settings)
                model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
                models[name] = model.to(device)
        # Validation predictions per pattern and Platt parameters per model/pattern.
        platt: dict[str, dict[int, np.ndarray]] = {name: {} for name in model_names}
        for index in range(len(PATTERNS)):
            rows = eligible[index]
            predictions = {name: predict_any(model, [values[rows] for values in features["validation"]], masked[index][rows], reliability["validation"][rows], device, eval_batch) for name, model in models.items()}
            mixed = mixtures(predictions, index)
            for name in model_names:
                platt[name][index] = fit_platt(target["validation"][rows], mixed[name])
        for condition in CONDITIONS:
            track_pattern = test_patterns[condition]
            predictions = {name: predict_any(model, features["test"], condition_masks[condition], reliability["test"], device, eval_batch) for name, model in models.items()}
            mixed_by_pattern: dict[str, np.ndarray] = {name: np.zeros((len(track_pattern), len(labels)), dtype=np.float64) for name in model_names}
            for index in np.unique(track_pattern):
                selected = track_pattern == index
                mixed = mixtures({k: v[selected] for k, v in predictions.items()}, int(index))
                for name in model_names:
                    mixed_by_pattern[name][selected] = mixed[name]
            for name in model_names:
                raw = mixed_by_pattern[name]
                cal_obs = apply_platt(raw, platt[name][observed_index])
                cal_aa = np.zeros_like(raw)
                for index in np.unique(track_pattern):
                    selected = track_pattern == index
                    cal_aa[selected] = apply_platt(raw[selected], platt[name][int(index)])
                for variant, values in (("raw", raw), ("cal-obs", cal_obs), ("cal-AA", cal_aa)):
                    accumulated.setdefault((name, variant, condition), []).append(values)
        print(f"calibration complete seed={seed}", flush=True)
        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()

    mean_probability = {key: np.mean(values, axis=0) for key, values in accumulated.items()}
    metric_rows, cache = [], {}
    for (name, variant, condition), probability in mean_probability.items():
        ece = ece_by_label(target["test"], probability)
        brier = brier_by_label(target["test"], probability)
        ap = np.asarray([__import__("sklearn.metrics", fromlist=["average_precision_score"]).average_precision_score(target["test"][:, label], probability[:, label]) for label in range(len(labels))])
        cache[(name, variant, condition)] = {"ECE": ece, "Brier": brier, "AP": ap}
        metric_rows.append({"model": name, "variant": variant, "condition": condition, "MacroECE15": float(ece.mean()), "Brier": float(brier.mean()), "LogLoss": log_loss(target["test"], probability), "mAP": float(ap.mean())})
    pd.DataFrame(metric_rows).to_csv(output_root / "calibration_metrics.csv", index=False)

    test_rows = []
    for condition in CONDITIONS:
        # (a) does calibration help each model?  (b) AA-MCE cal-AA vs every other model cal-AA.
        for name in model_names:
            for variant in ("cal-obs", "cal-AA"):
                for metric in ("ECE", "Brier"):
                    delta = cache[(name, "raw", condition)][metric] - cache[(name, variant, condition)][metric]
                    low, high, p_value = _bootstrap(delta, 20261701 + len(test_rows), replicates)
                    test_rows.append({"family": f"{variant}_vs_raw", "condition": condition, "metric": metric, "proposed": f"{name}/{variant}", "baseline": f"{name}/raw", "improvement_positive_is_better": float(delta.mean()), "ci95_low": low, "ci95_high": high, "p_value": p_value})
        for name in model_names:
            if name == "AA-MCE":
                continue
            for metric in ("ECE", "Brier"):
                delta = cache[(name, "cal-AA", condition)][metric] - cache[("AA-MCE", "cal-AA", condition)][metric]
                low, high, p_value = _bootstrap(delta, 20261701 + len(test_rows), replicates)
                test_rows.append({"family": "AA-MCE_cal-AA_vs_others_cal-AA", "condition": condition, "metric": metric, "proposed": "AA-MCE/cal-AA", "baseline": f"{name}/cal-AA", "improvement_positive_is_better": float(delta.mean()), "ci95_low": low, "ci95_high": high, "p_value": p_value})
    holm(pd.DataFrame(test_rows), ["family", "condition", "metric"]).to_csv(output_root / "calibration_paired_holm.csv", index=False)

    manifest = {
        "environment": environment_record(device),
        "task": "post-hoc per-label Platt scaling (validation only), observed-pattern vs availability-aware",
        "models": model_names,
        "variants": variants,
        "conditions": CONDITIONS,
        "aa_weights_source": str(aa_root),
        "statistical_unit": "label; five-seed mean of calibrated probabilities; paired bootstrap with Holm within (family, condition, metric)",
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
    parser.add_argument("--run-name", default="posthoc_calibration_v1")
    parser.add_argument("--replicates", type=int, default=5000)
    args = parser.parse_args()
    run(args.project_root, load_config(args.project_root, args.config), args.run_name, args.replicates)


if __name__ == "__main__":
    main()
