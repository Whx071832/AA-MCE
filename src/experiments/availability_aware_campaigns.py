"""AA-MCE campaigns on FMA (E8-FMA) and on the six-view Onion study (E8-6V).

Both campaigns reproduce the availability-aware procedure of the three-view
Onion study with the shared core (``aa_core``):

1. the locked-recipe components (Concat H384/MD10 and Concat H192/MD30), the
   gated RAMT and the two recent baselines are trained exactly as in the
   frozen campaign of the dataset (same seeds, split, early stopping) and
   their checkpoints are persisted; the reproduction check compares the
   retrained mean test predictions with the frozen files;
2. for every non-empty availability pattern (7 on FMA, 63 on six views) the
   mixing weights over the candidate components and the per-label thresholds
   are selected on validation tracks masked to that pattern; the locked
   0.6/0.4 mixture is retained unless the validation gain exceeds the margin;
3. MCE, MCE + AA thresholds, AA-MCE (weights only / locked components /
   full) and every control with and without AA thresholds are evaluated on
   the test conditions of the dataset; paired label bootstrap with Holm.

Usage::

    python -m src.experiments.availability_aware_campaigns --dataset fma --run-name availability_aware_ensemble_fma_v1
    python -m src.experiments.availability_aware_campaigns --dataset sixview --run-name availability_aware_ensemble_sixview_v1
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
import yaml

from . import extended_modalities as sixview
from . import fma_cross_dataset as fma
from .aa_core import all_patterns, evaluate_variants, pattern_index, pattern_name, select_pattern_weights
from .common import environment_record, load_config, resolve_device
from .robust_multimodal_tagging import build_dataset, targets


def load_frozen(run_root: Path, models: list[str], labels: list[str], track_ids: np.ndarray, conditions: list[str]) -> dict[str, dict[str, np.ndarray]]:
    """Frozen mean test predictions of a run for the given condition list."""
    stored = json.loads((run_root / "labels.json").read_text(encoding="utf-8"))["labels"]
    if stored != labels:
        raise RuntimeError(f"Label order mismatch against {run_root}")
    frame = pd.read_parquet(run_root / "mean_test_predictions_all_conditions.parquet")
    columns = [f"score_{label}" for label in labels]
    result: dict[str, dict[str, np.ndarray]] = {}
    for model in models:
        result[model] = {}
        for condition in conditions:
            group = frame.loc[(frame["model"] == model) & (frame["condition"] == condition)]
            if not np.array_equal(group["track_id"].astype(str).to_numpy(), track_ids):
                raise RuntimeError(f"Track order mismatch: {model}/{condition}")
            result[model][condition] = group[columns].to_numpy(dtype=np.float32)
    return result


class Dataset:
    """Adapter exposing data, training and prediction for one dataset."""

    def __init__(self, name: str, project_root: Path, config: dict[str, Any], settings: dict[str, Any]) -> None:
        self.name = name
        self.project_root = project_root
        self.base = dict(config[str(settings["base_section"])])
        processed_root = project_root / config["project"]["processed_root"]
        if name == "fma":
            fma_root = project_root / "Data" / "fma" / "fma_metadata" / "fma_metadata"
            split_frames, labels, features_by_modality, availability_all, reliability_all = fma.build_task(processed_root, fma_root, self.base)
            self.features = {split: [features_by_modality[m][frame["_row"].to_numpy()] for m in fma.MODALITIES] for split, frame in split_frames.items()}
            self.availability = {split: availability_all[frame["_row"].to_numpy()] for split, frame in split_frames.items()}
            self.reliability = {split: reliability_all[frame["_row"].to_numpy()] for split, frame in split_frames.items()}
            self.target = {split: fma.targets_for(frame, len(labels)) for split, frame in split_frames.items()}
            self.labels = labels
            self.view_names = list(fma.MODALITIES)
            self.conditions = list(fma.CONDITIONS)
            self.frequencies = None
        elif name == "sixview":
            split_frames, labels, frequencies = build_dataset(processed_root, int(self.base["top_labels"]), int(self.base["min_train_tag_frequency"]))
            views = [dict(view) for view in self.base["views"]]
            self.senses = [str(view["sense"]) for view in views]
            self.features, self.availability, self.reliability = sixview.load_views(processed_root, split_frames, views)
            self.target = {split: targets(frame, len(labels)) for split, frame in split_frames.items()}
            self.labels = labels
            self.view_names = [str(view["name"]) for view in views]
            self.conditions = list(sixview.CONDITIONS)
            self.frequencies = frequencies
        else:
            raise ValueError(name)
        self.split_frames = split_frames
        self.test_track_ids = split_frames["test"]["track_id"].astype(str).to_numpy()

    def train(self, model_name: str, seed: int, device: torch.device):
        if self.name == "fma":
            return fma.train_model(model_name, seed, self.features, self.availability, self.reliability, self.target, self.base, device)
        return sixview.train_model(model_name, seed, self.features, self.availability, self.reliability, self.target, self.base, device)

    def build(self, model_name: str):
        dims = [values.shape[1] for values in self.features["train"]]
        if self.name == "fma":
            return fma.build_model(model_name, dims, len(self.labels), self.base)
        return sixview.build_model(model_name, dims, len(self.labels), self.base)

    def predict(self, model, features, availability, reliability, device: torch.device) -> np.ndarray:
        batch = int(self.base["evaluation_batch_size"])
        if self.name == "fma":
            return fma.predict(model, features, availability, reliability, device, batch)
        return sixview.predict(model, features, availability, reliability, device, batch)[0]

    def condition_mask(self, condition: str, number: int) -> np.ndarray:
        if self.name == "fma":
            return fma.intervention(condition, self.availability["test"], 20260910 + number)
        return sixview.intervention(condition, self.availability["test"], self.senses, 20260910 + number)


def run(project_root: Path, config: dict[str, Any], dataset_name: str, run_name: str, overwrite: bool) -> Path:
    settings = dict(config[f"availability_aware_ensemble_{dataset_name}"])
    output_root = project_root / config["project"]["outputs_root"] / "experiments" / run_name
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root = output_root / "checkpoints"
    checkpoint_root.mkdir(exist_ok=True)
    started = time.perf_counter()
    device = resolve_device(config["runtime"]["device"])
    data = Dataset(dataset_name, project_root, config, settings)
    seeds = [int(value) for value in settings["seeds"]]
    components = [str(value) for value in settings["candidate_components"]]
    controls = [str(value) for value in settings["control_models"]]
    trained = list(dict.fromkeys(components + controls))
    patterns = all_patterns(len(data.view_names))
    pattern_labels = [pattern_name(pattern, data.view_names) for pattern in patterns]
    observed_index = patterns.index(tuple([1] * len(data.view_names)))
    validation_available = data.availability["validation"]
    eligible = {index: np.flatnonzero((validation_available >= np.asarray(pattern, dtype=np.float32)).all(axis=1)) for index, pattern in enumerate(patterns)}
    masked = {index: np.tile(np.asarray(pattern, dtype=np.float32), (len(validation_available), 1)) for index, pattern in enumerate(patterns)}
    condition_masks = {condition: data.condition_mask(condition, number) for number, condition in enumerate(data.conditions)}
    test_patterns = {condition: pattern_index(mask, patterns, observed_index) for condition, mask in condition_masks.items()}

    validation_predictions: dict[int, dict[int, dict[str, np.ndarray]]] = {}
    test_predictions: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    history_frames = []
    reuse_root = project_root / str(settings["reuse_checkpoints_from"]) / "checkpoints" if settings.get("reuse_checkpoints_from") else None
    for seed in seeds:
        models = {}
        for name in trained:
            reusable = reuse_root / f"{name}_{seed}.pt" if reuse_root is not None else None
            if reusable is not None and reusable.exists():
                model = data.build(name).to(device)
                model.load_state_dict(torch.load(reusable, map_location="cpu", weights_only=True))
                print(f"reused model={name} seed={seed} from {reusable.parent.parent.name}", flush=True)
            else:
                model, _, history = data.train(name, seed, device)
                history_frames.append(history)
                torch.save(model.state_dict(), checkpoint_root / f"{name}_{seed}.pt")
                print(f"trained model={name} seed={seed} best_validation_mAP={history['validation_mAP'].max():.5f}", flush=True)
            models[name] = model
        validation_predictions[seed] = {}
        for index in range(len(patterns)):
            rows = eligible[index]
            validation_predictions[seed][index] = {
                name: data.predict(model, [values[rows] for values in data.features["validation"]], masked[index][rows], data.reliability["validation"][rows], device)
                for name, model in models.items()
            }
        test_predictions[seed] = {
            condition: {name: data.predict(model, data.features["test"], condition_masks[condition], data.reliability["test"], device) for name, model in models.items()}
            for condition in data.conditions
        }
        print(f"predictions complete seed={seed}", flush=True)
        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if history_frames:
        pd.concat(history_frames, ignore_index=True).to_csv(output_root / "training_history.csv", index=False)

    locked = {name: float(value) for name, value in zip(settings["locked_components"], settings["locked_weights"], strict=True)}
    pattern_weights, weight_rows, grid_rows = select_pattern_weights(
        validation_predictions, eligible, data.target["validation"], patterns, pattern_labels, components, locked, float(settings["weight_step"]), float(settings["selection_margin"])
    )
    grid_rows.to_csv(output_root / "pattern_weight_grid.csv", index=False)
    weight_rows.to_csv(output_root / "pattern_weights.csv", index=False)
    print("pattern weights selected", flush=True)

    mean_test = evaluate_variants(
        output_root=output_root,
        labels=data.labels,
        patterns=patterns,
        pattern_labels=pattern_labels,
        observed_index=observed_index,
        components=components,
        controls=controls,
        locked=locked,
        pattern_weights=pattern_weights,
        validation_predictions=validation_predictions,
        eligible=eligible,
        target_validation=data.target["validation"],
        test_predictions=test_predictions,
        test_patterns=test_patterns,
        target_test=data.target["test"],
        conditions=data.conditions,
        test_track_ids=data.test_track_ids,
        test_proposed=[str(value) for value in settings["test_proposed"]],
        replicates=int(settings["bootstrap_replicates"]),
        bootstrap_seed=int(settings["bootstrap_seed"]),
    )

    checks: dict[str, float] = {}
    reference_root = project_root / str(settings["reference_run"])
    frozen = load_frozen(reference_root, [str(v) for v in settings["reference_models"]], data.labels, data.test_track_ids, data.conditions)
    for name, per_condition in frozen.items():
        local = "MCE" if name in ("MCE", "OptimizedEnsemble") else name
        checks[f"{name}_vs_frozen_max_abs_diff"] = float(max(np.abs(mean_test[local][condition] - per_condition[condition]).max() for condition in data.conditions))
    (output_root / "reproduction_check.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")

    payload = {"labels": data.labels}
    if data.frequencies is not None:
        payload["train_frequencies"] = {label: data.frequencies[label] for label in data.labels}
    (output_root / "labels.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    manifest = {
        "environment": environment_record(device),
        "task": f"AA-MCE on {dataset_name}: availability-conditioned mixing weights and thresholds (validation only)",
        "samples": {split: len(frame) for split, frame in data.split_frames.items()},
        "views": data.view_names,
        "patterns": len(patterns),
        "eligible_validation_tracks": {pattern_labels[index]: int(len(rows)) for index, rows in eligible.items()},
        "conditions": data.conditions,
        "candidate_components": components,
        "controls": controls,
        "locked": locked,
        "selection_margin": float(settings["selection_margin"]),
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
    parser.add_argument("--dataset", choices=["fma", "sixview"], required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.project_root, load_config(args.project_root, args.config), args.dataset, args.run_name, args.overwrite)


if __name__ == "__main__":
    main()
