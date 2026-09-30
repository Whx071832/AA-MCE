"""Select a tagging ensemble from validation predictions only."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

from .robust_multimodal_tagging import build_dataset, targets


def run(project_root: Path, run_name: str) -> Path:
    root = project_root / "outputs" / "experiments" / run_name
    predictions = pd.read_parquet(root / "validation_predictions.parquet")
    labels = json.loads((root / "labels.json").read_text(encoding="utf-8"))["labels"]
    columns = [f"score_{label}" for label in labels]
    split_frames, _, _ = build_dataset(project_root / "Data" / "processed" / "v1", len(labels), 400)
    target = targets(split_frames["validation"], len(labels))
    probability = {model: group[columns].to_numpy() for model, group in predictions.groupby("model", sort=False)}
    sweep = pd.read_csv(root / "validation_sweep.csv").sort_values("mAP", ascending=False)
    # Restrict the preregistered search to the five best single models. This
    # avoids a large, weakly justified combinatorial search on one validation
    # partition while retaining architecturally distinct candidates.
    selected_models = sweep.head(5)["model"].astype(str).tolist()
    probability = {model: probability[model] for model in selected_models}
    ranks = {model: np.apply_along_axis(rankdata, 0, values) / len(values) for model, values in probability.items()}
    rows: list[dict[str, object]] = []
    for kind, values_by_model in (("probability", probability), ("rank", ranks)):
        for left, right in itertools.combinations(values_by_model, 2):
            for weight in np.arange(0.10, 1.0, 0.10):
                combined = weight * values_by_model[left] + (1.0 - weight) * values_by_model[right]
                rows.append(
                    {
                        "kind": kind,
                        "left": left,
                        "right": right,
                        "left_weight": float(weight),
                        "validation_mAP": float(average_precision_score(target, combined, average="macro")),
                    }
                )
    result = pd.DataFrame(rows).sort_values("validation_mAP", ascending=False).reset_index(drop=True)
    result.to_csv(root / "validation_ensemble_search.csv", index=False)
    best = result.iloc[0].to_dict()
    best["right_weight"] = 1.0 - float(best["left_weight"])
    best["test_evaluated"] = False
    best["selection_metric"] = "validation mAP"
    (root / "ensemble_selection.json").write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(best, ensure_ascii=False, indent=2))
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-name", default="optimized_tagging_validation_v2")
    args = parser.parse_args()
    run(args.project_root, args.run_name)


if __name__ == "__main__":
    main()
