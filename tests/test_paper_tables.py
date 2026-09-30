"""The key results quoted in README.md are the ones stored in results/paper_tables."""

import json
from pathlib import Path

import pandas as pd
import pytest

TABLES = Path(__file__).resolve().parents[1] / "results" / "paper_tables"
ONION_MISSING = ["no_audio", "no_lyrics", "no_visual", "random_one_missing", "random_two_missing"]


def mean(frame: pd.DataFrame, model: str, condition: str, metric: str) -> float:
    row = frame.loc[(frame["model"] == model) & (frame["condition"] == condition) & (frame["metric"] == metric), "mean"]
    assert len(row) == 1, (model, condition, metric)
    return float(row.iloc[0])


@pytest.mark.parametrize(
    ("model", "complete", "missing", "keep_one_f1"),
    [
        ("AA-MCE", 0.4471, 0.3892, 0.3885),
        ("MCE", 0.4459, 0.3868, 0.3686),
        ("RAMT", 0.4286, 0.3770, 0.3790),
        ("AttnFusion", 0.4326, 0.3743, 0.3732),
        ("EvidFusion", 0.4220, 0.3715, 0.3421),
        ("H384", 0.4390, 0.3791, 0.3614),
    ],
)
def test_onion_key_results(model, complete, missing, keep_one_f1):
    onion = pd.read_csv(TABLES / "onion_summary.csv")
    assert round(mean(onion, model, "observed", "mAP"), 4) == complete
    assert round(sum(mean(onion, model, c, "mAP") for c in ONION_MISSING) / 5, 4) == missing
    assert round(mean(onion, model, "random_two_missing", "MacroF1"), 4) == keep_one_f1


def test_transfer_key_results():
    numbers = json.loads((TABLES / "numbers.json").read_text(encoding="utf-8"))
    assert round(numbers["fma_AA-MCE_missing_mean_mAP"], 4) == 0.4093
    assert round(numbers["fma_MCE_missing_mean_mAP"], 4) == 0.4045
    assert round(numbers["six_AA-MCE_missing_mean_mAP"], 4) == 0.4032
    assert round(numbers["six_MCE_missing_mean_mAP"], 4) == 0.3981


def test_calibration_key_result():
    facts = json.loads((TABLES / "facts.json").read_text(encoding="utf-8"))
    ece = {(row[0], row[1]): row[3] for row in facts["calibration"] if row[2] == "observed"}
    assert ece[("AA-MCE", "raw")] == 0.0921
    assert ece[("AA-MCE", "cal-AA")] == 0.0233
