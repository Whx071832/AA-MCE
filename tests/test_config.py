"""The configuration encodes the protocol described in the README."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "configs" / "experiments.yaml").read_text(encoding="utf-8"))
SEEDS = [20260724, 20260725, 20260726, 20260727, 20260728]


def test_locked_mce_recipe():
    settings = CONFIG["optimized_multimodal_tagging"]
    assert settings["ensemble"] == {"name": "OptimizedEnsemble", "components": ["FullConcatMD10-H384-BCE", "FullConcatMD-BCE"], "weights": [0.6, 0.4]}
    assert settings["seeds"] == SEEDS
    assert settings["top_labels"] == 50


def test_aa_mce_selection_settings():
    for section in ("availability_aware_ensemble", "availability_aware_ensemble_fma", "availability_aware_ensemble_sixview"):
        settings = CONFIG[section]
        assert len(settings["candidate_components"]) == 3
        assert settings["locked_weights"] == [0.6, 0.4]
        assert settings["weight_step"] == 0.1
        assert settings["selection_margin"] == 0.0005
        assert settings["seeds"] == SEEDS
    onion = CONFIG["availability_aware_ensemble"]
    assert onion["candidate_components"] == ["FullConcatMD10-H384-BCE", "FullConcatMD-BCE", "RAMT"]
    assert (onion["hidden_dimension"], onion["modality_dropout"]) == (192, 0.30)  # RAMT recipe (C3)
    assert len(CONFIG["availability_aware_ensemble_5c"]["candidate_components"]) == 5


def test_transfer_settings():
    fma = CONFIG["fma_cross_dataset"]
    assert fma["top_labels"] == 30
    assert fma["ensemble"]["weights"] == [0.6, 0.4]
    assert (fma["ensemble_hidden_dimension"], fma["ensemble_component_dropout"]) == (384, 0.10)
    views = CONFIG["extended_modalities"]["views"]
    assert len(views) == 6
    assert [view["sense"] for view in views] == ["audio", "audio", "lyrics", "lyrics", "visual", "visual"]


def test_tag_groups_partition_the_50_labels():
    tags = [tag for group in CONFIG["tag_groups"].values() for tag in group]
    assert len(tags) == len(set(tags)) == 50
