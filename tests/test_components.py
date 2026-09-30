"""Component taggers of AA-MCE on tiny random inputs (CPU, no data)."""

import torch

from src.experiments.availability_aware_ensemble import COMPONENT_SPECS
from src.experiments.full_feature_ablation import build_model as build_generic_model
from src.experiments.fusion_models import EvidFusionTagger, drop_modalities
from src.experiments.optimized_multimodal_tagging import FullFeatureTagger

DIMENSIONS = [12, 10, 16]  # stand-ins for the 1,034 / 1,000 / 4,096-d Onion views
LABELS = 5


def inputs():
    torch.manual_seed(0)
    features = [torch.randn(8, dimension) for dimension in DIMENSIONS]
    availability = torch.tensor([[1, 1, 1], [1, 0, 1], [0, 1, 0], [1, 1, 0]] * 2, dtype=torch.float32)
    return features, availability, torch.ones(8, 3)


def test_component_recipes():
    assert COMPONENT_SPECS == {
        "FullConcatMD10-H384-BCE": {"hidden_dimension": 384, "modality_dropout": 0.10},  # C1
        "FullConcatMD-BCE": {"hidden_dimension": 192, "modality_dropout": 0.30},  # C2
    }


def test_concat_components_and_ramt_forward():
    features, availability, reliability = inputs()
    for spec in COMPONENT_SPECS.values():
        model = FullFeatureTagger(DIMENSIONS, spec["hidden_dimension"], LABELS, "concat", spec["modality_dropout"], 0.20, False).eval()
        assert model(features, availability, reliability).shape == (8, LABELS)
    ramt = build_generic_model("RAMT", DIMENSIONS, LABELS, {"hidden_dimension": 192, "modality_dropout": 0.30, "representation_dropout": 0.20}).eval()  # C3
    logits, gates = ramt(features, availability, reliability)
    assert logits.shape == (8, LABELS)
    torch.testing.assert_close(gates.sum(dim=1), torch.ones(8))
    assert torch.all(gates[availability == 0] < 1e-6)  # missing modalities receive no gate weight


def test_modality_dropout_always_keeps_one_available_view():
    torch.manual_seed(0)
    availability = torch.ones(2000, 3)
    dropped = drop_modalities(availability, 0.9, training=True)
    assert torch.all(dropped.sum(dim=1) >= 1)
    assert torch.equal(drop_modalities(availability, 0.9, training=False), availability)
    component = FullFeatureTagger(DIMENSIONS, 16, LABELS, "concat", 0.9, 0.2, False).train()
    assert torch.all(component.drop_modalities(availability).sum(dim=1) >= 1)


def test_vacuous_opinion_is_the_identity_of_evidential_fusion():
    alpha = torch.tensor([[[3.0, 2.0], [1.5, 4.0]]])
    torch.testing.assert_close(EvidFusionTagger._combine(alpha, torch.ones_like(alpha)), alpha)
