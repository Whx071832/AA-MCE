"""Shared fusion architectures for the redesigned tagging study.

This module provides the two recent strong fusion baselines requested by the
redesigned protocol (E2/E3) and a generic multimodal tagger used by the FMA
cross-dataset campaign:

- ``AttnFusionTagger``: modality-token transformer fusion with masked
  attention pooling (multimodal-transformer style baseline).
- ``EvidFusionTagger``: multi-label subjective-logic adaptation of trusted
  multi-view classification.  Each modality emits per-label binary Dirichlet
  evidence; available opinions are combined with the reduced Dempster rule,
  and unavailable modalities contribute the vacuous (identity) opinion.
- ``GenericTagger``: single/mean/concat/reliability fusion with optional
  modality dropout, mirroring the frozen Onion architectures so the method
  recipe can be transferred to FMA without re-tuning.

All models share one forward signature::

    logits, gates = model(features_list, availability, reliability)

where ``gates`` reports per-modality mixture weights for mechanism figures.
"""

from __future__ import annotations

import math

import torch
from torch import nn

# The fused memory-efficient attention kernel produced a sporadic CUDA
# illegal-memory-access on this Windows/RTX 3090 host (observed during the
# first E2 campaign while the GPU was shared with another workload).  The
# math backend is deterministic and fast enough for these small models.
if torch.cuda.is_available():
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def drop_modalities(availability: torch.Tensor, dropout: float, training: bool) -> torch.Tensor:
    """Randomly hide modalities but always keep at least one available view."""
    if not training or dropout <= 0:
        return availability
    result = availability * (torch.rand_like(availability) >= dropout)
    missing_all = result.sum(dim=1) == 0
    if missing_all.any():
        original = availability[missing_all]
        scores = torch.rand_like(original).masked_fill(original == 0, -1.0)
        selected = scores.argmax(dim=1)
        repaired = result[missing_all]
        repaired[torch.arange(len(repaired), device=result.device), selected] = 1.0
        result[missing_all] = repaired
    return result


def modality_encoder(input_dimension: int, hidden_dimension: int, representation_dropout: float) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dimension, hidden_dimension),
        nn.LayerNorm(hidden_dimension),
        nn.GELU(),
        nn.Dropout(representation_dropout),
        nn.Linear(hidden_dimension, hidden_dimension),
        nn.GELU(),
    )


class GenericTagger(nn.Module):
    """Frozen-recipe fusion family: single/mean/concat/reliability (+dropout)."""

    def __init__(
        self,
        input_dimensions: list[int],
        hidden_dimension: int,
        labels: int,
        fusion: str,
        active_modalities: tuple[int, ...] | None = None,
        modality_dropout: float = 0.0,
        representation_dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.fusion = fusion
        self.count = len(input_dimensions)
        self.active_modalities = tuple(active_modalities) if active_modalities is not None else tuple(range(self.count))
        self.modality_dropout = modality_dropout
        self.encoders = nn.ModuleList(
            [modality_encoder(dimension, hidden_dimension, representation_dropout) for dimension in input_dimensions]
        )
        self.gates = nn.ModuleList([nn.Linear(hidden_dimension + 1, 1) for _ in input_dimensions])
        active = len(self.active_modalities)
        self.concat_head = nn.Sequential(
            nn.Linear(hidden_dimension * active + 2 * active, hidden_dimension),
            nn.GELU(),
            nn.Dropout(representation_dropout),
            nn.Linear(hidden_dimension, labels),
        )
        self.pool_head = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.GELU(),
            nn.Dropout(representation_dropout),
            nn.Linear(hidden_dimension, labels),
        )

    def forward(
        self,
        features: list[torch.Tensor],
        availability: torch.Tensor,
        reliability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        available = availability.clone()
        inactive = [index for index in range(self.count) if index not in self.active_modalities]
        if inactive:
            available[:, inactive] = 0.0
        available = drop_modalities(available, self.modality_dropout, self.training)
        hidden = torch.stack([encoder(features[index]) for index, encoder in enumerate(self.encoders)], dim=1)
        hidden = hidden * available[:, :, None]
        active = list(self.active_modalities)
        if self.fusion == "concat":
            fused = torch.cat(
                [hidden[:, active].reshape(len(available), -1), available[:, active], reliability[:, active] * available[:, active]],
                dim=1,
            )
            logits = self.concat_head(fused)
            gates = available / available.sum(dim=1, keepdim=True).clamp_min(1.0)
        elif self.fusion == "mean":
            gates = available / available.sum(dim=1, keepdim=True).clamp_min(1.0)
            logits = self.pool_head((hidden * gates[:, :, None]).sum(dim=1))
        elif self.fusion == "reliability":
            scores = torch.cat(
                [self.gates[index](torch.cat([hidden[:, index], reliability[:, index, None]], dim=1)) for index in range(self.count)],
                dim=1,
            )
            scores = scores + torch.log(reliability.clamp_min(0.05))
            scores = scores.masked_fill(available == 0, -1e4)
            gates = torch.softmax(scores, dim=1)
            logits = self.pool_head((hidden * gates[:, :, None]).sum(dim=1))
        else:
            raise ValueError(self.fusion)
        return logits, gates


class AttnFusionTagger(nn.Module):
    """Modality-token transformer fusion with masked attention pooling."""

    def __init__(
        self,
        input_dimensions: list[int],
        hidden_dimension: int,
        labels: int,
        heads: int = 4,
        layers: int = 2,
        modality_dropout: float = 0.10,
        representation_dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.modality_dropout = modality_dropout
        self.encoders = nn.ModuleList(
            [modality_encoder(dimension, hidden_dimension, representation_dropout) for dimension in input_dimensions]
        )
        self.state_projection = nn.Linear(2, hidden_dimension)
        self.modality_embedding = nn.Parameter(torch.zeros(len(input_dimensions), hidden_dimension))
        nn.init.normal_(self.modality_embedding, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dimension,
            nhead=heads,
            dim_feedforward=hidden_dimension * 2,
            dropout=representation_dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.pool_query = nn.Parameter(torch.zeros(hidden_dimension))
        nn.init.normal_(self.pool_query, std=0.02)
        self.head = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.GELU(),
            nn.Dropout(representation_dropout),
            nn.Linear(hidden_dimension, labels),
        )

    def forward(
        self,
        features: list[torch.Tensor],
        availability: torch.Tensor,
        reliability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        available = drop_modalities(availability, self.modality_dropout, self.training)
        tokens = torch.stack([encoder(features[index]) for index, encoder in enumerate(self.encoders)], dim=1)
        state = self.state_projection(torch.stack([available, reliability * available], dim=2))
        tokens = (tokens + state + self.modality_embedding[None, :, :]) * available[:, :, None]
        padding_mask = available == 0
        # Rows with no available view would make attention undefined; they
        # attend to the zeroed tokens instead and yield a constant output.
        empty = padding_mask.all(dim=1)
        if empty.any():
            padding_mask = padding_mask.clone()
            padding_mask[empty] = False
        encoded = self.transformer(tokens, src_key_padding_mask=padding_mask)
        scores = (encoded @ self.pool_query) / math.sqrt(encoded.shape[-1])
        scores = scores.masked_fill(padding_mask, -1e4)
        gates = torch.softmax(scores, dim=1)
        pooled = (encoded * gates[:, :, None]).sum(dim=1)
        return self.head(pooled), gates


class EvidFusionTagger(nn.Module):
    """Multi-label trusted fusion via per-label binary Dirichlet evidence."""

    def __init__(
        self,
        input_dimensions: list[int],
        hidden_dimension: int,
        labels: int,
        modality_dropout: float = 0.10,
        representation_dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.labels = labels
        self.modality_dropout = modality_dropout
        self.encoders = nn.ModuleList(
            [modality_encoder(dimension, hidden_dimension, representation_dropout) for dimension in input_dimensions]
        )
        self.evidence_heads = nn.ModuleList([nn.Linear(hidden_dimension, labels * 2) for _ in input_dimensions])

    @staticmethod
    def _combine(alpha_left: torch.Tensor, alpha_right: torch.Tensor) -> torch.Tensor:
        """Reduced Dempster combination of two binary Dirichlet opinions."""
        strength_left = alpha_left.sum(dim=-1)
        strength_right = alpha_right.sum(dim=-1)
        belief_left = (alpha_left - 1.0) / strength_left[..., None]
        belief_right = (alpha_right - 1.0) / strength_right[..., None]
        uncertainty_left = 2.0 / strength_left
        uncertainty_right = 2.0 / strength_right
        conflict = (
            belief_left[..., 0] * belief_right[..., 1] + belief_left[..., 1] * belief_right[..., 0]
        )
        scale = (1.0 - conflict).clamp_min(1e-6)
        belief = (
            belief_left * belief_right
            + belief_left * uncertainty_right[..., None]
            + belief_right * uncertainty_left[..., None]
        ) / scale[..., None]
        uncertainty = (uncertainty_left * uncertainty_right / scale).clamp_min(1e-6)
        strength = 2.0 / uncertainty
        return belief * strength[..., None] + 1.0

    def opinions(
        self,
        features: list[torch.Tensor],
        availability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        available = drop_modalities(availability, self.modality_dropout, self.training)
        alphas = []
        for index, (encoder, head) in enumerate(zip(self.encoders, self.evidence_heads, strict=True)):
            hidden = encoder(features[index])
            evidence = nn.functional.softplus(head(hidden)).reshape(len(hidden), self.labels, 2)
            # An unavailable modality contributes zero evidence, which is the
            # vacuous opinion and the identity element of the combination.
            evidence = evidence * available[:, index, None, None]
            alphas.append(evidence + 1.0)
        combined = alphas[0]
        for alpha in alphas[1:]:
            combined = self._combine(combined, alpha)
        return combined, torch.stack(alphas, dim=1)

    def forward(
        self,
        features: list[torch.Tensor],
        availability: torch.Tensor,
        reliability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        combined, per_modality = self.opinions(features, availability)
        probability = combined[..., 0] / combined.sum(dim=-1)
        # Return log-odds so downstream sigmoid recovers the probability.
        logits = torch.log(probability.clamp(1e-6, 1.0 - 1e-6)) - torch.log1p(-probability.clamp(1e-6, 1.0 - 1e-6))
        strengths = per_modality.sum(dim=-1).mean(dim=2)
        gates = strengths / strengths.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return logits, gates


def evidential_loss(
    combined_alpha: torch.Tensor,
    per_modality_alpha: torch.Tensor,
    availability: torch.Tensor,
    target: torch.Tensor,
    annealing: float,
) -> torch.Tensor:
    """TMC-style ACE + annealed KL loss over combined and per-view opinions."""

    def opinion_loss(alpha: torch.Tensor) -> torch.Tensor:
        one_hot = torch.stack([target, 1.0 - target], dim=-1)
        strength = alpha.sum(dim=-1, keepdim=True)
        ace = (one_hot * (torch.digamma(strength) - torch.digamma(alpha))).sum(dim=-1)
        adjusted = one_hot + (1.0 - one_hot) * alpha
        adjusted_strength = adjusted.sum(dim=-1)
        kl = (
            torch.lgamma(adjusted_strength)
            - math.lgamma(2.0)
            - torch.lgamma(adjusted).sum(dim=-1)
            + ((adjusted - 1.0) * (torch.digamma(adjusted) - torch.digamma(adjusted_strength[..., None]))).sum(dim=-1)
        )
        return ace + annealing * kl

    loss = opinion_loss(combined_alpha).mean()
    view_total = combined_alpha.new_zeros(())
    view_weight = combined_alpha.new_zeros(())
    for index in range(per_modality_alpha.shape[1]):
        view_loss = opinion_loss(per_modality_alpha[:, index])
        weight = availability[:, index, None]
        view_total = view_total + (view_loss * weight).sum()
        view_weight = view_weight + weight.expand_as(view_loss).sum()
    return loss + view_total / view_weight.clamp_min(1.0)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
