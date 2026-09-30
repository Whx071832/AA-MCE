"""Consistent efficiency audit for the paper (one run, one device state).

Measures parameters and warmed batch-inference latency (median of repeated
forward passes, batch of 1,024 tracks) for the components and ensembles that
appear in the manuscript: the two MCE capacity components, the gated RAMT,
AttnFusion, EvidFusion, the fixed-weight MCE, the three-component AA-MCE and
the five-candidate upper bound.  Ensemble latency is measured by running the
component networks back to back (sequential execution, no batching tricks);
the availability-pattern lookup of weights, thresholds and Platt parameters is
a table read and is timed separately.  GPU and single-process CPU timings are
reported.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .advanced_fusion_baselines import build_model as build_recent
from .full_feature_ablation import build_model as build_generic
from .fusion_models import parameter_count
from .optimized_multimodal_tagging import FullFeatureTagger
from .common import environment_record, load_config


DIMENSIONS = [1034, 1000, 4096]
LABELS = 50


def component(name: str, settings: dict) -> torch.nn.Module:
    if name == "H384":
        return FullFeatureTagger(DIMENSIONS, 384, LABELS, "concat", 0.10, 0.20, False)
    if name == "H192":
        return FullFeatureTagger(DIMENSIONS, 192, LABELS, "concat", 0.30, 0.20, False)
    if name in ("RAMT", "EarlyConcat", "ConcatModDrop", "ReliabilityGate", "UniformMean"):
        return build_generic(name, DIMENSIONS, LABELS, settings)
    if name in ("AttnFusion", "EvidFusion"):
        return build_recent(name, DIMENSIONS, LABELS, dict(settings, modality_dropout=0.10))
    raise ValueError(name)


GROUPS = {
    "UniformMean": ["UniformMean"],
    "EarlyConcat": ["EarlyConcat"],
    "ConcatModDrop": ["ConcatModDrop"],
    "ReliabilityGate": ["ReliabilityGate"],
    "Concat H384/MD10": ["H384"],
    "Concat H192/MD30": ["H192"],
    "RAMT": ["RAMT"],
    "AttnFusion": ["AttnFusion"],
    "EvidFusion": ["EvidFusion"],
    "MCE": ["H384", "H192"],
    "AA-MCE": ["H384", "H192", "RAMT"],
    "AA-MCE (5 candidates)": ["H384", "H192", "RAMT", "AttnFusion", "EvidFusion"],
}


@torch.no_grad()
def time_group(models: list[torch.nn.Module], device: torch.device, batch: int, repeats: int) -> float:
    features = [torch.randn(batch, d, device=device) for d in DIMENSIONS]
    availability = torch.ones(batch, len(DIMENSIONS), device=device)
    reliability = torch.ones(batch, len(DIMENSIONS), device=device)
    for _ in range(5):
        for model in models:
            model(features, availability, reliability)
    if device.type == "cuda":
        torch.cuda.synchronize()
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        for model in models:
            model(features, availability, reliability)
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000.0)
    return float(np.median(timings))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", default="outputs/experiments/paper_efficiency_v1")
    parser.add_argument("--gpu-repeats", type=int, default=50)
    parser.add_argument("--cpu-repeats", type=int, default=10)
    args = parser.parse_args()
    config = load_config(args.project_root, "configs/experiments.yaml")
    settings = config["availability_aware_ensemble"]
    output = args.project_root / args.output
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20260724)
    names = sorted({name for members in GROUPS.values() for name in members})
    rows = []
    for device_name, repeats in (("cuda", args.gpu_repeats), ("cpu", args.cpu_repeats)):
        if device_name == "cuda" and not torch.cuda.is_available():
            continue
        device = torch.device(device_name)
        if device_name == "cpu":
            torch.set_num_threads(1)
        models = {name: component(name, settings).to(device).eval() for name in names}
        for group, members in GROUPS.items():
            latency = time_group([models[m] for m in members], device, 1024, repeats)
            rows.append(
                {
                    "model": group,
                    "components": "+".join(members),
                    "parameters": int(sum(parameter_count(models[m]) for m in members)),
                    "device": device_name,
                    "batch_size": 1024,
                    "median_latency_ms": latency,
                    "throughput_tracks_per_s": 1024 / (latency / 1000.0),
                }
            )
            print(device_name, group, f"{latency:.3f} ms", flush=True)
        del models
        if device_name == "cuda":
            torch.cuda.empty_cache()
    # Pattern lookup cost: weights, thresholds and Platt parameters for 1,024 tracks.
    patterns = np.random.default_rng(0).integers(0, 7, size=1024)
    table_w = np.random.rand(7, 3)
    table_t = np.random.rand(7, LABELS)
    table_a = np.random.rand(7, LABELS)
    probs = np.random.rand(3, 1024, LABELS)
    start = time.perf_counter()
    for _ in range(100):
        w = table_w[patterns]
        mixed = np.einsum("kn,nkl->nl", w.T, probs.transpose(1, 0, 2)) if False else (w[:, :, None] * probs.transpose(1, 0, 2)).sum(axis=1)
        calibrated = 1.0 / (1.0 + np.exp(-(table_a[patterns] * mixed)))
        decisions = calibrated >= table_t[patterns]
    lookup_ms = (time.perf_counter() - start) * 1000.0 / 100
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "efficiency.csv", index=False)
    payload = {"environment": environment_record(torch.device("cuda" if torch.cuda.is_available() else "cpu")), "cpu_threads": 1, "lookup_ms_per_1024_tracks_numpy": lookup_ms}
    (output / "run_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output / "completion.json").write_text(json.dumps({"status": "complete"}, indent=2), encoding="utf-8")
    print(frame.to_string())
    print("lookup ms per 1024 tracks:", round(lookup_ms, 4), decisions.shape)


if __name__ == "__main__":
    main()
