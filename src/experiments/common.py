"""Shared run utilities: configuration loading, seeding, device selection,
environment records and the paired bootstrap used by every statistical test.

These helpers were originally defined in the project's earlier recommendation
runners; they are collected here verbatim so that the tagging study does not
depend on that code.
"""

from __future__ import annotations

import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

# ``datetime.UTC`` exists only on Python >= 3.11; it is an alias of ``timezone.utc``.
UTC = timezone.utc


def load_config(project_root: Path, config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = project_root / path
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_device(preference: str) -> torch.device:
    if preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if preference in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int, deterministic: bool) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def environment_record(device: torch.device) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        payload.update(
            {
                "gpu": torch.cuda.get_device_name(device),
                "cuda": torch.version.cuda,
                "gpu_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
            }
        )
    return payload


def _bootstrap(delta: np.ndarray, seed: int, replicates: int) -> tuple[float, float, float]:
    """Percentile CI and two-sided p-value of the mean of paired differences."""
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 64):
        count = min(64, replicates - start)
        draw = rng.integers(0, len(delta), size=(count, len(delta)))
        means[start : start + count] = delta[draw].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    p_value = min(1.0, 2.0 * min(float((means <= 0).mean()), float((means >= 0).mean())))
    return float(low), float(high), p_value
