from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters for TWEC (compass + per-slice SGNS) training."""

    dim: int = 100
    window: int = 5
    negative: int = 5
    min_count: int = 5
    subsample: float = 1e-3
    compass_epochs: int = 5
    slice_epochs: int = 5
    lr: float = 0.025
    min_lr: float = 1e-4
    batch_size: int = 8192
    seed: int = 42
    device: str | None = None  # None = auto-detect (cuda > mps > cpu)


@dataclass(frozen=True)
class QueryConfig:
    """Settings for serving-time queries."""

    k: int = 10
    # Word-slice results with raw frequency below this are flagged unreliable.
    low_frequency_threshold: float = 20.0
