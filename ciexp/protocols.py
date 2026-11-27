"""Evaluation protocols: forecast origins, splits, embargoes, context grids and data-scarcity subsets.

Every function returns integer positions on a regular time axis so the same
protocol can drive ci-proto (zero-shot: only the context moves) and supervised
baselines (fit on the training span, optionally refit).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Origin:
    """A forecast origin: history is ``[:cutoff]`` and the target window is ``[cutoff:cutoff+h]``."""

    cutoff: int
    horizon: int


def rolling_origins(
    n_obs: int,
    horizon: int,
    *,
    n_windows: int,
    step: int | None = None,
    min_history: int | None = None,
) -> list[Origin]:
    """Last ``n_windows`` non-overlapping (``step = horizon``) or overlapping origins.

    Mirrors fev-bench / GIFT-Eval: windows end at the end of the series; each origin
    needs at least ``min_history`` (default ``2*horizon + 1``) past observations.
    """
    step = horizon if step is None else step
    min_history = 2 * horizon + 1 if min_history is None else min_history
    cutoffs = [n_obs - horizon - k * step for k in range(n_windows)]
    cutoffs = sorted(c for c in cutoffs if c >= min_history)
    return [Origin(c, horizon) for c in cutoffs]


def fev_windows(n_series: int) -> int:
    """fev-bench rule: 20 windows (<10 series), 10 windows (10–2000 series), else 1."""
    if n_series < 10:
        return 20
    if n_series <= 2000:
        return 10
    return 1


def gift_eval_lengths(base_length: int) -> dict[str, int]:
    """GIFT-Eval short/medium/long horizons: medium = 10x, long = 15x the short length."""
    return {"short": base_length, "medium": 10 * base_length, "long": 15 * base_length}


def date_split_origins(
    index: pd.DatetimeIndex,
    horizon: int,
    *,
    test_start: str,
    test_end: str,
    step: int = 1,
) -> list[Origin]:
    """Daily (or step-wise) advancing origins whose *target windows* fall in the test period.

    Used for the electricity-price protocol (Energy and AI 100914, 2026): origins advance
    daily, the target window must lie in [test_start, test_end], models are not refit.
    """
    start, end = pd.Timestamp(test_start), pd.Timestamp(test_end)
    positions = np.arange(len(index))
    first_target = positions[index >= start]
    origins = []
    for c in first_target[::step]:
        if c + horizon - 1 < len(index) and index[c + horizon - 1] <= end:
            origins.append(Origin(int(c), horizon))
    return origins


def last_admissible_cutoff(index: pd.DatetimeIndex, horizon: int, boundary: str) -> int:
    """Largest cutoff whose whole target window ends on or before ``boundary`` (no leakage)."""
    end = pd.Timestamp(boundary)
    last_target = int(np.searchsorted(index.values, np.datetime64(end), side="right")) - 1
    return last_target - horizon + 1


def embargo_mask(n_obs: int, boundaries: list[int], width: int) -> np.ndarray:
    """Boolean mask that drops ``width`` observations after each partition boundary.

    Energy and AI 100886 (2026) removes L+H observations at every boundary to keep lagged
    features from leaking across folds.
    """
    keep = np.ones(n_obs, dtype=bool)
    for b in boundaries:
        keep[b : min(n_obs, b + width)] = False
    return keep


def leave_one_year_out(index: pd.DatetimeIndex, years: list[int]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Four-fold leave-one-year-out validation splits (train mask, validation mask)."""
    out = []
    for year in years:
        val = np.asarray(index.year == year)
        train = np.isin(index.year, [y for y in years if y != year])
        out.append((train, val))
    return out


def context_grid(kind: str, *, freq_steps_per_day: int | None = None, max_len: int | None = None) -> list[int]:
    """Context lengths for the context-scaling experiment.

    * ``"pow2"``: 64, 128, …, 2^21 (2,097,152) — probes ci-proto's 2M+ claim and the
      automatic switch to the long checkpoint above 16,384 observations.
    * ``"days"``: 30, 90, 180, 365 days (Energy and AI 100914) converted with
      ``freq_steps_per_day``.
    * ``"epf_grid"``: 24, 28, …, 2048 (step 4; Energy and AI 100886 exhaustive search).
    * ``"ltsf"``: 96, 192, 336, 720 (S-Mamba, Neurocomputing 2025) plus 1536, 4096.
    """
    if kind == "pow2":
        grid = [2**k for k in range(6, 22)]
    elif kind == "days":
        if freq_steps_per_day is None:
            raise ValueError("freq_steps_per_day is required for the 'days' grid")
        grid = [d * freq_steps_per_day for d in (30, 90, 180, 365)]
    elif kind == "epf_grid":
        grid = list(range(24, 2049, 4))
    elif kind == "ltsf":
        grid = [96, 192, 336, 720, 1536, 4096]
    else:
        raise ValueError(f"unknown context grid {kind!r}")
    if max_len is not None:
        grid = [g for g in grid if g <= max_len]
    return grid


def scarcity_subset(
    n_train: int,
    ratio: float,
    *,
    mode: str = "contiguous_blocks",
    n_blocks: int = 4,
    seed: int = 0,
) -> np.ndarray:
    """Training positions kept under data scarcity (Water Research X 30, 2026).

    * ``"contiguous_blocks"``: ``n_blocks`` time-continuous segments, one per quarter of
      the training span, each holding ``ratio`` of that quarter (benchmarks' scheme).
    * ``"seasonal_random"``: random positions drawn within each of ``n_blocks`` seasonal
      segments (the fine-tuning scheme for the foundation model).
    """
    if not 0 < ratio <= 1:
        raise ValueError("ratio must be in (0, 1]")
    rng = np.random.default_rng(seed)
    edges = np.linspace(0, n_train, n_blocks + 1).astype(int)
    keep = []
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        size = max(1, int(round((b - a) * ratio)))
        if mode == "contiguous_blocks":
            start = rng.integers(a, max(a + 1, b - size + 1))
            keep.append(np.arange(start, min(b, start + size)))
        elif mode == "seasonal_random":
            keep.append(np.sort(rng.choice(np.arange(a, b), size=min(size, b - a), replace=False)))
        else:
            raise ValueError(f"unknown mode {mode!r}")
    return np.concatenate(keep)


def inject_missing(
    values: np.ndarray, rate: float, *, pattern: str = "random", block: int = 24, seed: int = 0
) -> np.ndarray:
    """Return a copy of ``values`` with NaNs injected (robustness experiment)."""
    rng = np.random.default_rng(seed)
    out = np.array(values, dtype=float, copy=True)
    n = out.shape[-1]
    if rate <= 0:
        return out
    if pattern == "random":
        mask = rng.random(out.shape) < rate
    elif pattern == "block":
        mask = np.zeros(out.shape, dtype=bool)
        n_blocks = max(1, int(round(rate * n / block)))
        for start in rng.integers(0, max(1, n - block), size=n_blocks):
            mask[..., start : start + block] = True
    else:
        raise ValueError(f"unknown pattern {pattern!r}")
    mask[..., -1] = False  # keep the last observation so the origin is defined
    out[mask] = np.nan
    return out
