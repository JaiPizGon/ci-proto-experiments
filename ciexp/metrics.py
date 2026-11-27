"""Forecast accuracy metrics used across the experiment suite.

Conventions
-----------
* ``y``      : array ``[..., H]`` of observed values (NaN = not observed, ignored).
* ``point``  : array ``[..., H]`` point forecast (the median for quantile models).
* ``qpred``  : array ``[..., Q, H]`` quantile forecasts, ordered like ``levels``.
* ``levels`` : array ``[Q]`` of quantile levels in (0, 1).

Definitions follow the papers whose protocols are replicated:

* MASE / SQL follow fev-bench (Shchur et al., 2025): errors are scaled by the
  in-sample mean absolute error of the seasonal-naive forecast. SQL uses the
  factor 2 so that the SQL of a single 0.5-quantile equals its MASE.
* WQL follows Chronos (Ansari et al., 2024): ``2 * sum(pinball) / sum(|y|)``,
  averaged over quantile levels.
* QLIKE follows Patton (2011), as used for volatility in Finance Research Letters
  110 (2026) 110606.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def _valid(y: np.ndarray, *others: np.ndarray) -> np.ndarray:
    mask = np.isfinite(y)
    for other in others:
        mask &= np.isfinite(other)
    return mask


# --------------------------------------------------------------------------- point metrics
def mae(y: np.ndarray, point: np.ndarray) -> float:
    m = _valid(y, point)
    return float(np.mean(np.abs(y[m] - point[m]))) if m.any() else np.nan


def rmse(y: np.ndarray, point: np.ndarray) -> float:
    m = _valid(y, point)
    return float(np.sqrt(np.mean((y[m] - point[m]) ** 2))) if m.any() else np.nan


def mape(y: np.ndarray, point: np.ndarray) -> float:
    m = _valid(y, point) & (np.abs(y) > EPS)
    return float(100 * np.mean(np.abs((y[m] - point[m]) / y[m]))) if m.any() else np.nan


def smape(y: np.ndarray, point: np.ndarray) -> float:
    m = _valid(y, point)
    denom = np.abs(y[m]) + np.abs(point[m])
    ok = denom > EPS
    if not ok.any():
        return np.nan
    return float(200 * np.mean(np.abs(y[m] - point[m])[ok] / denom[ok]))


def wape(y: np.ndarray, point: np.ndarray) -> float:
    m = _valid(y, point)
    denom = np.sum(np.abs(y[m]))
    return float(np.sum(np.abs(y[m] - point[m])) / denom) if denom > EPS else np.nan


def seasonal_naive_scale(history: np.ndarray, m: int) -> float:
    """In-sample mean |y_t - y_{t-m}| over the observed history (fev-bench / MASE scale).

    Falls back to lag 1 when the history is shorter than ``m + 1`` observations.
    """
    h = np.asarray(history, dtype=float)
    lag = m if len(h) > m else 1
    diff = np.abs(h[lag:] - h[:-lag])
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return np.nan
    scale = float(np.mean(diff))
    return scale if scale > EPS else np.nan


def mase(y: np.ndarray, point: np.ndarray, history: np.ndarray, m: int) -> float:
    scale = seasonal_naive_scale(history, m)
    return mae(y, point) / scale if np.isfinite(scale) else np.nan


# ------------------------------------------------------------------- probabilistic metrics
def pinball(y: np.ndarray, qpred: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Element-wise pinball loss, shape ``[..., Q, H]``."""
    levels = np.asarray(levels, dtype=float).reshape((-1, 1))
    diff = y[..., None, :] - qpred
    return np.maximum(levels * diff, (levels - 1) * diff)


def mean_quantile_loss(y: np.ndarray, qpred: np.ndarray, levels: np.ndarray) -> float:
    """Average of ``2 * pinball`` over quantiles and steps (a CRPS approximation)."""
    loss = 2 * pinball(y, qpred, levels)
    m = np.broadcast_to(np.isfinite(y)[..., None, :], loss.shape) & np.isfinite(loss)
    return float(np.mean(loss[m])) if m.any() else np.nan


def wql(y: np.ndarray, qpred: np.ndarray, levels: np.ndarray) -> float:
    """Weighted quantile loss (Chronos): mean over q of 2*sum(pinball_q)/sum|y|."""
    m = np.isfinite(y)
    denom = np.sum(np.abs(y[m]))
    if denom <= EPS:
        return np.nan
    loss = 2 * pinball(np.where(m, y, 0.0), qpred, levels)
    loss = np.where(m[..., None, :], loss, 0.0)
    per_q = loss.reshape(-1, loss.shape[-2], loss.shape[-1]).sum(axis=(0, 2)) / denom
    return float(np.mean(per_q))


def sql(y: np.ndarray, qpred: np.ndarray, levels: np.ndarray, history: np.ndarray, m: int) -> float:
    """Scaled quantile loss (fev-bench)."""
    scale = seasonal_naive_scale(history, m)
    return mean_quantile_loss(y, qpred, levels) / scale if np.isfinite(scale) else np.nan


def crps_quantile(y: np.ndarray, qpred: np.ndarray, levels: np.ndarray) -> float:
    """CRPS approximated by the mean quantile loss over the available levels (GIFT-Eval)."""
    return mean_quantile_loss(y, qpred, levels)


def _level_index(levels: np.ndarray, level: float) -> int:
    idx = int(np.argmin(np.abs(np.asarray(levels) - level)))
    if abs(levels[idx] - level) > 1e-6:
        raise ValueError(f"quantile level {level} not available in {list(levels)}")
    return idx


def coverage(y: np.ndarray, qpred: np.ndarray, levels: np.ndarray, lo: float, hi: float) -> float:
    """Empirical share of observations inside [q_lo, q_hi]."""
    lower = qpred[..., _level_index(levels, lo), :]
    upper = qpred[..., _level_index(levels, hi), :]
    m = _valid(y, lower, upper)
    return float(np.mean((y[m] >= lower[m]) & (y[m] <= upper[m]))) if m.any() else np.nan


def interval_width(qpred: np.ndarray, levels: np.ndarray, lo: float, hi: float) -> float:
    lower = qpred[..., _level_index(levels, lo), :]
    upper = qpred[..., _level_index(levels, hi), :]
    m = _valid(lower, upper)
    return float(np.mean(upper[m] - lower[m])) if m.any() else np.nan


def interval_score(y: np.ndarray, qpred: np.ndarray, levels: np.ndarray, lo: float, hi: float) -> float:
    """Gneiting & Raftery (2007) interval score for the central (hi-lo) interval."""
    alpha = 1 - (hi - lo)
    lower = qpred[..., _level_index(levels, lo), :]
    upper = qpred[..., _level_index(levels, hi), :]
    m = _valid(y, lower, upper)
    if not m.any():
        return np.nan
    yy, ll, uu = y[m], lower[m], upper[m]
    score = (uu - ll) + (2 / alpha) * (ll - yy) * (yy < ll) + (2 / alpha) * (yy - uu) * (yy > uu)
    return float(np.mean(score))


def calibration_curve(y: np.ndarray, qpred: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Observed frequency ``P(y <= q_tau)`` for each nominal level (reliability diagram)."""
    out = []
    for i in range(len(levels)):
        q = qpred[..., i, :]
        m = _valid(y, q)
        out.append(np.mean(y[m] <= q[m]) if m.any() else np.nan)
    return np.asarray(out)


# ------------------------------------------------------------------- domain-specific metrics
def qlike(rv: np.ndarray, rv_hat: np.ndarray) -> float:
    """QLIKE loss for volatility forecasts (Patton, 2011)."""
    m = _valid(rv, rv_hat) & (rv > EPS) & (rv_hat > EPS)
    ratio = rv[m] / rv_hat[m]
    return float(np.mean(ratio - np.log(ratio) - 1)) if m.any() else np.nan


def peak_rmse(y: np.ndarray, point: np.ndarray, threshold: float) -> float:
    """RMSE restricted to observations above ``threshold`` (peak events, Water Res. X 2026)."""
    m = _valid(y, point) & (y >= threshold)
    return float(np.sqrt(np.mean((y[m] - point[m]) ** 2))) if m.any() else np.nan


def r2(y: np.ndarray, point: np.ndarray) -> float:
    m = _valid(y, point)
    if m.sum() < 2:
        return np.nan
    ss_res = np.sum((y[m] - point[m]) ** 2)
    ss_tot = np.sum((y[m] - np.mean(y[m])) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > EPS else np.nan


def nse(y: np.ndarray, point: np.ndarray) -> float:
    """Nash–Sutcliffe efficiency (hydrology); identical to R² against the observed mean."""
    return r2(y, point)


def kge(y: np.ndarray, point: np.ndarray) -> float:
    """Kling–Gupta efficiency (Gupta et al., 2009)."""
    m = _valid(y, point)
    if m.sum() < 2:
        return np.nan
    obs, sim = y[m], point[m]
    if np.std(obs) <= EPS or abs(np.mean(obs)) <= EPS:
        return np.nan
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim) / np.std(obs)
    beta = np.mean(sim) / np.mean(obs)
    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


# --------------------------------------------------------------------------- convenience
def score_forecast(
    y: np.ndarray,
    qpred: np.ndarray,
    levels: np.ndarray,
    history: np.ndarray,
    m: int,
    *,
    point: np.ndarray | None = None,
) -> dict[str, float]:
    """All standard metrics for one univariate forecast window (``y``: [H], ``qpred``: [Q, H])."""
    levels = np.asarray(levels, dtype=float)
    if point is None:
        point = qpred[_level_index(levels, 0.5)]
    out = {
        "MAE": mae(y, point),
        "RMSE": rmse(y, point),
        "sMAPE": smape(y, point),
        "MAPE": mape(y, point),
        "WAPE": wape(y, point),
        "MASE": mase(y, point, history, m),
        "WQL": wql(y, qpred, levels),
        "SQL": sql(y, qpred, levels, history, m),
        "CRPS": crps_quantile(y, qpred, levels),
        "scale": seasonal_naive_scale(history, m),
        "abs_sum": float(np.nansum(np.abs(y))),
    }
    rounded = set(np.round(levels, 6).tolist())
    for lo, hi in ((0.1, 0.9), (0.2, 0.8), (0.3, 0.7)):
        if lo in rounded and hi in rounded:
            nominal = int(round(100 * (hi - lo)))
            out[f"COV{nominal}"] = coverage(y, qpred, levels, lo, hi)
            out[f"WIDTH{nominal}"] = interval_width(qpred, levels, lo, hi)
            out[f"IS{nominal}"] = interval_score(y, qpred, levels, lo, hi)
    return out
