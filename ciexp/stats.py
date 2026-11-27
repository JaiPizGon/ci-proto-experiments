"""Statistical comparison of forecasters.

Implements the tests and aggregation rules used by the replicated studies:

* Diebold–Mariano with HAC (Bartlett, lag h-1) variance and the
  Harvey–Leybourne–Newbold correction, Student-t(n-1) p-values
  (Energy and AI 100914, 2026).
* Holm step-down and Benjamini–Hochberg multiplicity control across markets.
* Model Confidence Set, T_max statistic with moving-block bootstrap
  (Hansen, Lunde & Nason, 2011; Energy and AI 100886, 2026; Finance Research
  Letters 110606, 2026).
* fev-bench aggregation: geometric-mean skill score versus Seasonal Naive with
  clipping to [1e-2, 100], win rates with ties as half wins, and paired
  bootstrap confidence intervals over tasks (B = 1000).
* Friedman test with Nemenyi critical difference over datasets (Demšar, 2006),
  the convention of most Q1 journals for multi-dataset comparisons.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import stats as sps


# --------------------------------------------------------------------- Diebold–Mariano
@dataclass(frozen=True)
class DMResult:
    statistic: float
    p_two_sided: float
    p_less: float  # H1: model 1 has lower expected loss than model 2
    mean_diff: float
    n: int


def _hac_variance(d: np.ndarray, lag: int) -> float:
    n = len(d)
    dc = d - d.mean()
    gamma0 = float(np.dot(dc, dc) / n)
    var = gamma0
    for k in range(1, lag + 1):
        weight = 1.0 - k / (lag + 1.0)  # Bartlett kernel
        gamma_k = float(np.dot(dc[k:], dc[:-k]) / n)
        var += 2.0 * weight * gamma_k
    return var if var > 0 else gamma0


def diebold_mariano(
    e1: np.ndarray,
    e2: np.ndarray,
    horizon: int,
    *,
    loss: str | Callable[[np.ndarray], np.ndarray] = "abs",
    lag: int | None = None,
    hln: bool = True,
) -> DMResult:
    """DM test on forecast errors ``e1`` and ``e2`` for the same targets.

    ``loss`` is ``"abs"``, ``"sq"`` or a callable applied to the errors. Losses can be
    passed directly by giving ``loss=lambda x: x`` with ``e1``/``e2`` already losses.
    """
    e1, e2 = np.asarray(e1, float), np.asarray(e2, float)
    ok = np.isfinite(e1) & np.isfinite(e2)
    e1, e2 = e1[ok], e2[ok]
    if loss == "abs":
        d = np.abs(e1) - np.abs(e2)
    elif loss == "sq":
        d = e1**2 - e2**2
    else:
        d = loss(e1) - loss(e2)
    n = len(d)
    if n < 3:
        return DMResult(np.nan, np.nan, np.nan, np.nan, n)
    lag = horizon - 1 if lag is None else lag
    var = _hac_variance(d, max(lag, 0)) / n
    if var <= 0:
        return DMResult(np.nan, np.nan, np.nan, float(d.mean()), n)
    dm = float(d.mean() / np.sqrt(var))
    if hln:
        h = horizon
        dm *= np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-12))
    p_two = float(2 * sps.t.sf(abs(dm), df=n - 1))
    p_less = float(sps.t.cdf(dm, df=n - 1))
    return DMResult(dm, p_two, p_less, float(d.mean()), n)


# ------------------------------------------------------------------ multiplicity control
def holm(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    m = len(pv)
    if m == 0:
        return out
    order = np.argsort(pv)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pv[idx]))
        adjusted[idx] = running
    out[ok] = adjusted
    return out


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    m = len(pv)
    if m == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(m)
    adjusted[order] = np.minimum(ranked, 1.0)
    out[ok] = adjusted
    return out


# ------------------------------------------------------------------ Model Confidence Set
def _block_bootstrap_indices(n: int, block: int, reps: int, rng: np.random.Generator) -> np.ndarray:
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(reps, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(reps, -1)[:, :n]
    return idx


@dataclass(frozen=True)
class MCSResult:
    included: list[str]
    pvalues: dict[str, float]
    elimination_order: list[str]


def model_confidence_set(
    losses: np.ndarray,
    names: list[str],
    *,
    alpha: float = 0.10,
    reps: int = 1000,
    block: int | None = None,
    seed: int = 0,
) -> MCSResult:
    """Hansen–Lunde–Nason MCS with the T_max statistic.

    ``losses``: ``[n_obs, n_models]`` loss matrix aligned on the same targets.
    """
    losses = np.asarray(losses, float)
    keep_rows = np.isfinite(losses).all(axis=1)
    losses = losses[keep_rows]
    n, m = losses.shape
    if m != len(names):
        raise ValueError("names must match loss columns")
    rng = np.random.default_rng(seed)
    block = block or max(1, int(round(n ** (1 / 3))))
    idx = _block_bootstrap_indices(n, block, reps, rng)
    boot_means = losses[idx].mean(axis=1)  # [reps, m]
    full_means = losses.mean(axis=0)
    alive = list(range(m))
    pvals: dict[str, float] = {}
    order: list[str] = []
    running_p = 0.0
    while len(alive) > 1:
        sub = full_means[alive]
        bsub = boot_means[:, alive]
        d_i = sub - sub.mean()
        d_i_boot = bsub - bsub.mean(axis=1, keepdims=True)
        var = np.mean((d_i_boot - d_i) ** 2, axis=0)
        var = np.where(var > 0, var, np.nan)
        t_i = d_i / np.sqrt(var)
        t_max = np.nanmax(t_i)
        t_boot = np.nanmax((d_i_boot - d_i) / np.sqrt(var), axis=1)
        p = float(np.mean(t_boot >= t_max))
        running_p = max(running_p, p)
        worst = alive[int(np.nanargmax(t_i))]
        if p >= alpha:
            break
        pvals[names[worst]] = running_p
        order.append(names[worst])
        alive.remove(worst)
    for j in alive:
        pvals[names[j]] = 1.0 if len(alive) == 1 else max(running_p, pvals.get(names[j], 0.0))
    return MCSResult([names[j] for j in alive], pvals, order)


# ------------------------------------------------------- fev-bench style aggregation
def fill_failures(errors: np.ndarray, baseline_col: int) -> np.ndarray:
    """Replace failed (NaN) task–model entries with the baseline error (fev-bench rule)."""
    errors = np.array(errors, float, copy=True)
    base = errors[:, baseline_col][:, None]
    return np.where(np.isfinite(errors), errors, base)


def skill_score(errors: np.ndarray, baseline_col: int, clip: tuple[float, float] = (1e-2, 100.0)) -> np.ndarray:
    """``1 - geometric_mean(clip(E_model / E_baseline))`` per model column."""
    errors = np.asarray(errors, float)
    rel = errors / errors[:, [baseline_col]]
    rel = np.clip(rel, *clip)
    return 1.0 - np.exp(np.nanmean(np.log(rel), axis=0))


def win_rate(errors: np.ndarray) -> np.ndarray:
    """Average probability of beating every other model (ties = half a win)."""
    errors = np.asarray(errors, float)
    n_tasks, n_models = errors.shape
    rates = np.zeros(n_models)
    for j in range(n_models):
        wins = []
        for k in range(n_models):
            if k == j:
                continue
            ej, ek = errors[:, j], errors[:, k]
            ok = np.isfinite(ej) & np.isfinite(ek)
            wins.append(np.mean((ej[ok] < ek[ok]) + 0.5 * (ej[ok] == ek[ok])) if ok.any() else np.nan)
        rates[j] = np.nanmean(wins)
    return rates


def paired_bootstrap_ci(
    errors: np.ndarray,
    statistic: Callable[[np.ndarray], np.ndarray],
    *,
    reps: int = 1000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Point estimate and percentile CI by resampling tasks (rows) with replacement."""
    errors = np.asarray(errors, float)
    rng = np.random.default_rng(seed)
    n = errors.shape[0]
    point = statistic(errors)
    boots = np.stack([statistic(errors[rng.integers(0, n, n)]) for _ in range(reps)])
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    return point, np.nanquantile(boots, lo, axis=0), np.nanquantile(boots, hi, axis=0)


def pairwise_skill(errors: np.ndarray, names: list[str], *, reps: int = 1000, seed: int = 0):
    """Pairwise skill score of row-model vs column-model with bootstrap CIs."""
    m = len(names)
    table = {}
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            sub = errors[:, [i, j]]

            def stat(x: np.ndarray) -> np.ndarray:
                return skill_score(x, baseline_col=1)[:1]

            point, lo, hi = paired_bootstrap_ci(sub, stat, reps=reps, seed=seed)
            table[(names[i], names[j])] = (float(point[0]), float(lo[0]), float(hi[0]))
    return table


# ------------------------------------------------------------- multi-dataset ranking
_NEMENYI_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
                11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391}


def friedman_nemenyi(errors: np.ndarray) -> dict[str, object]:
    """Friedman test over datasets (rows) and the Nemenyi critical difference at α = 0.05."""
    errors = np.asarray(errors, float)
    errors = errors[np.isfinite(errors).all(axis=1)]
    n, k = errors.shape
    ranks = np.apply_along_axis(sps.rankdata, 1, errors)
    avg_ranks = ranks.mean(axis=0)
    stat, p = sps.friedmanchisquare(*errors.T) if k >= 3 and n >= 2 else (np.nan, np.nan)
    q = _NEMENYI_Q05.get(k, np.nan)
    cd = q * np.sqrt(k * (k + 1) / (6.0 * n)) if n else np.nan
    return {"avg_ranks": avg_ranks, "friedman_stat": float(stat), "friedman_p": float(p), "cd": float(cd), "n": n}


def wilcoxon_pair(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5 or np.allclose(a[ok], b[ok]):
        return np.nan
    return float(sps.wilcoxon(a[ok], b[ok]).pvalue)


# ------------------------------------------------------- calibration tests (probabilistic forecasts)
def _xlogy(x: float, y: float) -> float:
    return 0.0 if x == 0 else x * np.log(y)


def kupiec_pof(hits: np.ndarray, p: float) -> tuple[float, float, float]:
    """Kupiec (1995) proportion-of-failures LR test of unconditional coverage.

    ``hits``: boolean array, True where the event of nominal probability ``p`` occurred (e.g.
    y <= q_tau with p = tau, or y outside the central interval with p = 1 - level).
    Returns (observed rate, LR statistic, p-value ~ chi2(1)).
    """
    h = np.asarray(hits, bool)
    n, x = h.size, int(h.sum())
    if n == 0:
        return np.nan, np.nan, np.nan
    pi = x / n
    ll0 = _xlogy(n - x, 1 - p) + _xlogy(x, p)
    ll1 = _xlogy(n - x, 1 - pi) + _xlogy(x, pi)
    lr = max(0.0, -2 * (ll0 - ll1))
    return float(pi), float(lr), float(sps.chi2.sf(lr, 1))


def christoffersen(hits: np.ndarray, p: float) -> dict[str, float]:
    """Christoffersen (1998) tests on an ordered hit sequence: independence (Markov) and conditional
    coverage (= Kupiec + independence, chi2(2)). Use on one-step-ahead hits of a single series."""
    h = np.asarray(hits, bool)
    if h.size < 3:
        return {"LR_ind": np.nan, "p_ind": np.nan, "LR_cc": np.nan, "p_cc": np.nan}
    a, b = h[:-1], h[1:]
    n00, n01 = int(np.sum(~a & ~b)), int(np.sum(~a & b))
    n10, n11 = int(np.sum(a & ~b)), int(np.sum(a & b))
    p01 = n01 / max(1, n00 + n01)
    p11 = n11 / max(1, n10 + n11)
    p1 = (n01 + n11) / max(1, n00 + n01 + n10 + n11)
    ll_ind = _xlogy(n00, 1 - p01) + _xlogy(n01, p01) + _xlogy(n10, 1 - p11) + _xlogy(n11, p11)
    ll_null = _xlogy(n00 + n10, 1 - p1) + _xlogy(n01 + n11, p1)
    lr_ind = max(0.0, -2 * (ll_null - ll_ind))
    _, lr_uc, _ = kupiec_pof(h, p)
    lr_cc = lr_ind + lr_uc
    return {"LR_ind": float(lr_ind), "p_ind": float(sps.chi2.sf(lr_ind, 1)),
            "LR_cc": float(lr_cc), "p_cc": float(sps.chi2.sf(lr_cc, 2))}


def pit_uniformity(y: np.ndarray, q: np.ndarray, levels: np.ndarray) -> dict[str, float]:
    """Discretised PIT test for quantile forecasts: the K levels split the line into K+1 bins whose
    probabilities are the level gaps; Pearson chi2(K) goodness of fit of the observed bin counts.
    ``y``: [N]; ``q``: [N, K] sorted quantiles. Serial dependence (multi-step, overlapping origins)
    inflates the statistic: report it per step or on one-step forecasts."""
    y = np.asarray(y, float)
    q = np.asarray(q, float)
    ok = np.isfinite(y) & np.isfinite(q).all(axis=1)
    y, q = y[ok], q[ok]
    if y.size < 10:
        return {"chi2": np.nan, "p": np.nan, "n": int(y.size)}
    bins = (y[:, None] > q).sum(axis=1)  # 0 .. K
    counts = np.bincount(bins, minlength=q.shape[1] + 1)
    probs = np.diff(np.r_[0.0, np.asarray(levels, float), 1.0])
    expected = probs * y.size
    stat = float(np.sum((counts - expected) ** 2 / expected))
    return {"chi2": stat, "p": float(sps.chi2.sf(stat, q.shape[1])), "n": int(y.size),
            **{f"bin{k}": float(c / y.size) for k, c in enumerate(counts)}}
