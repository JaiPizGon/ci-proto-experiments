"""Rolling-origin backtesting shared by all experiments.

``run_backtest`` produces a *forecast table* (one row per series × origin × step with all
quantiles), from which ``metrics_table`` computes per-window scores. Keeping raw
forecasts lets the analysis run Diebold–Mariano tests on aligned per-target losses and
recompute any metric without re-running models.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .data import SeriesSet
from .metrics import score_forecast
from .models import LEVELS, BaseForecaster
from .protocols import Origin

QCOLS = [f"q{q:.1f}" for q in LEVELS]


def _slice_cov(cov: np.ndarray | None, start: int, stop: int) -> np.ndarray | None:
    if cov is None:
        return None
    return cov[..., start:stop]


def run_backtest(
    model: BaseForecaster,
    data: SeriesSet,
    origins: Sequence[Origin],
    *,
    context: int | None = None,
    rows: Sequence[int] | None = None,
    joint: bool = True,
    covariates: bool = True,
    calendar: bool = True,
    train_end: int | None = None,
    refit: str = "never",
    tag: str = "",
    history_transform=None,
    future_transform=None,
) -> pd.DataFrame:
    """Forecast every origin and return the long forecast table.

    ``context``: number of most recent observations given to the model (None = all).
    ``train_end``: last training position for supervised models (default: first cutoff).
    ``refit``: ``"never"`` (one fit before the test period) or ``"every"`` origin.
    ``history_transform``: optional callable ``f(history[V,T]) -> history`` (e.g. NaN injection).
    ``future_transform``: optional callable ``f(future_cov[K, ctx+H], horizon_start, cutoff) -> future_cov``
    used to perturb known covariates inside the forecast window only (covariate-noise study).
    """
    rows = list(range(data.values.shape[0])) if rows is None else list(rows)
    values = data.values[rows]
    past_all = data.past_covariates
    if past_all is not None and past_all.ndim == 3:
        past_all = past_all[rows]
    future_all = data.future_covariates
    stamps_all = data.timestamps_ns
    records = []
    fit_error = None
    if not model.zero_shot and refit == "never":
        end = origins[0].cutoff if train_end is None else train_end
        try:
            model.fit([v[:end] for v in values])
        except Exception as exc:  # noqa: BLE001  a failed fit is a result (NaN forecasts + status), not a crash
            fit_error = f"error: fit {type(exc).__name__}: {exc}"[:200]
            print(f"[warn] {model.name}: {fit_error}")

    groups = [list(range(len(rows)))] if (joint and model.multivariate) else [[i] for i in range(len(rows))]
    has_cov = covariates and model.uses_covariates and (past_all is not None or future_all is not None)
    batched = len(groups) > 1 and not has_cov  # independent univariate rows: one batched call per origin
    every = int(getattr(model, "refit_every", 0) or 0)  # recalibrate every k origins (LEAR, Lago et al. 2021)
    for k_origin, origin in enumerate(origins):
        c, h = origin.cutoff, origin.horizon
        start = 0 if context is None else max(0, c - context)
        if fit_error is not None:
            for row in range(len(rows)):
                _emit(records, model, tag, data, rows, row, c, h, start, values,
                      np.full((len(LEVELS), h), np.nan), 0.0, fit_error)
            continue
        if not model.zero_shot and (refit == "every" or (every and k_origin % every == 0 and k_origin)):
            model.fit([v[:c] for v in values])
        if batched:
            hist = values[:, start:c]
            if history_transform is not None:
                hist = history_transform(hist)
            kwargs = {"timestamps": stamps_all[start : c + h], "freq": data.freq} if calendar else {}
            tic = time.perf_counter()
            try:
                qs = model.predict_many(list(hist), h, **kwargs)
                q_all = np.concatenate([np.asarray(q, dtype=float).reshape(1, len(LEVELS), h) for q in qs])
                status = "ok"
            except Exception as exc:  # noqa: BLE001
                q_all = np.full((len(rows), len(LEVELS), h), np.nan)
                status = f"error: {type(exc).__name__}: {exc}"[:200]
            latency = (time.perf_counter() - tic) / len(rows)
            for row in range(len(rows)):
                _emit(records, model, tag, data, rows, row, c, h, start, values, q_all[row], latency, status)
            continue
        for group in groups:
            hist = values[group, start:c]
            if history_transform is not None:
                hist = history_transform(hist)
            kwargs: dict[str, object] = {}
            if covariates and model.uses_covariates:
                past = past_all
                if past is not None and past.ndim == 3:
                    # per-series covariates: [P, T] for a single row, [V, P, T] for a cohort
                    past = past[group[0]] if len(group) == 1 else past[group]
                kwargs["past_covariates"] = _slice_cov(past, start, c)
                fut = future_all
                if fut is not None and fut.ndim == 3:  # per-series known covariates [V, K, T]
                    fut = fut[rows][group[0]] if len(group) == 1 else fut[rows][group]
                fut = _slice_cov(fut, start, c + h)
                if fut is not None and future_transform is not None:
                    fut = future_transform(np.array(fut, copy=True), c - start, c)
                kwargs["future_covariates"] = fut
            if calendar:
                kwargs["timestamps"] = stamps_all[start : c + h]
                kwargs["freq"] = data.freq
            tic = time.perf_counter()
            try:
                q = model.predict(hist, h, **kwargs)
                status = "ok"
            except Exception as exc:  # recorded so failures are replaced by the baseline (fev rule)
                q = np.full((len(group), len(LEVELS), h), np.nan)
                status = f"error: {type(exc).__name__}: {exc}"[:200]
            latency = time.perf_counter() - tic
            q = np.asarray(q, dtype=float)
            for gi, row in enumerate(group):
                _emit(records, model, tag, data, rows, row, c, h, start, values, q[gi], latency / len(group), status)
    return pd.DataFrame.from_records(records)


def _emit(records, model, tag, data, rows, row, c, h, start, values, q, latency, status) -> None:
    """Append one forecast-table row per step for series ``row`` (``q``: [Q, H])."""
    y = values[row, c : c + h]
    for step in range(h):
        rec = {
            "model": model.name, "tag": tag, "series": data.ids[rows[row]], "cutoff": c,
            "origin_time": data.index[c - 1], "step": step + 1, "target_time": data.index[c + step],
            "y": y[step], "context": c - start, "latency_s": latency, "status": status,
        }
        rec.update({col: q[k, step] for k, col in enumerate(QCOLS)})
        records.append(rec)


def metrics_table(forecasts: pd.DataFrame, data: SeriesSet) -> pd.DataFrame:
    """Per (model, tag, series, cutoff) metrics. Scales use the *full* history before the
    origin, so they do not change with the context length given to the model."""
    pos = {sid: i for i, sid in enumerate(data.ids)}
    out = []
    for (model, tag, series, cutoff), g in forecasts.groupby(["model", "tag", "series", "cutoff"], sort=False):
        g = g.sort_values("step")
        y = g["y"].to_numpy(float)
        q = g[QCOLS].to_numpy(float).T
        history = data.values[pos[series], :cutoff]
        scores = score_forecast(y, q, LEVELS, history, data.season)
        scores.update(model=model, tag=tag, series=series, cutoff=cutoff, horizon=len(g),
                      context=int(g["context"].iloc[0]), latency_s=float(g["latency_s"].iloc[0]),
                      status=g["status"].iloc[0])
        out.append(scores)
    return pd.DataFrame(out)


def aggregate(metrics: pd.DataFrame, by: Sequence[str] = ("model", "tag")) -> pd.DataFrame:
    """Dataset-level summary: MASE/SQL averaged; WQL/WAPE pooled as ratios of sums."""
    rows = []
    for key, g in metrics.groupby(list(by)):
        key = key if isinstance(key, tuple) else (key,)
        rec = dict(zip(by, key, strict=True))
        for col in ("MAE", "RMSE", "sMAPE", "MASE", "SQL", "CRPS", "COV80", "COV60", "COV40", "WIDTH80", "IS80"):
            if col in g:
                rec[col] = float(np.nanmean(g[col]))
        if "WQL" in g:
            rec["WQL_pooled"] = float(np.nansum(g["WQL"] * g["abs_sum"]) / np.nansum(g["abs_sum"]))
        rec["n_windows"] = len(g)
        rec["fail_rate"] = float(np.mean(g["status"] != "ok"))
        rec["latency_s_mean"] = float(np.nanmean(g["latency_s"]))
        rows.append(rec)
    return pd.DataFrame(rows)


def save(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    for col in df.columns:  # mixed int/str columns (e.g. context_cfg = 365 | "full") -> str
        if df[col].dtype == object and df[col].map(type).nunique() > 1:
            df[col] = df[col].astype(str)
    try:
        df.to_parquet(path.with_suffix(".parquet"))
        return path.with_suffix(".parquet")
    except (ImportError, ValueError):
        df.to_csv(path.with_suffix(".csv.gz"), index=False)
        return path.with_suffix(".csv.gz")


def load(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    for candidate in (path.with_suffix(".parquet"), path.with_suffix(".csv.gz"), path):
        if candidate.exists():
            return pd.read_parquet(candidate) if candidate.suffix == ".parquet" else pd.read_csv(candidate)
    raise FileNotFoundError(path)


def aligned_losses(forecasts: pd.DataFrame, models: Sequence[str], loss: str = "abs",
                   key: Sequence[str] = ("series", "cutoff", "step")) -> pd.DataFrame:
    """Wide table of per-target losses for DM / MCS, aligned on identical targets."""
    f = forecasts.copy()
    median = f["q0.5"]
    err = f["y"] - median
    if loss == "abs":
        f["loss"] = err.abs()
    elif loss == "sq":
        f["loss"] = err**2
    elif loss == "pinball":
        levels = LEVELS
        qs = f[QCOLS].to_numpy(float)
        diff = f["y"].to_numpy(float)[:, None] - qs
        f["loss"] = (2 * np.maximum(levels * diff, (levels - 1) * diff)).mean(axis=1)
    else:
        raise ValueError(loss)
    wide = f.pivot_table(index=list(key), columns="model", values="loss")
    return wide[list(models)].dropna()
