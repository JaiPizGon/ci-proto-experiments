"""Adapters that plug any ``ciexp`` forecaster into the official benchmark harnesses.

* ``run_fev_task`` / ``run_fev_benchmark`` drive fev-bench through the ``fev`` package
  (autogluon/fev >= 0.10) and return fev evaluation summaries, so results are directly
  comparable with the official leaderboard files (benchmarks/fev_bench/results/*.csv).
* ``GluonTSAdapter`` is a GluonTS-style predictor for the GIFT-Eval evaluation notebook.

Covariates: numeric known/past dynamic columns are passed as ``future_covariates`` /
``past_covariates``; string/categorical columns are integer-coded per item (codes shared between
past and future) and flagged as categorical for ci-proto. Models without covariate support get the
target only. Univariate models receive the items of a window as one batch (``predict_many``).
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator

import numpy as np

from .models import LEVELS, BaseForecaster


def _encode(past_vals, future_vals) -> tuple[np.ndarray, bool]:
    """Numeric array for one covariate over past+future; categorical columns are integer-coded."""
    vals = list(past_vals) + list(future_vals)
    try:
        return np.asarray(vals, dtype=float), False
    except (TypeError, ValueError):
        codes: dict[object, int] = {}
        return np.asarray([np.nan if v is None else codes.setdefault(v, len(codes)) for v in vals], float), True


def _window_items(past, future, task):
    """Per-item arrays: id, target [V, T], past covariates [P, T], known covariates [K, T+H], stamps, cats."""
    import pandas as pd

    id_col, ts_col = task.id_column, task.timestamp_column
    targets = list(task.target_columns)
    known = list(task.known_dynamic_columns or [])
    past_only = list(task.past_dynamic_columns or [])
    future_by_id = {row[id_col]: row for row in future}
    items = []
    for row in past:
        item_id = row[id_col]
        hist = np.stack([np.asarray(row[c], dtype=float) for c in targets])
        fut = future_by_id[item_id]
        stamps = pd.DatetimeIndex(list(row[ts_col]) + list(fut[ts_col]))
        known_arr, cat_f = None, []
        if known:
            enc = [_encode(row[c], fut[c]) for c in known]
            known_arr = np.stack([e[0] for e in enc])
            cat_f = [i for i, e in enumerate(enc) if e[1]]
        past_arr, cat_p = None, []
        if past_only:
            enc = [_encode(row[c], []) for c in past_only]
            past_arr = np.stack([e[0] for e in enc])
            cat_p = [i for i, e in enumerate(enc) if e[1]]
        items.append({"id": item_id, "hist": hist, "past": past_arr, "known": known_arr, "stamps": stamps,
                      "cat_past": tuple(cat_p), "cat_future": tuple(cat_f)})
    return items, targets


def _as_fev(q: np.ndarray, quantile_levels) -> dict:
    """[Q, H] on ciexp LEVELS -> fev prediction dict (``predictions`` = median)."""
    out = {"predictions": q[int(np.argmin(np.abs(LEVELS - 0.5)))].astype(float).tolist()}
    for level in quantile_levels:
        out[str(level)] = np.array([np.interp(level, LEVELS, q[:, h]) for h in range(q.shape[1])]).tolist()
    return out


def predict_window(model: BaseForecaster, past, future, task, *, use_covariates: bool = True,
                   timestamps: bool = True) -> dict[str, list[dict]] | list[dict]:
    items, targets = _window_items(past, future, task)
    H, freq = task.horizon, task.freq
    has_cov = use_covariates and model.uses_covariates and any(
        it["past"] is not None or it["known"] is not None for it in items)
    results: list[np.ndarray] = [None] * len(items)  # each [V, Q, H]
    if len(targets) == 1 and not has_cov:
        # univariate, no covariates: one batch per window (equal-length items share a call inside wrappers)
        kw = {}
        if timestamps:
            longest = max(items, key=lambda it: it["hist"].shape[-1])
            kw = {"timestamps": longest["stamps"].asi8, "freq": freq}
            same_axis = all(it["stamps"][-1] == longest["stamps"][-1] for it in items)
            if not same_axis:
                kw = {}
        qs = model.predict_many([it["hist"][0] for it in items], H, **kw)
        results = [np.asarray(q).reshape(1, len(LEVELS), H) for q in qs]
    else:
        for i, it in enumerate(items):
            kw = {"timestamps": it["stamps"].asi8, "freq": freq} if timestamps else {}
            if has_cov:
                kw.update(past_covariates=it["past"], future_covariates=it["known"])
                if it["cat_past"] or it["cat_future"]:
                    kw.update(categorical_past=it["cat_past"], categorical_future=it["cat_future"])
            if len(targets) > 1 and not model.multivariate:
                qs = model.predict_many(list(it["hist"]), H)
                results[i] = np.concatenate([np.asarray(q).reshape(1, len(LEVELS), H) for q in qs])
            else:
                results[i] = np.asarray(model.predict(it["hist"], H, **kw)).reshape(len(targets), len(LEVELS), H)
    per_target = {t: [_as_fev(results[i][v], task.quantile_levels) for i in range(len(items))]
                  for v, t in enumerate(targets)}
    return per_target if len(targets) > 1 else per_target[targets[0]]


def run_fev_task(model: BaseForecaster, task, **kw) -> dict:
    preds, tic = [], time.perf_counter()
    for window in task.iter_windows():
        past, future = window.get_input_data()
        preds.append(predict_window(model, past, future, task, **kw))
    elapsed = time.perf_counter() - tic
    return task.evaluation_summary(preds, model_name=model.name, inference_time_s=elapsed,
                                   trained_on_this_dataset=False)


CUDA_FATAL_EXIT = 75


def is_fatal_cuda(exc: BaseException) -> bool:
    """A CUDA driver fault (seen under WSL: "device not ready", then ``!handles_.at(i)`` asserts) poisons the CUDA
    context, so every later task in the process fails too. Callers abort and let a fresh process resume."""
    msg = str(exc)
    return any(s in msg for s in ("CUDA driver error", "handles_.at(i)", "CUDA error: unspecified launch failure",
                                  "CUDA error: an illegal memory access"))


def fatal_list(path) -> set[str]:
    """Names of tasks that crashed the CUDA driver in an earlier process (skipped, then imputed as failures)."""
    from pathlib import Path

    p = Path(path)
    return set(p.read_text().split("\n")) - {""} if p.exists() else set()


def record_fatal(path, name: str) -> None:
    from pathlib import Path

    with Path(path).open("a") as fh:
        fh.write(name + "\n")


def run_fev_benchmark(model: BaseForecaster, tasks_yaml: str, *, max_tasks: int | None = None,
                      done: set[str] | None = None, on_result=None, fatal_path=None, **kw) -> list[dict]:
    """Evaluate ``model`` on a fev benchmark definition; ``done`` skips task names already evaluated."""
    import fev

    benchmark = fev.Benchmark.from_yaml(tasks_yaml)
    summaries = []
    for k, task in enumerate(benchmark.tasks):
        if max_tasks is not None and k >= max_tasks:
            break
        if done and task.task_name in done:
            continue
        if fatal_path is not None and task.task_name in fatal_list(fatal_path):
            print(f"  [skip] {task.task_name}: crashed the CUDA driver in an earlier attempt", flush=True)
            continue
        try:
            summary = run_fev_task(model, task, **kw)
        except Exception as exc:  # noqa: BLE001  recorded; fev rule imputes failures with Seasonal Naive
            print(f"  [fail] {task.task_name}: {type(exc).__name__}: {exc}"[:300], flush=True)
            if is_fatal_cuda(exc):
                if fatal_path is not None:
                    record_fatal(fatal_path, task.task_name)
                print("  CUDA context lost; aborting so a fresh process can resume", flush=True)
                raise SystemExit(CUDA_FATAL_EXIT) from exc
            continue
        summaries.append(summary)
        if on_result is not None:
            on_result(summary)
        print(f"  {model.name:>18s} {task.task_name:<40s} SQL={summary.get('SQL', float('nan')):.4f}"
              f" ({summary.get('inference_time_s', 0):.0f}s)")
    return summaries


# ----------------------------------------------------------------------------- GIFT-Eval
class GluonTSAdapter:
    """Minimal GluonTS predictor: yields ``QuantileForecast`` objects (univariate entries).

    Use with GIFT-Eval's ``Dataset(name, term, to_univariate=True)`` and
    ``gluonts.model.evaluate_model(adapter, test_data=ds.test_data, metrics=..., seasonality=...)``.
    The ``mean`` key is filled with the median so mean-based metrics (MSE) are defined.
    """

    def __init__(self, model: BaseForecaster, prediction_length: int, freq: str | None = None) -> None:
        self.model, self.prediction_length, self.freq = model, prediction_length, freq

    def predict(self, dataset: Iterable[dict], **_: object) -> Iterator[object]:
        from gluonts.model.forecast import QuantileForecast

        for entry in dataset:
            target = np.asarray(entry["target"], dtype=float)
            hist = np.atleast_2d(target)
            start = entry["start"]
            q = np.asarray(self.model.predict(hist, self.prediction_length, freq=self.freq))[0]  # [Q, H]
            arrays = np.vstack([q, q[4:5]])
            keys = [str(level) for level in LEVELS] + ["mean"]
            yield QuantileForecast(forecast_arrays=arrays, start_date=start + hist.shape[-1], forecast_keys=keys,
                                   item_id=entry.get("item_id"))
