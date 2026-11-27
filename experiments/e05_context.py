"""E5 — Context-length scaling up to ci-proto's 2M+ window, and context inertia after regime breaks.

Part A (long real series): UCI household power, 1-minute, ≈2.07M points. Contexts
64 … 2,097,152 (powers of two); ci-proto switches automatically from the standard
checkpoint to ``model-long.ckpt`` above 16,384 observations, so the curve also measures
that hand-over. Competitors are run up to their own maximum context and flagged beyond.
The MASE/SQL scale uses the full pre-origin history, so it is identical across contexts.

Part B (synthetic regime breaks): origins placed 1–30 days after a mean / variance /
seasonality / trend break; a longer context contains more pre-break data, which can hurt
("context inertia", Neurocomputing 2026).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from common import (checkpointed, knob, load_config, make_model, model_keys, out_dir, parse, run_isolated_per_model,
                    set_cache_limit, write_manifest)

from ciexp import data as D
from ciexp import protocols as P
from ciexp import runner as R
from ciexp.profiling import MemoryProbe

# Native maximum contexts as documented by each model (verify for the versions you pin).
MAX_CONTEXT = {"chronos-2": 8192, "timesfm-2.5": 15360, "timesfm-3": 16384, "moirai-2": 1680,
               "tirex": 2048, "tirex-2": None, "ci-proto": None}  # TiRex truncates to 2048 internally (checked)
# Contexts beyond a model's native window are still run when the wrapper accepts them (the model
# then truncates internally); such rows are flagged ``exceeds_native_context`` and drawn dashed.


def cuda_peak_reset() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def cuda_peak_mb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated() / 2**20
    except ImportError:
        pass
    return float("nan")


def daily_origins(index: pd.DatetimeIndex, horizon: int, n: int) -> list[P.Origin]:
    midnight = np.where((index.hour == 0) & (index.minute == 0))[0]
    midnight = midnight[midnight + horizon <= len(index)]
    return [P.Origin(int(c), horizon) for c in midnight[-n:]]



def _free_gpu(hist):
    """Release cached GPU blocks before each very long forecast (PROTOCOL_CHANGES §14): on the 8 GB
    card, fragmentation across the 20 origins otherwise turns a feasible 262k-524k call into an OOM."""
    import gc

    import torch

    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001  a sticky CUDA error is reported by the forecast call itself
        pass
    return hist

def synthetic_long(length: int = 2**17, seed: int = 0) -> D.SeriesSet:
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    weekly = 0.5 * np.sin(2 * np.pi * t / (1440 * 7))
    daily = np.sin(2 * np.pi * t / 1440)
    noise = np.convolve(rng.normal(size=length), np.ones(30) / 30, mode="same")
    y = 2 + daily + weekly + 0.3 * noise + 0.2 * (rng.random(length) < 0.002) * rng.exponential(5, length)
    index = pd.date_range("2030-01-01", periods=length, freq="min")
    return D.SeriesSet("synthetic_minutely", y[None], index, "min", 1440, ["power"])


def part_a(cfg, args, exp, out) -> pd.DataFrame:
    a = exp["long_series"]
    ds = synthetic_long() if args.quick else D.uci_household_power(Path(args.data_dir) / "household_power_consumption.txt")
    T = ds.values.shape[1]
    contexts = P.context_grid("pow2", max_len=T - max(a["horizons"]) - 1)
    if args.quick:
        contexts = contexts[::2]
    contexts = [c for c in knob(args, "contexts", contexts) if c <= T - max(a["horizons"]) - 1]
    rows = []
    for horizon in a["horizons"][:1] if args.quick else knob(args, "horizons", a["horizons"]):
        origins = daily_origins(ds.index, horizon, 3 if args.quick else knob(args, "n_origins", a["n_origins"]))
        longest = origins[0].cutoff  # whole history before the first origin (2.05M points for UCI power)
        h_contexts = contexts + ([longest] if knob(args, "max_context", True) and longest > max(contexts) else [])
        for key in model_keys(a, args):  # one checkpoint per (horizon, model)

            def compute(key=key, horizon=horizon, origins=origins, h_contexts=h_contexts):
                krows = []
                model = make_model(key, cfg, season=ds.season, horizon=horizon, args=args)
                failed = None  # once the GPU hits OOM / a CUDA fault the context is poisoned (§14)
                for ctx in h_contexts:
                    limit = MAX_CONTEXT.get(key)
                    exceeds = limit is not None and ctx > limit
                    if failed is not None:
                        krows.append({"model": model.name, "horizon": horizon, "context": ctx,
                                      "exceeds_native_context": exceeds, "fail_rate": 1.0,
                                      "note": f"not run: GPU unusable after {failed}"})
                        continue
                    try:
                        with MemoryProbe(model) as probe:
                            fc = R.run_backtest(model, ds, origins, context=ctx, calendar=True, covariates=False,
                                                tag=f"h{horizon}|ctx{ctx}",
                                                history_transform=_free_gpu if ctx > 131072 else None)
                        wall, mem = probe.wall_s, probe.result()
                    except Exception as exc:  # noqa: BLE001  OOM / CUDA fault outside the per-origin guard
                        failed = f"{type(exc).__name__} at ctx {ctx}: {str(exc)[:120]}"
                        krows.append({"model": model.name, "horizon": horizon, "context": ctx,
                                      "exceeds_native_context": exceeds, "fail_rate": 1.0, "note": failed})
                        print(f"  {model.name:>16s} h={horizon:<5d} ctx={ctx:<8d} FAILED {failed}")
                        continue
                    met = R.metrics_table(fc, ds)
                    bad = fc["status"].astype(str)
                    if bad.str.contains("out of memory|handles_|CUDA driver error|illegal memory").any():
                        failed = f"ctx {ctx}: " + bad[bad != "ok"].iloc[0][:120]
                    krows.append({
                        "model": model.name, "horizon": horizon, "context": ctx, "exceeds_native_context": exceeds,
                        "checkpoint": ("long" if ctx > 16384 else "standard") if key.startswith("ci-proto") else "",
                        "MASE": met["MASE"].mean(), "SQL": met["SQL"].mean(), "COV80": met.get("COV80", pd.Series()).mean(),
                        "latency_s": met["latency_s"].median(), "wall_s": wall, "peak_cuda_mb": mem["gpu_alloc_mb"],
                        "peak_cuda_reserved_mb": mem["gpu_reserved_mb"], "peak_host_rss_mb": mem["host_rss_peak_mb"],
                        "fail_rate": float((met["status"] != "ok").mean()),
                    })
                    print(f"  {model.name:>16s} h={horizon:<5d} ctx={ctx:<8d} MASE={krows[-1]['MASE']:.3f}")
                if failed is not None:
                    state["restart"] = True
                return pd.DataFrame(krows)

            state = {"restart": False}
            rows += checkpointed(out / "_ckpt" / f"partA_h{horizon}_{key}", compute).to_dict("records")
            if state["restart"]:  # checkpoint saved; continue in a fresh process with a clean CUDA context
                print(f"[fatal] GPU unusable after partA_h{horizon}_{key}; exiting 75 for a fresh process")
                raise SystemExit(75)
    table = pd.DataFrame(rows)
    table.to_csv(out / "context_scaling.csv", index=False)
    return table


def part_b(cfg, args, exp, out) -> pd.DataFrame:
    b = exp["regime_breaks"]
    length = 2**15 if args.quick else 2**17
    brk = D.regime_break_location(length)
    horizon = 24
    days = knob(args, "break_days", b.get("days_after_break", [1, 2, 3, 5, 7, 10, 14, 21, 30]))
    seeds = [0] if args.quick else knob(args, "break_seeds", b.get("seeds", list(range(10))))
    origins = [P.Origin(brk + 24 * d, horizon) for d in days]
    tables = []
    keys = model_keys({"models": b.get("models", ["ci-proto", "chronos-2", "tirex-2", "seasonal-naive"])}, args)
    suites = {seed: D.synthetic_suite(length=length, seed=seed) for seed in seeds}
    for key in keys:  # model outermost: one load per model (cache limit 1 in this experiment)

        def compute(key=key):
            parts = []
            for seed, ds in suites.items():
                rows_idx = [i for i, s in enumerate(ds.ids) if s.startswith("break_")]
                model = make_model(key, cfg, season=ds.season, horizon=horizon, args=args)
                for ctx in knob(args, "break_contexts", b["contexts"]):
                    if ctx >= brk:
                        continue
                    fc = R.run_backtest(model, ds, origins, context=ctx, rows=rows_idx, joint=False,
                                        tag=f"seed{seed}|ctx{ctx}")
                    met = R.metrics_table(fc, ds)
                    met["context_cfg"], met["seed"] = ctx, seed
                    met["days_after_break"] = (met["cutoff"] - brk) // 24
                    met["exceeds_native_context"] = MAX_CONTEXT.get(key) is not None and ctx > MAX_CONTEXT[key]
                    parts.append(met)
            print(f"  regime breaks {key}: done")
            return pd.concat(parts, ignore_index=True)

        tables.append(checkpointed(out / "_ckpt" / f"partB_{key}", compute))
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(out / "regime_breaks.csv", index=False)
    return table


def main() -> None:
    args = parse(__doc__)
    set_cache_limit(1)  # long contexts: never keep more than one model on the GPU
    cfg = load_config(args.config)
    exp = cfg["experiments"]["E5_context_scaling"]
    out = out_dir(cfg, args, "E5_context_scaling")
    if not args.quick:  # one process per model; this process then only aggregates the checkpoints
        keys = list(dict.fromkeys(exp["long_series"].get("models", []) + exp["regime_breaks"].get("models", [])))
        run_isolated_per_model(args, keys)
    a = part_a(cfg, args, exp, out)
    b = part_b(cfg, args, exp, out)
    write_manifest(out, exp, args)
    print(a.pivot_table(index="context", columns="model", values="MASE").round(3).to_string())
    print(b.pivot_table(index=["series", "days_after_break"], columns=["model", "context_cfg"], values="MASE")
          .round(2).iloc[:8].to_string())


if __name__ == "__main__":
    main()
