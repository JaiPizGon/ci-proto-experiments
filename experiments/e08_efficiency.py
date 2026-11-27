"""E8 — Inference efficiency: latency, throughput and peak memory versus context length.

Adapts the efficiency protocol of S-Mamba (Neurocomputing 619, 2025: GPU memory and time
against accuracy) to inference-only foundation models, and tests TiRex-2's claim of
constant per-patch cost for recurrent mixers. Accuracy for the bubble chart is joined from E5.

Comparability. Each library ships its own inference stack (fused kernels, ``torch.compile``, CUDA
graphs, bfloat16, a separate process), so every model is measured in up to three *variants*:

* ``default`` — the library's out-of-the-box GPU path (what a user gets; ``is_default``);
* ``controlled`` — eager PyTorch in float32 without compilation or custom kernels, where the library
  allows it (``is_controlled``): differences then reflect the architecture, not the software stack;
* extra vendor paths (ci-proto's portable torch backend, TiRex's xLSTM CUDA kernels, TiRex-2 without
  CUDA graphs) and CPU.

Every row records precision, backend, compilation and process, the first-call (cold, incl.
compilation) time, and for Moirai-2 the in-worker compute time so the pipe overhead is reported
separately. FLOPs per series (``torch.utils.flop_counter``, matmul/attention only) are counted on
the eager variant at batch 1 as a hardware-independent cost measure.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from common import (checkpointed, knob, load_config, make_model, model_keys, out_dir, parse, run_isolated_per_model,
                    set_cache_limit, write_manifest)

import common
from e05_context import MAX_CONTEXT

from ciexp.profiling import MemoryProbe, count_flops, count_parameters

# (label, constructor overrides, precision, backend, compiled, is_default, is_controlled, grid, count FLOPs)
# grid: "full" = every context x horizon x batch; "reduced" = batch 1 and 32, first horizon.
_EAGER = "PyTorch eager"
VARIANTS = {
    "ci-proto": [
        ("default", {}, "bf16 autocast", "Mamba-3 fused kernels", False, True, False, "full", False),
        ("torch-bf16", {"backend": "torch"}, "bf16 autocast", _EAGER, False, False, False, "reduced", False),
        ("controlled", {"backend": "torch", "tta_overrides": {"bfloat16": False}}, "fp32", _EAGER, False,
         False, True, "reduced", True),
        ("cpu", {"device": "cpu", "backend": "torch"}, "bf16 autocast", _EAGER + " (CPU)", False, False, False,
         "cpu", False),
    ],
    "chronos-2": [
        ("default", {}, "fp32", _EAGER, False, True, True, "full", True),
        ("cpu", {"device": "cpu"}, "fp32", _EAGER + " (CPU)", False, False, False, "cpu", False),
    ],
    "timesfm-3": [
        ("default", {}, "fp32", _EAGER, False, True, True, "full", True),
        ("cpu", {"device": "cpu"}, "fp32", _EAGER + " (CPU)", False, False, False, "cpu", False),
    ],
    "timesfm-2.5": [
        ("default", {}, "fp32", "torch.compile", True, True, False, "full", False),
        ("controlled", {"torch_compile": False}, "fp32", _EAGER, False, False, True, "reduced", True),
    ],
    "tirex-2": [  # on CUDA the recurrent layers always use TiRex-2's custom kernels; eager only on CPU
        ("default", {}, "fp32", "custom CUDA recurrent kernels + CUDA graphs", False, True, False, "full", False),
        ("no-cuda-graphs", {"cuda_graphs": False}, "fp32", "custom CUDA recurrent kernels", False, False, False,
         "reduced", False),
        ("cpu", {"device": "cpu"}, "fp32", _EAGER + " (CPU)", False, False, False, "cpu", True),
    ],
    "tirex": [
        ("default", {}, "fp32", _EAGER, False, True, True, "full", True),
        ("xlstm-cuda", {"backend": "cuda"}, "bf16 recurrent weights", "xLSTM sLSTM CUDA kernels", False, False,
         False, "reduced", False),
        ("cpu", {"device": "cpu"}, "fp32", _EAGER + " (CPU)", False, False, False, "cpu", False),
    ],
    "moirai-2": [  # separate process (envs/moirai, torch 2.4): wall time includes pickling over pipes
        ("default", {}, "fp32", _EAGER + ", worker process", False, True, True, "full", True),
    ],
}
# A cell whose first call exceeds this many seconds is not repeated (cost bound): on an 8 GB WSL GPU a cell that
# outgrows VRAM spills to host memory and runs 50-100x slower, which is not a valid latency anyway. Such cells keep
# the single-call timing (status "slow"), and larger contexts at the same batch size are skipped.
CELL_BUDGET_S = 300.0
FLOP_MAX_CONTEXT = 16384  # FLOP counting re-runs the forward pass with Python hooks: keep it bounded


def _gpu_free_mb() -> float:
    """Device memory free after a cell (all processes): flags runs that shared the GPU."""
    try:
        import torch

        return torch.cuda.mem_get_info()[0] / 2**20 if torch.cuda.is_available() else float("nan")
    except Exception:  # noqa: BLE001
        return float("nan")


def param_dtype(model) -> str:
    """Dtype of the weights behind a wrapper (precision as loaded; autocast is recorded separately)."""
    try:
        import torch
    except ImportError:
        return ""
    stack, seen = [model], set()
    while stack and len(seen) < 2000:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, torch.nn.Module):
            p = next(obj.parameters(), None)
            if p is not None:
                return str(p.dtype).replace("torch.", "")
            continue
        stack += [v for v in getattr(obj, "__dict__", {}).values() if hasattr(v, "__dict__")
                  and type(v).__module__.split(".")[0] not in ("builtins", "numpy", "pandas")]
    return "remote" if hasattr(model, "remote_parameters") else ""


def random_walks(batch: int, length: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    return (np.cumsum(rng.normal(size=(batch, length)), axis=1) + 5 * np.sin(2 * np.pi * t / 24)).astype(np.float32)


def measure(model, batch: int, context: int, horizon: int, repeats: int, warmup: int, cohort: bool) -> dict:
    x = random_walks(batch, context)
    # every model forecasts the batch as independent univariate series (same work for all models)
    if cohort:
        x_list = list(x)

        def call():
            return model.predict(x, horizon, series_ids=[f"s{i}" for i in range(batch)])
    else:
        x_list = list(x)

        def call():
            return model.predict_many(x_list, horizon)
    remote = hasattr(model, "remote_last_compute")
    with MemoryProbe(model) as cold:  # first call of this shape: includes compilation / CUDA-graph capture
        call()
    if cold.wall_s > CELL_BUDGET_S:
        m = cold.result()
        return {"first_call_s": cold.wall_s, "median_latency_s": float("nan"), "p95_latency_s": float("nan"),
                "series_per_s": float("nan"), "peak_cuda_mb": float(m["gpu_alloc_mb"]),
                "peak_cuda_reserved_mb": float(m["gpu_reserved_mb"]), "peak_host_rss_mb": float(m["host_rss_peak_mb"]),
                "host_rss_delta_mb": float(m["host_rss_delta_mb"]), "_slow": True}
    for _ in range(max(0, warmup - 1)):
        call()
    times, mems, compute = [], [], []
    for _ in range(repeats):  # each repeat is timed and memory-profiled on its own
        with MemoryProbe(model) as probe:
            call()
        times.append(probe.wall_s)
        mems.append(probe.result())
        if remote:
            compute.append(model.remote_last_compute())
    times = np.asarray(times)
    mem = pd.DataFrame(mems).max()
    extra = {}
    if remote:
        extra = {"compute_median_s": float(np.median(compute)),
                 "ipc_overhead_s": float(np.median(times) - np.median(compute))}
    return {**extra, "first_call_s": cold.wall_s,
            "median_latency_s": float(np.median(times)), "p95_latency_s": float(np.quantile(times, 0.95)),
            "series_per_s": float(batch / np.median(times)), "peak_cuda_mb": float(mem["gpu_alloc_mb"]),
            "peak_cuda_reserved_mb": float(mem["gpu_reserved_mb"]),
            "peak_host_rss_mb": float(mem["host_rss_peak_mb"]), "host_rss_delta_mb": float(mem["host_rss_delta_mb"])}


def main() -> None:
    args = parse(__doc__)
    set_cache_limit(1)  # long contexts: never keep more than one model on the GPU
    cfg = load_config(args.config)
    exp = cfg["experiments"]["E8_efficiency"]
    out = out_dir(cfg, args, "E8_efficiency")
    contexts = [512, 2048, 8192] if args.quick else knob(args, "contexts", exp["contexts"])
    horizons = exp["horizons"][:1] if args.quick else knob(args, "horizons", exp["horizons"])
    batches = [1, 32] if args.quick else knob(args, "batch_sizes", exp["batch_sizes"])
    repeats, warmup = (3, 1) if args.quick else (knob(args, "repeats", exp["repeats"]), knob(args, "warmup", exp["warmup"]))
    devices = knob(args, "devices", exp["devices"])
    keys = model_keys(exp, args)
    if not args.quick and run_isolated_per_model(args, keys):
        args.models = None  # aggregate the per-model checkpoints below
    rows = []
    for key in keys:

        def compute(key=key):
            krows = []
            variants = VARIANTS.get(key, [("default", {}, "", "", False, True, False, "full", False)])
            if args.quick:
                variants = [v for v in variants if v[0] == "default"]
            variants = [v for v in variants if ("cpu" if v[7] == "cpu" else "cuda") in devices]
            variants = sorted(variants, key=lambda v: v[7] != "cpu")  # CPU first: never hit by a lost CUDA context
            state = {"cuda_lost": False}
            for label, override, precision, backend_desc, compiled, is_default, is_controlled, grid_kind, \
                    flops_on in variants:

                def run_variant(label=label, override=override, precision=precision, backend_desc=backend_desc,
                                compiled=compiled, is_default=is_default, is_controlled=is_controlled,
                                grid_kind=grid_kind, flops_on=flops_on):
                    vrows = []
                    if state["cuda_lost"] and grid_kind != "cpu":
                        print(f"skip {key}/{label}: CUDA context lost")
                        return pd.DataFrame(vrows)
                    try:
                        model = make_model(key, cfg, season=24, args=args, **override)
                    except Exception as exc:  # e.g. CUDA unavailable, kernels missing
                        print(f"skip {key}/{label}: {type(exc).__name__}: {exc}")
                        return pd.DataFrame(vrows)
                    meta = {"model": model.name, "variant": label, "device": "cpu" if grid_kind == "cpu" else "cuda",
                            "precision": precision, "backend": backend_desc, "compiled": compiled,
                            "process": "worker" if hasattr(model, "remote_parameters") else "in-process",
                            "param_dtype": param_dtype(model), "is_default": is_default,
                            "is_controlled": is_controlled, "params_M": count_parameters(model)}
                    limit = MAX_CONTEXT.get(key)
                    grid = sorted({c for c in contexts if limit is None or c <= limit} | ({limit} if limit else set()))
                    hs = horizons if grid_kind == "full" else horizons[:1]
                    bs = [b for b in batches if grid_kind == "full" or b in (1, 32)]
                    slow = set()  # (horizon, batch) whose first call exceeded CELL_BUDGET_S at a smaller context
                    for context in grid:  # every context within the native window, plus the native maximum
                        for horizon in hs:
                            for batch in bs:
                                if context * batch > 2**27:  # keep host memory bounded
                                    continue
                                if grid_kind == "cpu" and context * batch > 2**16:  # CPU: skip hour-long cells
                                    continue
                                if state["cuda_lost"] and grid_kind != "cpu":
                                    res, status = {}, "not run: CUDA context lost after an earlier OOM"
                                elif (horizon, batch) in slow:
                                    res, status = {}, f"not run: a smaller context exceeded {CELL_BUDGET_S:.0f} s"
                                else:
                                    try:
                                        res = measure(model, batch, context, horizon, repeats, warmup, cohort=False)
                                        status = "ok"
                                        if res.pop("_slow", False):
                                            status = f"slow: first call {res['first_call_s']:.0f} s > budget, not repeated"
                                            slow.add((horizon, batch))
                                        elif flops_on and batch == 1 and horizon == hs[0] and context <= FLOP_MAX_CONTEXT:
                                            try:
                                                res["gflops_per_series"] = count_flops(
                                                    model, list(random_walks(1, context)), horizon) / 1e9
                                            except Exception as exc:  # noqa: BLE001 - FLOPs are optional
                                                print(f"  flops failed {key}/{label}: {exc}")
                                    except Exception as exc:  # OOM etc. is a result: record it and free the GPU
                                        res, status = {}, f"error: {type(exc).__name__}: {str(exc)[:80]}"
                                        state["cuda_lost"] = (state["cuda_lost"] or "CUDA error" in str(exc)
                                                              or "AcceleratorError" in status)
                                        try:
                                            import torch

                                            torch.cuda.empty_cache()
                                        except Exception:  # noqa: BLE001
                                            pass
                                vrows.append({**meta, "context": context, "horizon": horizon, "batch": batch,
                                              "status": status, "gpu_free_mb_after": _gpu_free_mb(), **res})
                                print(f"  {model.name:>16s} {label:>14s} ctx={context:<8d} B={batch:<4d} {status}",
                                      flush=True)
                    del model  # one variant on the device at a time: a cached earlier variant plus the next one
                    common._MODEL_CACHE.clear()  # can exceed VRAM, and WSL/WDDM then silently spills to host
                    common._free_gpu()  # memory (100x slower), which would corrupt the latency measurements
                    return pd.DataFrame(vrows)

                # one checkpoint per variant: a stop loses at most the variant in progress
                krows += checkpointed(out / "_ckpt" / f"{key}__{label}", run_variant).to_dict("records")
            return pd.DataFrame(krows)

        rows += checkpointed(out / "_ckpt" / f"{key}", compute).to_dict("records")
    table = pd.DataFrame(rows)
    table.to_csv(out / "efficiency.csv", index=False)
    write_manifest(out, exp, args)
    if table.empty or "median_latency_s" not in table:
        print("no efficiency measurements (no model available)")
        return
    for flag in ("is_default", "is_controlled"):
        sub = table[table[flag].astype(bool)] if flag in table else table
        print(f"--- {flag}")
        print(sub.pivot_table(index=["context", "batch"], columns="model", values="median_latency_s").to_string())


if __name__ == "__main__":
    main()
