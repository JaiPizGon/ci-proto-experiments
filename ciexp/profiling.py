"""Inference cost probes: wall time, peak GPU memory (in-process and remote workers), peak host RAM,
and parameter counts — used by E5 (context scaling) and E8 (efficiency).

GPU memory is reported two ways: ``gpu_alloc_mb`` (peak tensor memory, torch.cuda.max_memory_allocated,
comparable across models) and ``gpu_reserved_mb`` (peak memory held by the caching allocator, what
the device actually had to provide). Models running in a worker process (Moirai-2) report their own
peaks. Host RAM is the peak RSS of this process plus its children above the level before the call,
sampled every 5 ms.
"""

from __future__ import annotations

import threading
import time

import numpy as np


def _torch():
    try:
        import torch

        return torch if torch.cuda.is_available() else None
    except ImportError:
        return None


class MemoryProbe:
    """``with MemoryProbe(model) as m: ...`` then ``m.result()``."""

    def __init__(self, model=None, interval: float = 0.005) -> None:
        self.model, self.interval = model, interval
        self._peak_rss = 0
        self._stop = threading.Event()

    def _rss(self) -> int:
        import psutil

        p = psutil.Process()
        total = p.memory_info().rss
        for c in p.children(recursive=True):
            try:
                total += c.memory_info().rss
            except psutil.Error:
                pass
        return total

    def _sample(self) -> None:
        while not self._stop.is_set():
            self._peak_rss = max(self._peak_rss, self._rss())
            time.sleep(self.interval)

    def __enter__(self):
        torch = _torch()
        if torch is not None:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        if hasattr(self.model, "remote_memory_reset"):
            self.model.remote_memory_reset()
        self._base_rss = self._rss()
        self._peak_rss = self._base_rss
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        self._tic = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        torch = _torch()
        if torch is not None:
            torch.cuda.synchronize()
        self.wall_s = time.perf_counter() - self._tic
        self._stop.set()
        self._thread.join()
        self._peak_rss = max(self._peak_rss, self._rss())
        self.gpu_alloc = self.gpu_reserved = float("nan")
        if hasattr(self.model, "remote_memory_peak"):
            self.gpu_alloc, self.gpu_reserved = self.model.remote_memory_peak()
        elif torch is not None:
            self.gpu_alloc = torch.cuda.max_memory_allocated() / 2**20
            self.gpu_reserved = torch.cuda.max_memory_reserved() / 2**20

    def result(self) -> dict:
        return {"wall_s": self.wall_s, "gpu_alloc_mb": self.gpu_alloc, "gpu_reserved_mb": self.gpu_reserved,
                "host_rss_peak_mb": self._peak_rss / 2**20,
                "host_rss_delta_mb": (self._peak_rss - self._base_rss) / 2**20}


def count_parameters(model) -> float:
    """Parameters (millions) of the torch module(s) behind a wrapper; NaN when unknown (e.g. remote)."""
    if hasattr(model, "remote_parameters"):
        return model.remote_parameters()
    try:
        import torch
    except ImportError:
        return float("nan")
    seen, total = set(), 0
    stack = [model]
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, torch.nn.Module):
            total += sum(p.numel() for p in obj.parameters())
            continue
        if len(seen) > 2000:
            break
        attrs = getattr(obj, "__dict__", {})
        for v in attrs.values():
            if isinstance(v, torch.nn.Module) or (hasattr(v, "__dict__") and type(v).__module__.split(".")[0]
                                                     not in ("builtins", "numpy", "pandas")):
                stack.append(v)
    return total / 1e6 if total else float("nan")


def summarise_latency(latencies: list[float]) -> dict:
    t = np.asarray(latencies, float)
    return {"median_latency_s": float(np.median(t)), "p95_latency_s": float(np.quantile(t, 0.95)),
            "min_latency_s": float(t.min())}


def count_flops(model, histories: list, horizon: int) -> float:
    """FLOPs of one ``predict_many`` call, counted by ``torch.utils.flop_counter.FlopCounterMode``.

    Hardware- and software-independent cost measure. It counts matrix products, convolutions and
    attention (2 FLOPs per multiply-add); element-wise work (activations, gates, the element-wise part
    of recurrent scans) is not counted, so it is a lower bound dominated by the projections. Ops inside
    fused custom kernels (Mamba-3, xLSTM CUDA) or ``torch.compile`` graphs are invisible to the counter,
    so call it on an eager PyTorch variant only. NaN when the model is not a torch model.
    """
    if hasattr(model, "remote_flops"):
        return model.remote_flops(histories, horizon)
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return float("nan")
    counter = FlopCounterMode(display=False)
    with counter:
        model.predict_many(histories, horizon)
    total = counter.get_total_flops()
    return float(total) if total else float("nan")
