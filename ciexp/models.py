"""Uniform wrappers around ci-proto, competing foundation models and baselines.

Every forecaster exposes::

    predict(history [V, T], horizon, *, past_covariates=None [P, T],
            future_covariates=None [K, T+H], timestamps=None int64 ns [T+H],
            freq=None, series_ids=None, static_features=None) -> [V, Q, H]

returning the nine quantile levels 0.1, …, 0.9 used by ci-proto, Chronos-2,
TiRex-2 and TimesFM 2.5, so every metric is computed on identical levels.
Supervised baselines additionally implement ``fit(histories)``; the runner calls it
once on the training span (Energy and AI 100914: one pre-test refit, no test-time
updates) or at every origin when ``refit="every"``.

Third-party imports are lazy so that the evaluation core runs without GPUs or the
foundation-model packages. The third-party call signatures follow each package's
README at the versions pinned in ``requirements.txt``; re-check them when upgrading.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

LEVELS = np.round(np.arange(1, 10) / 10, 1)


def to_levels(qpred: np.ndarray, src: Sequence[float], dst: Sequence[float] = LEVELS) -> np.ndarray:
    """Linearly interpolate quantile forecasts ``[..., Q_src, H]`` onto ``dst`` levels."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    if src.shape == dst.shape and np.allclose(src, dst):
        return qpred
    moved = np.moveaxis(qpred, -2, -1)  # [..., H, Q]
    flat = moved.reshape(-1, moved.shape[-1])
    out = np.stack([np.interp(dst, src, row) for row in flat]).reshape(*moved.shape[:-1], len(dst))
    return np.moveaxis(out, -1, -2)


class BaseForecaster:
    name = "base"
    levels = LEVELS
    zero_shot = True
    uses_covariates = False
    multivariate = False

    def fit(self, histories: Sequence[np.ndarray], **_: object) -> BaseForecaster:
        return self

    def predict(self, history: np.ndarray, horizon: int, **kwargs: object) -> np.ndarray:
        raise NotImplementedError

    def predict_many(self, histories: Sequence[np.ndarray], horizon: int, **kwargs: object) -> list[np.ndarray]:
        """Forecast independent univariate series (possibly of different lengths): list of ``[1, Q, H]``.

        The default loops over ``predict``; batch-capable wrappers override it. ``kwargs`` may only
        hold covariate-free arguments shared by all series (``timestamps`` aligned to the end, ``freq``).
        """
        out = []
        for h in histories:
            kw = dict(kwargs)
            stamps = kw.get("timestamps")
            if stamps is not None:
                kw["timestamps"] = np.asarray(stamps)[-(np.asarray(h).shape[-1] + horizon):]
            out.append(self.predict(np.atleast_2d(h), horizon, **kw))
        return out


# ============================================================================ ci-proto
def patched_bundle(source: str, revision: str | None, overrides: Mapping[str, object]) -> Path:
    """Copy a ci-proto bundle and edit ``tta.json`` flags (README: 'Edit a local tta.json').

    Only documented fields may be changed; unknown fields are rejected by the loader.
    """
    from huggingface_hub import snapshot_download

    src = Path(source)
    if not src.exists():
        src = Path(
            snapshot_download(
                repo_id=source,
                revision=revision,
                allow_patterns=["model.ckpt", "model-long.ckpt", "tta.json", "config.json"],
            )
        )
    dst = Path(tempfile.mkdtemp(prefix="ci-proto-ablation-"))
    for item in src.iterdir():
        shutil.copy2(item.resolve(), dst / item.name)
    tta_path = dst / "tta.json"
    tta = json.loads(tta_path.read_text(encoding="utf-8"))
    unknown = set(overrides) - set(tta)
    if unknown:
        raise KeyError(f"unknown tta.json fields: {sorted(unknown)}")
    tta.update(overrides)
    tta_path.write_text(json.dumps(tta, indent=2), encoding="utf-8")
    return dst


def _force_torch_backend(forecaster) -> None:
    from continuousintelligence.ci_proto._mamba import Mamba3

    model = forecaster._model
    for cell in model.modules():
        if isinstance(cell, Mamba3):
            cell.fused = False
    model.backend = "torch"
    base_long = forecaster._long_model
    if base_long is not None and not getattr(base_long, "_torch_forced", False):
        def long_model():
            long = base_long()
            _force_torch_backend_model(long._model)
            return long

        long_model._torch_forced = True
        forecaster._long_model = long_model


def _force_torch_backend_model(model) -> None:
    from continuousintelligence.ci_proto._mamba import Mamba3

    for cell in model.modules():
        if isinstance(cell, Mamba3):
            cell.fused = False
    model.backend = "torch"


class CIProto(BaseForecaster):
    """ci-proto (Continuous Intelligence). ``mode``: joint | cohort | independent.

    * ``joint``: rows of ``history`` are forecast jointly (multivariate, README default).
    * ``cohort``: rows are independent series identified by ``series_ids``; cohort policies
      (static groups, correlated neighbours, chunked cross-learning) apply.
    * ``independent``: one call per row; no information is shared between rows.
    """

    uses_covariates = True
    multivariate = True

    def __init__(
        self,
        source: str = "ContinuousIntelligence/ci-proto",
        *,
        revision: str | None = None,
        device: str = "auto",
        tta_overrides: Mapping[str, object] | None = None,
        max_context: int | None = None,
        mode: str = "joint",
        use_covariates: bool = True,
        use_calendar: bool = True,
        categorical_past: tuple[int, ...] = (),
        categorical_future: tuple[int, ...] = (),
        name: str | None = None,
        backend: str = "auto",
    ) -> None:
        import continuousintelligence as ci

        if mode not in ("joint", "cohort", "independent"):
            raise ValueError("mode must be joint, cohort or independent")
        if tta_overrides:
            source, revision = str(patched_bundle(source, revision, tta_overrides)), None
            source_arg: object = Path(source)
        else:
            source_arg = source
        self.model = ci.load_model(source_arg, device=device, revision=revision)
        if backend == "torch":  # portable PyTorch path even where fused Mamba-3 kernels are available
            _force_torch_backend(self.model)
        self.levels = self.model.quantiles.numpy().astype(float)
        self.max_context = max_context
        self.mode = mode
        self.use_covariates = use_covariates
        self.use_calendar = use_calendar
        self.categorical_past = tuple(categorical_past)
        self.categorical_future = tuple(categorical_future)
        self.name = name or f"ci-proto[{mode}]"
        # ~1.2 GB per 35k-point series on the long checkpoint: 2^17 points per call fits an 8 GB GPU
        self.max_points_per_call = 2**17

    @property
    def backend(self) -> str:
        return self.model.backend

    def _call(self, history, horizon, past, future, stamps, freq, series_ids, static):
        import torch

        kwargs: dict[str, object] = {}
        if self.use_covariates and past is not None:
            kwargs["past_covariates"] = torch.as_tensor(past, dtype=torch.float32)
            if self.categorical_past:
                kwargs["categorical_past"] = self.categorical_past
        if self.use_covariates and future is not None:
            kwargs["future_covariates"] = torch.as_tensor(future, dtype=torch.float32)
            if self.categorical_future:
                kwargs["categorical_future"] = self.categorical_future
        if self.use_calendar and stamps is not None and freq is not None:
            kwargs["timestamps"] = torch.as_tensor(np.asarray(stamps, dtype=np.int64))
            kwargs["freq"] = freq
        if series_ids is not None:
            kwargs["series_ids"] = list(series_ids)
            if static is not None:
                kwargs["static_features"] = {k: list(v) for k, v in static.items()}
        out = self.model.forecast(torch.as_tensor(history, dtype=torch.float32), horizon=horizon, **kwargs)
        return out.numpy()

    def predict(
        self,
        history: np.ndarray,
        horizon: int,
        *,
        past_covariates: np.ndarray | None = None,
        future_covariates: np.ndarray | None = None,
        timestamps: np.ndarray | None = None,
        freq: str | None = None,
        series_ids: Sequence[str] | None = None,
        static_features: Mapping[str, Sequence[object]] | None = None,
        categorical_past: tuple[int, ...] | None = None,
        categorical_future: tuple[int, ...] | None = None,
    ) -> np.ndarray:
        if categorical_past is not None or categorical_future is not None:  # per-call (fev tasks)
            saved = self.categorical_past, self.categorical_future
            self.categorical_past = tuple(categorical_past or ())
            self.categorical_future = tuple(categorical_future or ())
            try:
                return self.predict(history, horizon, past_covariates=past_covariates,
                                    future_covariates=future_covariates, timestamps=timestamps, freq=freq,
                                    series_ids=series_ids, static_features=static_features)
            finally:
                self.categorical_past, self.categorical_future = saved
        history = np.atleast_2d(np.asarray(history, dtype=np.float32))
        if self.max_context is not None and history.shape[-1] > self.max_context:
            cut = history.shape[-1] - self.max_context
            history = history[..., cut:]
            past_covariates = None if past_covariates is None else past_covariates[..., cut:]
            future_covariates = None if future_covariates is None else future_covariates[..., cut:]
            timestamps = None if timestamps is None else timestamps[..., cut:]
        if self.mode == "independent" and not self.use_covariates_present(past_covariates, future_covariates) \
                and history.shape[0] > 1:
            return np.concatenate(self.predict_many(list(history), horizon, timestamps=timestamps, freq=freq))
        if self.mode == "independent":
            rows = []
            for v in range(history.shape[0]):
                rows.append(
                    self._call(history[v : v + 1], horizon, past_covariates, future_covariates,
                               timestamps, freq, None, None)[0]
                )
            return np.stack(rows)
        ids = None
        if self.mode == "cohort":
            ids = list(series_ids) if series_ids is not None else [f"s{v}" for v in range(history.shape[0])]
        return self._call(history, horizon, past_covariates, future_covariates, timestamps, freq,
                          ids, static_features)


    @staticmethod
    def use_covariates_present(past, future) -> bool:
        return past is not None or future is not None

    def _independent_model(self):
        """Same weights, cohort policies off (chunk_joint = joint_neighbours = 0): every series is its
        own case, forecast in packed GPU batches with no information shared between series."""
        if getattr(self, "_indep", None) is None:
            import copy

            off = {"chunk_joint": 0, "joint_neighbours": 0, "static_groups": 0}
            indep = copy.copy(self.model)
            indep._tta = {**self.model._tta, **off}
            base_long = self.model._long_model
            if base_long is not None:
                cache: dict[str, object] = {}

                def long_model():
                    if "m" not in cache:
                        m = copy.copy(base_long())
                        m._tta = {**m._tta, **off}
                        cache["m"] = m
                    return cache["m"]

                indep._long_model = long_model
            self._indep = indep
        return self._indep

    def predict_many(self, histories, horizon, *, timestamps=None, freq=None, **_):
        import torch

        if self.mode == "cohort" and len({np.asarray(h).size for h in histories}) == 1:
            # the cohort sees the whole panel at once (cross-learning across series is the point)
            q = self.predict(np.stack([np.asarray(h).reshape(-1) for h in histories]), horizon,
                             timestamps=timestamps, freq=freq)
            return [q[i : i + 1] for i in range(len(histories))]
        # joint on a single row is univariate, so 'joint' and 'independent' share the batched path
        rows = [np.asarray(h, dtype=np.float32).reshape(-1) for h in histories]
        if self.max_context:
            rows = [r[-self.max_context:] for r in rows]
        lengths = {len(r) for r in rows}
        out: list[np.ndarray | None] = [None] * len(rows)
        model = self._independent_model()
        chunks = []
        for length in lengths:  # equal lengths share a call (the long checkpoint is chosen per length)
            same = [i for i, r in enumerate(rows) if len(r) == length]
            per_call = max(1, self.max_points_per_call // max(1, length))  # GPU memory ~ rows x length
            chunks += [same[k : k + per_call] for k in range(0, len(same), per_call)]
        for idx in chunks:
            length = len(rows[idx[0]])
            kwargs: dict[str, object] = {"series_ids": [f"s{i}" for i in idx]}
            if self.use_calendar and timestamps is not None and freq is not None:
                stamps = np.asarray(timestamps, dtype=np.int64)[-(length + horizon):]
                kwargs["timestamps"] = torch.as_tensor(np.broadcast_to(stamps, (len(idx), len(stamps))).copy())
                kwargs["freq"] = freq
            q = model.forecast(torch.as_tensor(np.stack([rows[i] for i in idx])), horizon=horizon, **kwargs).numpy()
            for k, i in enumerate(idx):
                out[i] = q[k : k + 1]
        return out


# ===================================================================== other TSFMs
class Chronos2(BaseForecaster):
    """amazon/chronos-2 via chronos-forecasting>=2.2 (Ansari et al., 2025)."""

    uses_covariates = True
    multivariate = True

    def __init__(self, model_id: str = "amazon/chronos-2", device: str = "cuda", max_context: int | None = None,
                 name: str = "Chronos-2") -> None:
        from chronos import BaseChronosPipeline

        self.pipe = BaseChronosPipeline.from_pretrained(model_id, device_map=device)
        self.max_context = max_context
        self.name = name

    def predict(self, history, horizon, *, past_covariates=None, future_covariates=None, **_):
        history = np.atleast_2d(np.asarray(history, dtype=np.float32))
        T = history.shape[-1]
        if self.max_context is not None and T > self.max_context:
            cut = T - self.max_context
        else:
            cut = 0
        task: dict[str, object] = {"target": history[:, cut:]}
        if past_covariates is not None or future_covariates is not None:
            past_dict, future_dict = {}, {}
            if past_covariates is not None:
                for i, row in enumerate(np.atleast_2d(past_covariates)):
                    past_dict[f"p{i}"] = row[cut:T]
            if future_covariates is not None:
                for i, row in enumerate(np.atleast_2d(future_covariates)):
                    past_dict[f"k{i}"] = row[cut:T]
                    future_dict[f"k{i}"] = row[T : T + horizon]
            task["past_covariates"] = past_dict
            if future_dict:
                task["future_covariates"] = future_dict
        quantiles, _mean = self.pipe.predict_quantiles([task], prediction_length=horizon,
                                                       quantile_levels=LEVELS.tolist())
        q = np.asarray(quantiles[0])  # [V, H, Q]
        return np.moveaxis(q, -1, -2)

    def predict_many(self, histories, horizon, **_):
        tasks = [{"target": np.asarray(h, dtype=np.float32).reshape(-1)[-(self.max_context or 10**9):]}
                 for h in histories]
        quantiles, _mean = self.pipe.predict_quantiles(tasks, prediction_length=horizon,
                                                       quantile_levels=LEVELS.tolist(), batch_size=256)
        return [np.moveaxis(np.asarray(q), -1, -2) for q in quantiles]


class TimesFM25(BaseForecaster):
    """google/timesfm-2.5-200m-pytorch (univariate; deciles from the quantile head).

    TimesFM pads every input to the compiled ``max_context``, so the model is compiled lazily
    for the smallest bucket (1k, 2k, 4k, 8k, 15,360) that holds the requested context; the hard
    limit is context + horizon <= 16,384.
    """

    BUCKETS = (1024, 2048, 4096, 8192, 15360)

    def __init__(self, max_context: int = 15360, max_horizon: int = 1024, batch_size: int = 64,
                 name: str = "TimesFM-2.5", torch_compile: bool = True) -> None:
        import timesfm

        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch",
                                                                     torch_compile=torch_compile)
        self.max_context, self.max_horizon, self.batch_size = max_context, max_horizon, batch_size
        self._compiled: tuple[int, int] | None = None
        self.name = name

    def _ensure(self, length: int, horizon: int, n_inputs: int) -> int:
        import timesfm

        ctx = next((b for b in self.BUCKETS if b >= min(length, self.max_context)), self.BUCKETS[-1])
        hor = max(128, int(np.ceil(horizon / 128) * 128))
        if ctx + hor > 16384:
            ctx = 16384 - hor
        # TimesFM pads every call to a multiple of the batch size: use the smallest power of two that
        # holds the inputs, capped so that batch x context stays within an 8 GB GPU
        cap = max(1, min(self.batch_size, (64 * 2048) // ctx))
        bs = min(cap, 1 << max(0, int(np.ceil(np.log2(max(1, n_inputs))))))
        key = (ctx, hor, bs)
        if self._compiled != key:
            cache = self.__dict__.setdefault("_decoders", {})
            if key in cache:  # re-use an earlier compilation (compile may trigger torch.compile)
                for attr, value in cache[key].items():
                    setattr(self.model, attr, value)
            else:
                self.model.compile(timesfm.ForecastConfig(
                    max_context=ctx, max_horizon=hor, normalize_inputs=True, per_core_batch_size=bs,
                    use_continuous_quantile_head=True, force_flip_invariance=True, infer_is_positive=True,
                    fix_quantile_crossing=True))
                cache[key] = {a: getattr(self.model, a) for a in ("compiled_decode", "forecast_config",
                                                                  "global_batch_size")}
            self._compiled = key
        return ctx

    def predict_many(self, histories, horizon, **_):
        rows = [np.asarray(h, float).reshape(-1) for h in histories]
        ctx = self._ensure(max(len(r) for r in rows), horizon, len(rows))
        inputs = [_ffill(r[-ctx:]) for r in rows]
        _point, quant = self.model.forecast(horizon=horizon, inputs=inputs)
        q = np.moveaxis(np.asarray(quant)[:, :horizon, 1:10], -1, -2)  # drop the mean channel
        return [q[i : i + 1] for i in range(len(inputs))]

    def predict(self, history, horizon, **_):
        return np.concatenate(self.predict_many(list(np.atleast_2d(history)), horizon))


class TimesFM3(BaseForecaster):
    """google/timesfm-3.0-pytorch (TimesFM 3, univariate here; nine deciles)."""

    def __init__(self, model_id: str = "google/timesfm-3.0-pytorch", device: str = "cuda",
                 max_context: int | None = 16384, name: str = "TimesFM-3") -> None:
        import timesfm

        self.model = timesfm.TimesFM3Forecaster.from_pretrained(model_id, device=device)
        self.max_context = max_context
        self.name = name

    def predict(self, history, horizon, **_):
        history = np.atleast_2d(np.asarray(history, dtype=np.float32))
        if self.max_context:
            history = history[:, -self.max_context :]
        inputs = [_ffill(row[np.argmax(np.isfinite(row)):]).astype(np.float32) for row in history]
        outs = list(self.model.predict_batch(inputs, horizon, return_quantiles=True))
        return np.stack([np.asarray(o.quantiles)[:horizon].T for o in outs])  # [V, 9, H]

    def predict_many(self, histories, horizon, **_):
        inputs = [_ffill(np.asarray(h, float).reshape(-1)[-(self.max_context or 10**9):]).astype(np.float32)
                  for h in histories]
        outs = list(self.model.predict_batch(inputs, horizon, return_quantiles=True))
        return [np.asarray(o.quantiles)[:horizon].T[None] for o in outs]


class TiRex2(BaseForecaster):
    """NX-AI/TiRex-2: the xLSTM architecture that ci-proto adapts (Podest et al., 2026)."""

    uses_covariates = True
    multivariate = True

    def __init__(self, device: str = "cuda", name: str = "TiRex-2", cuda_graphs: bool | None = None) -> None:
        from tirex2 import load_model

        self.model = load_model("NX-AI/TiRex-2", device=device, cuda_graphs=cuda_graphs)
        self.name = name

    def _forecast(self, targets: list[np.ndarray], horizon: int, past=None, future=None) -> list[np.ndarray]:
        """Forecast ``horizon`` steps; beyond the checkpoint's maximum output length (``future_len``,
        320) the library silently truncates, so longer horizons are rolled out autoregressively:
        forecast ``future_len`` steps, append the median to the history, repeat (the usual treatment
        of horizon-limited TSFMs in GIFT-Eval / fev wrappers). Covariates are not rolled out."""
        import torch
        from tirex2 import TimeseriesType

        max_h = int(getattr(self.model, "future_len", horizon))
        if horizon > max_h and (past is not None or future is not None):
            raise ValueError(f"TiRex-2 rollout beyond {max_h} steps is not supported with covariates")
        hs = [np.atleast_2d(np.asarray(h, dtype=np.float32)) for h in targets]
        outs: list[np.ndarray] | None = None
        remaining = horizon
        while remaining > 0:
            step = min(remaining, max_h)
            ts = [TimeseriesType(
                target=torch.as_tensor(h),
                past_covariates=None if past is None else torch.as_tensor(past, dtype=torch.float32),
                future_covariates=None if future is None else torch.as_tensor(future, dtype=torch.float32),
            ) for h in hs]
            out = [np.asarray(o) for o in self.model.forecast(ts, prediction_length=step, output_type="numpy")]
            outs = out if outs is None else [np.concatenate([a, b], axis=-1) for a, b in zip(outs, out, strict=True)]
            remaining -= step
            if remaining:
                median = LEVELS.index(0.5) if isinstance(LEVELS, list) else int(np.argmin(np.abs(np.asarray(LEVELS) - 0.5)))
                hs = [np.concatenate([h, o[..., median, :].reshape(h.shape[0], -1).astype(np.float32)], axis=-1)
                      for h, o in zip(hs, out, strict=True)]
        return outs

    def predict(self, history, horizon, *, past_covariates=None, future_covariates=None, **_):
        return self._forecast([history], horizon, past_covariates, future_covariates)[0]

    def predict_many(self, histories, horizon, **_):
        return self._forecast(list(histories), horizon)


class TiRex(BaseForecaster):
    """NX-AI/TiRex (univariate xLSTM, NeurIPS 2025)."""

    def __init__(self, device: str = "cuda", name: str = "TiRex", backend: str = "torch") -> None:
        """``backend``: ``torch`` (library default, used for every accuracy experiment) or ``cuda``
        (sLSTM kernels of the ``xlstm`` package, bfloat16 recurrent weights; E8 timing only)."""
        from tirex import load_model

        self.model = load_model("NX-AI/TiRex", device=device, backend=backend)
        self.name = name
        self.backend = backend

    def predict(self, history, horizon, **_):
        import torch

        ctx = torch.as_tensor(np.atleast_2d(history), dtype=torch.float32)
        quantiles, _ = self.model.forecast(context=ctx, prediction_length=horizon)
        return np.moveaxis(np.asarray(quantiles), -1, -2)

    def predict_many(self, histories, horizon, **_):
        import torch

        ctx = [torch.as_tensor(np.asarray(h, dtype=np.float32).reshape(-1)) for h in histories]
        quantiles, _ = self.model.forecast(context=ctx, prediction_length=horizon)
        q = np.moveaxis(np.asarray(quantiles), -1, -2)
        return [q[i : i + 1] for i in range(len(ctx))]


class RemoteForecaster(BaseForecaster):
    """Runs a forecaster that needs an incompatible environment in a persistent subprocess.

    Moirai-2 (uni2ts) pins torch 2.4 / numpy 1.26, which cannot coexist with ci-proto's
    Mamba-3 kernels, so it lives in ``envs/moirai`` and is driven through ``envs/moirai/worker.py``
    with pickled requests over stdin/stdout. Same ``predict`` contract as every other model.
    """

    def __init__(self, env: str, model: str, name: str, **init_kwargs: object) -> None:
        import pickle
        import subprocess
        import sys

        root = Path(__file__).resolve().parents[1] / "envs" / env
        python = root / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        if not python.exists():
            raise FileNotFoundError(f"{python} missing: run `uv sync` inside {root}")
        self._pickle = pickle
        self.proc = subprocess.Popen([str(python), "-I", str(root / "worker.py")], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE)
        self.name = name
        self._send({"op": "init", "model": model, "kwargs": init_kwargs})

    def _send(self, msg: dict) -> object:
        self._pickle.dump(msg, self.proc.stdin, protocol=4)
        self.proc.stdin.flush()
        reply = self._pickle.load(self.proc.stdout)
        if isinstance(reply, dict) and "error" in reply:
            raise RuntimeError(reply["error"])
        return reply

    def predict(self, history, horizon, *, freq: str | None = None, **_):
        history = np.atleast_2d(np.asarray(history, dtype=np.float64))
        return np.asarray(self._send({"op": "predict", "history": history, "horizon": int(horizon),
                                      "freq": freq}))

    def remote_memory_reset(self) -> None:
        self._send({"op": "mem_reset"})

    def remote_memory_peak(self) -> tuple[float, float]:
        return tuple(self._send({"op": "mem_peak"}))

    def remote_parameters(self) -> float:
        return float(self._send({"op": "params"}))

    def remote_last_compute(self) -> float:
        """Seconds the worker spent computing the last ``predict_many`` (wall time minus IPC)."""
        return float(self._send({"op": "last_compute"}))

    def remote_flops(self, histories, horizon, *, freq: str | None = None) -> float:
        return float(self._send({"op": "flops", "histories": [np.asarray(h, float).reshape(-1) for h in histories],
                                 "horizon": int(horizon), "freq": freq}))

    def predict_many(self, histories, horizon, *, freq: str | None = None, **_):
        out = self._send({"op": "predict_many", "histories": [np.asarray(h, float).reshape(-1) for h in histories],
                          "horizon": int(horizon), "freq": freq})
        return [np.asarray(o) for o in out]

    def __del__(self) -> None:
        try:
            self._pickle.dump({"op": "exit"}, self.proc.stdin)
            self.proc.stdin.close()
            self.proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            pass


def moirai2(size: str = "small", context_length: int = 1680, name: str = "Moirai-2") -> BaseForecaster:
    """Moirai-2 in the isolated ``envs/moirai`` environment (falls back to in-process if uni2ts is importable)."""
    import importlib.util

    if importlib.util.find_spec("uni2ts") is not None:
        return Moirai2(size=size, context_length=context_length, name=name)
    return RemoteForecaster("moirai", "moirai-2", name, size=size, context_length=context_length)


class Moirai2(BaseForecaster):
    """Salesforce/moirai-2.0-R-small via uni2ts + GluonTS (univariate use)."""

    def __init__(self, size: str = "small", context_length: int = 1680, name: str = "Moirai-2") -> None:
        from uni2ts.model.moirai2 import Moirai2Module

        self.module = Moirai2Module.from_pretrained(f"Salesforce/moirai-2.0-R-{size}")
        self.context_length = context_length
        self.name = name

    def predict(self, history, horizon, *, freq: str | None = None, **_):
        import pandas as pd
        from gluonts.dataset.common import ListDataset
        from uni2ts.model.moirai2 import Moirai2Forecast

        model = Moirai2Forecast(module=self.module, prediction_length=horizon, context_length=self.context_length,
                                target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
        predictor = model.create_predictor(batch_size=32)
        history = np.atleast_2d(history)
        ds = ListDataset([{"start": pd.Period("2000-01-01", freq=freq or "h"), "target": row} for row in history],
                         freq=freq or "h")
        out = []
        for fc in predictor.predict(ds):
            out.append(np.stack([fc.quantile(q) for q in LEVELS]))
        return np.stack(out)


# ===================================================================== simple baselines
def _empirical_offsets(history: np.ndarray, lag: int, horizon: int, levels: np.ndarray, growth: str) -> np.ndarray:
    resid = history[lag:] - history[:-lag]
    resid = resid[np.isfinite(resid)]
    if resid.size < 5:
        return np.zeros((len(levels), horizon))
    base = np.quantile(resid - np.median(resid), levels)
    steps = np.arange(1, horizon + 1)
    factor = np.sqrt(np.ceil(steps / lag)) if growth == "seasonal" else np.sqrt(steps)
    return base[:, None] * factor[None, :]


class SeasonalNaive(BaseForecaster):
    """Repeats the last season; quantiles from empirical seasonal differences (fev-bench baseline)."""

    def __init__(self, season: int, name: str | None = None) -> None:
        self.season = season
        self.name = name or "SeasonalNaive"

    def predict(self, history, horizon, **_):
        history = np.atleast_2d(np.asarray(history, dtype=float))
        out = []
        for row in history:
            row = _ffill(row)
            m = self.season if len(row) > self.season else 1
            last = row[-m:]
            point = np.resize(last, horizon) if m > 1 else np.full(horizon, row[-1])
            out.append(point[None, :] + _empirical_offsets(row, m, horizon, self.levels, "seasonal"))
        return np.stack(out)


class Naive(BaseForecaster):
    name = "Naive"

    def predict(self, history, horizon, **_):
        history = np.atleast_2d(np.asarray(history, dtype=float))
        out = []
        for row in history:
            row = _ffill(row)
            out.append(np.full((1, horizon), row[-1]) + _empirical_offsets(row, 1, horizon, self.levels, "rw"))
        return np.stack(out)


class HistoricalMean(BaseForecaster):
    name = "HistoricalMean"

    def predict(self, history, horizon, **_):
        history = np.atleast_2d(np.asarray(history, dtype=float))
        out = []
        for row in history:
            q = np.nanquantile(row, self.levels)
            out.append(np.repeat(q[:, None], horizon, axis=1))
        return np.stack(out)


def _ffill(row: np.ndarray) -> np.ndarray:
    row = np.asarray(row, dtype=float).copy()
    ok = np.isfinite(row)
    if not ok.any():
        return np.zeros_like(row)
    first = int(np.argmax(ok))
    row[:first] = row[first]
    idx = np.where(np.isfinite(row), np.arange(len(row)), 0)
    np.maximum.accumulate(idx, out=idx)
    return row[idx]


# ==================================================================== supervised baselines
def _lag_matrix(y: np.ndarray, lags: Sequence[int], horizon: int):
    """Stacked direct-multi-horizon design: one row per (origin, step), step as the last feature."""
    y = np.asarray(y, float)
    max_lag = max(lags)
    t = np.arange(max_lag, len(y) - horizon + 1)
    if t.size == 0:
        return np.empty((0, len(lags) + 1)), np.empty(0)
    feats = np.stack([y[t - lag] for lag in lags], axis=1)
    finite = np.isfinite(feats).all(axis=1)
    X, Y = [], []
    for h in range(horizon):
        target = y[t + h]
        ok = finite & np.isfinite(target)
        X.append(np.column_stack([feats[ok], np.full(int(ok.sum()), h)]))
        Y.append(target[ok])
    return np.concatenate(X), np.concatenate(Y)


class LEAR(BaseForecaster):
    """LASSO-estimated autoregression (Lago et al., 2021), one model per step, empirical quantiles."""

    zero_shot = False

    def __init__(self, lags: Sequence[int], horizon: int, calibration_window: int | None = None,
                 name: str = "LEAR", refit_every: int = 0) -> None:
        self.lags, self.horizon, self.window, self.name = list(lags), horizon, calibration_window, name
        self.refit_every = refit_every

    def fit(self, histories, **_):
        from sklearn.linear_model import LassoLarsIC

        self.models, self.resid_q = [], []
        series = [np.asarray(h, float)[-self.window:] if self.window else np.asarray(h, float) for h in histories]
        designs = [_lag_matrix(_ffill(y), self.lags, self.horizon) for y in series]  # built once, all steps
        for h in range(self.horizon):
            Xs, Ys = [], []
            for X, Y in designs:
                sel = X[:, -1] == h
                Xs.append(X[sel, :-1])
                Ys.append(Y[sel])
            X, Y = np.concatenate(Xs), np.concatenate(Ys)
            mu, sd = X.mean(0), X.std(0) + 1e-9
            model = LassoLarsIC(criterion="aic").fit((X - mu) / sd, Y)
            resid = Y - model.predict((X - mu) / sd)
            self.models.append((model, mu, sd))
            self.resid_q.append(np.quantile(resid, self.levels))
        return self

    def predict(self, history, horizon, **_):
        history = np.atleast_2d(np.asarray(history, float))
        out = []
        for row in history:
            row = _ffill(row)
            feats = np.asarray([row[len(row) - lag] for lag in self.lags])
            q = np.empty((len(self.levels), horizon))
            for h in range(horizon):
                model, mu, sd = self.models[min(h, self.horizon - 1)]
                q[:, h] = model.predict(((feats - mu) / sd)[None])[0] + self.resid_q[min(h, self.horizon - 1)]
            out.append(np.sort(q, axis=0))
        return np.stack(out)


class PerSeries(BaseForecaster):
    """Fits one clone of a supervised model per series (local models, e.g. LEAR per market as in
    Lago et al., 2021). ``fit`` and ``predict_many`` must receive the series in the same order."""

    zero_shot = False

    def __init__(self, template: BaseForecaster) -> None:
        self.template, self.clones = template, []
        self.name = template.name
        self.refit_every = getattr(template, "refit_every", 0)

    def fit(self, histories, **kw):
        import copy

        self.clones = [copy.deepcopy(self.template).fit([h], **kw) for h in histories]
        return self

    def predict_many(self, histories, horizon, **kw):
        if len(histories) != len(self.clones):
            raise ValueError("PerSeries needs the same series, in the same order, as in fit()")
        return [m.predict(np.atleast_2d(h), horizon, **kw) for m, h in zip(self.clones, histories, strict=True)]

    def predict(self, history, horizon, **kw):
        history = np.atleast_2d(history)
        if len(self.clones) == 1 or history.shape[0] == len(self.clones):
            return np.concatenate(self.predict_many(list(history), horizon, **kw)) if history.shape[0] > 1 \
                else self.clones[0].predict(history, horizon, **kw)
        raise ValueError("PerSeries.predict: ambiguous series identity; use the batched path")


class GradientBoostingQuantile(BaseForecaster):
    """Global gradient-boosted quantile regressor on lags + step (LightGBM if installed)."""

    zero_shot = False

    def __init__(self, lags: Sequence[int], horizon: int, name: str = "GBM-quantile", max_iter: int = 300,
                 max_rows: int = 500_000, seed: int = 0) -> None:
        self.lags, self.horizon, self.name, self.max_iter = list(lags), horizon, name, max_iter
        self.max_rows, self.seed = max_rows, seed

    def fit(self, histories, **_):
        Xs, Ys = zip(*[_lag_matrix(_ffill(np.asarray(h, float)), self.lags, self.horizon) for h in histories],
                     strict=True)
        X, Y = np.concatenate(Xs), np.concatenate(Ys)
        if len(Y) > self.max_rows:  # uniform subsample keeps training time bounded
            keep = np.random.default_rng(self.seed).choice(len(Y), self.max_rows, replace=False)
            X, Y = X[keep], Y[keep]
        self.scale = np.nanmean(np.abs(Y)) + 1e-9
        X = np.column_stack([X[:, :-1] / self.scale, X[:, -1]])
        Y = Y / self.scale
        try:
            import lightgbm as lgb

            self.models = [lgb.LGBMRegressor(objective="quantile", alpha=q, n_estimators=self.max_iter,
                                             verbose=-1).fit(X, Y) for q in self.levels]
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingRegressor

            self.models = [HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=self.max_iter)
                           .fit(X, Y) for q in self.levels]
        return self

    def predict(self, history, horizon, **_):
        history = np.atleast_2d(np.asarray(history, float))
        out = []
        for row in history:
            row = _ffill(row)
            feats = np.asarray([row[len(row) - lag] for lag in self.lags]) / self.scale
            X = np.column_stack([np.repeat(feats[None], horizon, 0), np.minimum(np.arange(horizon), self.horizon - 1)])
            q = np.stack([m.predict(X) for m in self.models]) * self.scale
            out.append(np.sort(q, axis=0))
        return np.stack(out)


class StatsForecastModel(BaseForecaster):
    """AutoETS / AutoARIMA / AutoTheta / MSTL from statsforecast, intervals mapped to quantiles."""

    def __init__(self, model: str, season: int, name: str | None = None, n_jobs: int = 8) -> None:
        self.model_name, self.season, self.name, self.n_jobs = model, season, name or model, n_jobs

    def predict_many(self, histories, horizon, **kw):
        rows = [np.asarray(h, float).reshape(-1) for h in histories]
        if len({len(r) for r in rows}) == 1:
            q = self.predict(np.stack(rows), horizon, **kw)
            return [q[i : i + 1] for i in range(len(rows))]
        return super().predict_many(histories, horizon, **kw)

    def predict(self, history, horizon, *, freq: str | None = None, **_):
        import pandas as pd
        from statsforecast import StatsForecast
        from statsforecast import models as sfm

        ctor = {"AutoETS": lambda: sfm.AutoETS(season_length=self.season),
                "AutoARIMA": lambda: sfm.AutoARIMA(season_length=self.season),
                "AutoTheta": lambda: sfm.AutoTheta(season_length=self.season),
                "MSTL": lambda: sfm.MSTL(season_length=[self.season, 7 * self.season])}[self.model_name]
        history = np.atleast_2d(np.asarray(history, float))
        frames = [pd.DataFrame({"unique_id": i, "ds": np.arange(len(r)), "y": _ffill(r)}) for i, r in enumerate(history)]
        # one worker per series (pool start-up ~0.3 s; an AutoTheta fit on 4k points takes seconds): 3.4x faster
        # on 10 series, bit-identical forecasts
        n_jobs = 1 if len(history) < 4 else min(len(history), max(self.n_jobs, 12))
        sf = StatsForecast(models=[ctor()], freq=1, n_jobs=n_jobs)
        fc = sf.forecast(df=pd.concat(frames), h=horizon, level=[20, 40, 60, 80])
        col = [c for c in fc.columns if c not in ("unique_id", "ds") and "-" not in c][0]
        out = []
        for i in range(len(history)):
            f = fc[fc["unique_id"] == i]
            q = [f[f"{col}-lo-80"], f[f"{col}-lo-60"], f[f"{col}-lo-40"], f[f"{col}-lo-20"], f[col],
                 f[f"{col}-hi-20"], f[f"{col}-hi-40"], f[f"{col}-hi-60"], f[f"{col}-hi-80"]]
            out.append(np.stack([np.asarray(x) for x in q]))
        return np.stack(out)


class NeuralForecastModel(BaseForecaster):
    """Supervised deep baselines from neuralforecast (N-BEATS, PatchTST, iTransformer, LSTM, DLinear, TimesNet).

    Trained with a multi-quantile loss on the training span, then used at fixed weights
    across test origins. S-Mamba / RLMamba / DPEM are not in neuralforecast: run them from
    their official repositories with the same splits (see README, experiment E1-LTSF).
    """

    zero_shot = False

    def __init__(self, model: str, horizon: int, input_size: int, max_steps: int = 1000, name: str | None = None,
                 freq: str = "h", seed: int = 1) -> None:
        self.model_name, self.horizon, self.input_size, self.seed = model, horizon, input_size, seed
        self.max_steps, self.freq, self.name = max_steps, freq, name or model

    def fit(self, histories, **_):
        import pandas as pd
        from neuralforecast import NeuralForecast
        from neuralforecast import models as nfm
        from neuralforecast.losses.pytorch import MQLoss

        cls = getattr(nfm, self.model_name)
        extra = {}
        if self.model_name == "NBEATS":  # generic N-BEATS (identity stacks): valid for every horizon incl. h=1
            extra = {"stack_types": ["identity", "identity", "identity"]}
        model = cls(h=self.horizon, input_size=self.input_size, max_steps=self.max_steps,
                    loss=MQLoss(quantiles=LEVELS.tolist()), random_seed=self.seed, **extra)
        frames = [pd.DataFrame({"unique_id": i, "ds": np.arange(len(h)), "y": _ffill(np.asarray(h, float))})
                  for i, h in enumerate(histories)]
        self.nf = NeuralForecast(models=[model], freq=1)
        self.nf.fit(pd.concat(frames))
        return self

    def predict(self, history, horizon, **_):
        import pandas as pd

        history = np.atleast_2d(np.asarray(history, float))
        frames = [pd.DataFrame({"unique_id": i, "ds": np.arange(len(r)), "y": _ffill(r)}) for i, r in enumerate(history)]
        fc = self.nf.predict(df=pd.concat(frames))
        cols = _mq_columns(fc.columns)
        out = []
        for i in range(len(history)):
            f = fc[fc["unique_id"] == i]
            q = np.stack([np.asarray(f[c]) for c in cols])[:, :horizon]
            out.append(np.sort(q, axis=0))
        return np.stack(out)

    def predict_many(self, histories, horizon, **_):
        import pandas as pd

        frames = [pd.DataFrame({"unique_id": i, "ds": np.arange(len(np.asarray(r).reshape(-1))),
                                "y": _ffill(np.asarray(r, float).reshape(-1))}) for i, r in enumerate(histories)]
        fc = self.nf.predict(df=pd.concat(frames))
        cols = _mq_columns(fc.columns)
        out = []
        for i in range(len(histories)):
            f = fc[fc["unique_id"] == i]
            out.append(np.sort(np.stack([np.asarray(f[c]) for c in cols])[:, :horizon], axis=0)[None])
        return out


def _mq_columns(columns) -> list[str]:
    """neuralforecast MQLoss columns ``<M>-lo-80``, ``<M>-median``, ``<M>-hi-80`` sorted by quantile level."""
    levels = {}
    for c in columns:
        c = str(c)
        if c.endswith("-median"):
            levels[c] = 0.5
        elif "-lo-" in c:
            levels[c] = (1 - float(c.rsplit("-lo-", 1)[1]) / 100) / 2
        elif "-hi-" in c:
            levels[c] = (1 + float(c.rsplit("-hi-", 1)[1]) / 100) / 2
    cols = sorted(levels, key=levels.get)
    if len(cols) != len(LEVELS):
        raise ValueError(f"expected {len(LEVELS)} quantile columns, got {cols}")
    return cols


class HARRV(BaseForecaster):
    """Heterogeneous autoregressive model of realized volatility (Corsi, 2009), OLS per series.

    The econometric benchmark that zero-shot Kronos failed to beat in Finance Research
    Letters 110 (2026) 110606. Multi-step forecasts are iterated.
    """

    zero_shot = False

    def __init__(self, name: str = "HAR-RV") -> None:
        self.name = name

    @staticmethod
    def _design(rv: np.ndarray):
        X, Y = [], []
        for t in range(22, len(rv)):
            X.append([1.0, rv[t - 1], rv[t - 5 : t].mean(), rv[t - 22 : t].mean()])
            Y.append(rv[t])
        return np.asarray(X), np.asarray(Y)

    def predict(self, history, horizon, **_):
        history = np.atleast_2d(np.asarray(history, float))
        out = []
        for row in history:
            rv = _ffill(row)
            X, Y = self._design(rv)
            beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
            resid_q = np.quantile(Y - X @ beta, self.levels)
            path = list(rv)
            point = []
            for _h in range(horizon):
                x = np.array([1.0, path[-1], np.mean(path[-5:]), np.mean(path[-22:])])
                point.append(float(x @ beta))
                path.append(point[-1])
            q = np.asarray(point)[None, :] + resid_q[:, None] * np.sqrt(np.arange(1, horizon + 1))[None, :]
            out.append(np.maximum(q, 0.0))
        return np.stack(out)


# ================================================================================ registry
def build(spec: Mapping[str, object]) -> BaseForecaster:
    """Instantiate a forecaster from a config entry ``{"kind": ..., **kwargs}``."""
    spec = dict(spec)
    kind = spec.pop("kind")
    registry = {
        "ci-proto": CIProto, "chronos-2": Chronos2, "timesfm-2.5": TimesFM25, "tirex-2": TiRex2,
        "tirex": TiRex, "moirai-2": moirai2, "timesfm-3": TimesFM3, "seasonal-naive": SeasonalNaive, "naive": Naive,
        "historical-mean": HistoricalMean, "lear": LEAR, "gbm-quantile": GradientBoostingQuantile,
        "statsforecast": StatsForecastModel, "neuralforecast": NeuralForecastModel, "har-rv": HARRV,
    }
    if kind not in registry:
        raise KeyError(f"unknown model kind {kind!r}; choose from {sorted(registry)}")
    per_series = spec.pop("per_series", False)
    model = registry[kind](**spec)
    return PerSeries(model) if per_series else model
