"""Persistent Moirai-2 worker: pickled requests on stdin, pickled replies on stdout.

Started by ``ciexp.models.RemoteForecaster``; never imports the main project's code.
Replies: ``{"ok": True}`` for init, an ``ndarray [V, 9, H]`` for predict, ``{"error": str}`` on failure.
"""

import os
import pickle
import sys
import time
import traceback

import numpy as np

LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _legacy_freq(freq):
    """Map pandas>=2.2 aliases (ME, QE-DEC, YE-DEC, BME, ...) to the ones this env's pandas 2.1 knows."""
    import re

    if not freq:
        return "h"
    base, sep, anchor = freq.partition("-")
    m = re.fullmatch(r"(\d*)([A-Za-z]+)", base)
    if not m:
        return freq
    n, code = m.groups()
    if code in {"ME", "QE", "YE", "BME", "SME", "BQE", "BYE", "CBME"}:
        code = code[:-1]
    code = {"Y": "A", "BY": "BA", "YS": "AS", "BYS": "BAS"}.get(code, code)
    return n + code + sep + anchor


class Moirai2:
    def __init__(self, size="small", context_length=1680):
        import torch
        from uni2ts.model.moirai2 import Moirai2Module

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.module = Moirai2Module.from_pretrained(f"Salesforce/moirai-2.0-R-{size}")
        self.context_length = context_length
        self._predictors = {}

    def predictor(self, horizon):
        if horizon not in self._predictors:
            from uni2ts.model.moirai2 import Moirai2Forecast

            model = Moirai2Forecast(module=self.module, prediction_length=horizon, context_length=self.context_length,
                                    target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
            self._predictors[horizon] = model.create_predictor(batch_size=32, device=self.device)
        return self._predictors[horizon]

    def predict(self, history, horizon, freq=None):
        import pandas as pd
        from gluonts.dataset.common import ListDataset

        f = _legacy_freq(freq)
        ds = ListDataset([{"start": pd.Period("2000-01-01", freq=f), "target": row[-self.context_length:]}
                          for row in np.atleast_2d(history)], freq=f)
        return np.stack([np.stack([fc.quantile(q) for q in LEVELS]) for fc in self.predictor(horizon).predict(ds)])


def _predict_many(self, histories, horizon, freq=None):
    import pandas as pd
    from gluonts.dataset.common import ListDataset

    f = _legacy_freq(freq)
    ds = ListDataset([{"start": pd.Period("2000-01-01", freq=f), "target": np.asarray(h)[-self.context_length:]}
                      for h in histories], freq=f)
    return [np.stack([fc.quantile(q) for q in LEVELS])[None] for fc in self.predictor(horizon).predict(ds)]


Moirai2.predict_many = _predict_many


def main():
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr  # library prints must not corrupt the pickle stream
    model = None
    last_compute = float("nan")  # seconds spent inside the last predict call (excludes pickling / pipes)
    while True:
        try:
            msg = pickle.load(stdin)
        except EOFError:
            return
        if msg["op"] == "exit":
            return
        try:
            if msg["op"] == "init":
                model = Moirai2(**msg["kwargs"])
                reply = {"ok": True}
            elif msg["op"] == "mem_reset":
                import torch

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                reply = {"ok": True}
            elif msg["op"] == "mem_peak":
                import torch

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    reply = (torch.cuda.max_memory_allocated() / 2**20, torch.cuda.max_memory_reserved() / 2**20)
                else:
                    reply = (float("nan"), float("nan"))
            elif msg["op"] == "params":
                reply = sum(p.numel() for p in model.module.parameters()) / 1e6
            elif msg["op"] == "last_compute":
                reply = last_compute
            elif msg["op"] == "flops":  # matmul/attention FLOPs of one predict_many call (torch FlopCounterMode)
                from torch.utils.flop_counter import FlopCounterMode

                counter = FlopCounterMode(display=False)
                with counter:
                    model.predict_many(msg["histories"], msg["horizon"], msg.get("freq"))
                reply = float(counter.get_total_flops())
            elif msg["op"] == "predict_many":
                import torch

                tic = time.perf_counter()
                reply = model.predict_many(msg["histories"], msg["horizon"], msg.get("freq"))
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                last_compute = time.perf_counter() - tic
            else:
                reply = model.predict(msg["history"], msg["horizon"], msg.get("freq"))
        except Exception:  # noqa: BLE001
            reply = {"error": traceback.format_exc()[-2000:]}
        pickle.dump(reply, stdout, protocol=4)
        stdout.flush()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
