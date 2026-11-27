"""E6 — Data scarcity: zero-shot ci-proto versus supervised models trained on 20–100 % of the data.

Replicates Zheng et al. (Water Research X 30, 2026, 100469): training subsets of
20/40/60/80/100 % drawn as four time-continuous seasonal blocks; lead times 1–12 steps
(here 1, 3, 6, 12, 24, 48 h on hourly data); a fixed 120-step input window for the
supervised models; RMSE and R² plus peak-event RMSE. ci-proto is run with (i) the same
120-step window, (ii) a history matched to the supervised training ratio, and (iii) the
full history, which separates "more context" from "zero-shot skill".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from common import checkpointed, knob, load_config, make_model, model_keys, out_dir, parse, write_manifest

from ciexp import data as D
from ciexp import metrics as MT
from ciexp import protocols as P
from ciexp import runner as R

SUPERVISED = {"lstm", "patchtst", "gbm-quantile", "lear", "nbeats"}


def synthetic_stations(n: int = 3, seed: int = 0) -> D.SeriesSet:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2014-03-01", "2017-02-28 23:00", freq="h")
    t = np.arange(len(index))
    rows = []
    for s in range(n):
        base = np.zeros(len(t))
        e = rng.normal(size=len(t))
        for i in range(1, len(t)):
            base[i] = 0.97 * base[i - 1] + e[i]
        winter = 1 + 0.6 * np.cos(2 * np.pi * (index.dayofyear.to_numpy() - 15) / 365)
        rows.append(np.maximum(1, 60 * winter + 15 * base + 10 * np.sin(2 * np.pi * t / 24) * (1 + 0.2 * s)))
    return D.SeriesSet("synthetic_stations", np.stack(rows), index, "h", 24, [f"S{s}" for s in range(n)])


def lead_metrics(fc: pd.DataFrame, leads: list[int], thresholds: dict[str, float]) -> pd.DataFrame:
    rows = []
    for (model, tag, series), g in fc.groupby(["model", "tag", "series"]):
        for lead in leads:
            s = g[g["step"] == lead]
            y, p = s["y"].to_numpy(float), s["q0.5"].to_numpy(float)
            q = s[R.QCOLS].to_numpy(float).T
            rows.append({"model": model, "tag": tag, "series": series, "lead": lead,
                         "RMSE": MT.rmse(y, p), "MAE": MT.mae(y, p), "R2": MT.r2(y, p),
                         "peak_RMSE": MT.peak_rmse(y, p, thresholds[series]),
                         "CRPS": MT.crps_quantile(y[None], q[None], R.LEVELS)})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse(__doc__)
    cfg = load_config(args.config)
    exp = cfg["experiments"]["E6_data_scarcity"]
    out = out_dir(cfg, args, "E6_data_scarcity")
    ds = synthetic_stations() if args.quick else D.beijing_air_quality(Path(args.data_dir) / "beijing_air_quality")
    train_mask = (ds.index >= exp["train_period"][0]) & (ds.index <= pd.Timestamp(exp["train_period"][1]) + pd.Timedelta(hours=23))
    train_lo, train_hi = int(np.argmax(train_mask)), int(len(train_mask) - np.argmax(train_mask[::-1]))
    test_start = int(np.searchsorted(ds.index.values, np.datetime64(pd.Timestamp(exp["test_period"][0]))))
    H = max(exp["horizons_steps"])
    step = 24 * 14 if args.quick else int(knob(args, "origin_step", 24))
    origins = [P.Origin(c, H) for c in range(test_start, ds.values.shape[1] - H, step)]
    thresholds = {sid: float(np.nanquantile(ds.values[i, train_lo:train_hi], 0.9)) for i, sid in enumerate(ds.ids)}
    ratios = [0.2, 1.0] if args.quick else knob(args, "ratios", exp["ratios"])
    repeats = 1 if args.quick else knob(args, "repeats", exp["repeats"])
    keys = model_keys(exp, args) + (["gbm-quantile"] if args.quick else [])
    n_train = train_hi - train_lo
    tables = []
    for key in keys:  # one checkpoint per model

        def compute(key=key):
            parts = []
            if key in SUPERVISED:
                for ratio in ratios:
                    for rep in range(repeats):
                        keep = P.scarcity_subset(n_train, ratio, mode=exp["subset_mode"], seed=rep) + train_lo
                        blocks = np.split(keep, np.where(np.diff(keep) != 1)[0] + 1)
                        model = make_model(key, cfg, season=ds.season, horizon=H, lags=[1, 2, 3, 6, 12, 24, 48, 120],
                                           args=args, **({"input_size": exp["input_window"],
                                                          "max_steps": int(knob(args, "max_steps", 1000))}
                                                         if key in ("lstm", "patchtst", "nbeats") else {}))
                        model.fit([ds.values[i, b] for i in range(len(ds.ids)) for b in blocks])
                        model.zero_shot = True  # already fitted: the runner must not refit
                        fc = R.run_backtest(model, ds, origins, context=exp["input_window"], joint=False,
                                            covariates=False, calendar=False, tag=f"ratio{ratio}|rep{rep}")
                        parts.append(lead_metrics(fc, exp["horizons_steps"], thresholds).assign(ratio=ratio, rep=rep))
            else:
                contexts = {"ctx120": exp["input_window"], "full": None}
                contexts.update({f"matched{r}": int(r * n_train) for r in ratios})
                for label, ctx in contexts.items():
                    model = make_model(key, cfg, season=ds.season, horizon=H, args=args)
                    fc = R.run_backtest(model, ds, origins, context=ctx, joint=False, covariates=False,
                                        calendar=True, tag=label)
                    parts.append(lead_metrics(fc, exp["horizons_steps"], thresholds).assign(ratio=label, rep=0))
            print(f"  done {key}")
            return pd.concat(parts, ignore_index=True)

        tables.append(checkpointed(out / "_ckpt" / f"{key}", compute))
    table = pd.concat(tables, ignore_index=True)
    R.save(table, out / "lead_metrics")
    summary = table.groupby(["model", "ratio", "lead"])[["RMSE", "MAE", "R2", "peak_RMSE", "CRPS"]].mean().reset_index()
    summary.to_csv(out / "summary.csv", index=False)
    write_manifest(out, exp, args)
    print(summary.pivot_table(index=["lead"], columns=["model", "ratio"], values="RMSE").round(2).to_string())


if __name__ == "__main__":
    main()
