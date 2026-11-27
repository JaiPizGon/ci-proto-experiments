"""E3 — Electricity price forecasting (EPF).

Block A replicates Lin & T. (Energy and AI 100914, 2026): Ember daily prices for 23
European markets, 2015–2025; horizons 1/7/30 days; validation 2023, test 2024–2025;
daily advancing origins; TSFM weights fixed (context only), supervised models refit once
before the test; context 30/90/180/365 days — extended here to 3 years and the full
history to exercise ci-proto's long context. Probabilistic scores are added because the
original study did not evaluate predictive distributions.

Block B replicates Miseta et al. (Energy and AI 100886, 2026): hourly EPEX-DE and
N2EX-UK, half-hourly NEMS-SG; horizons 4 and 40 steps; tuning 2020–2023, test 2024,
robustness 2025; context length selected on validation MAE from 24..2048 (step 4).

Statistical tests (DM-HLN + Holm, MCS) are computed by ``analysis/report.py`` from the
saved forecast tables.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from common import checkpointed, knob, load_config, make_model, model_keys, out_dir, parse, write_manifest

from ciexp import data as D
from ciexp import protocols as P
from ciexp import runner as R

SUPERVISED = {"lear", "lear-recal", "gbm-quantile", "nbeats", "lstm", "patchtst"}
STATS = {"autoets", "autotheta", "autoarima", "mstl"}


def synthetic_prices(n_markets: int = 4, seed: int = 0, freq: str = "D") -> D.SeriesSet:
    """Price-like panel (weekly seasonality, regime level shift in 2021–22, spikes, negatives)."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2015-01-01", "2025-12-31", freq=freq)
    t = np.arange(len(index))
    season = 7 if freq == "D" else 168
    level = 40 + 120 * ((index.year >= 2021) & (index.year <= 2022)) + 40 * (index.year >= 2023)
    rows = []
    for m in range(n_markets):
        noise = np.zeros(len(t))
        e = rng.normal(scale=8, size=len(t))
        for i in range(1, len(t)):
            noise[i] = 0.8 * noise[i - 1] + e[i]
        spikes = (rng.random(len(t)) < 0.01) * rng.exponential(80, len(t))
        rows.append(level * (1 + 0.05 * m) + 6 * np.sin(2 * np.pi * t / season) + noise + spikes - 10)
    return D.SeriesSet("synthetic_prices", np.stack(rows), index, freq, season, [f"M{m}" for m in range(n_markets)])


def _origins(index, horizon, start, end, step):
    return P.date_split_origins(index, horizon, test_start=start, test_end=end, step=step)


def run_block(ds: D.SeriesSet, *, horizons, contexts, keys, cfg, args, phases, step, lags, tag_prefix,
              val_step=None, calib=None):
    tables = []
    for horizon in horizons:
        for phase, (start, end, fit_until) in phases.items():
            origins = _origins(ds.index, horizon, start, end, val_step if (phase == "validation" and val_step)
                               else step)
            if not origins:
                continue
            fit_end = int(np.searchsorted(ds.index.values, np.datetime64(pd.Timestamp(fit_until)), side="right"))
            for key in keys:
                is_sup = key in SUPERVISED
                if key in STATS:  # local statistical models: one fixed 3-year window, no context selection
                    ctx_list = [3 * 365 * (ds.season // 7 if ds.season > 7 else 1)]
                    if phase == "validation":
                        continue
                else:
                    ctx_list = [None] if is_sup else contexts
                for ctx in ctx_list:
                    if is_sup and phase == "validation":
                        continue  # supervised models have no context to select
                    extra = {"calibration_window": calib} if (key == "lear-recal" and calib) else {}
                    model = make_model(key, cfg, season=ds.season, horizon=horizon, lags=lags, args=args, **extra)
                    tag = f"{tag_prefix}|h{horizon}|{phase}|ctx{'full' if ctx is None else ctx}"
                    fc = R.run_backtest(model, ds, origins, context=ctx, joint=False, covariates=False,
                                        calendar=True, train_end=fit_end, tag=tag)
                    fc["horizon_cfg"], fc["phase"], fc["context_cfg"] = horizon, phase, "full" if ctx is None else ctx
                    fc["model_key"] = key
                    tables.append(fc)
                    print(f"  {model.name:>18s} {tag}: {len(origins)} origins")
    return pd.concat(tables, ignore_index=True)


def main() -> None:
    args = parse(__doc__)
    cfg = load_config(args.config)
    exp = cfg["experiments"]["E3_electricity_prices"]
    out = out_dir(cfg, args, "E3_electricity_prices")

    # ------------------------------------------------------------ block A: 23 markets, daily
    a = exp["daily_23_markets"]
    if args.quick:
        ds = synthetic_prices()
        horizons, step = [1, 7], 28
        ctx_days = [30, 365, "full"]
    else:
        ds = D.ember_daily_prices(Path(args.data_dir) / "ember_daily.csv")
        horizons, step = knob(args, "horizons", a["horizons"]), int(knob(args, "step", a.get("step", 1)))
        ctx_days = knob(args, "context_days", a["context_days"])
    contexts = [None if c == "full" else int(c) for c in ctx_days]
    keys = model_keys(a, args)
    if args.quick:
        keys = keys + ["lear"]
    phases = {
        "validation": (a["splits"]["validation"][0], a["splits"]["validation"][1], a["splits"]["train_end"]),
        "test": (a["splits"]["test"][0], a["splits"]["test"][1], a["splits"]["validation"][1]),
    }
    phases = {k: v for k, v in phases.items() if k in knob(args, "phases", list(phases))}
    print(f"Block A: {ds.values.shape[0]} markets, {len(ds.index)} days")
    # model outermost, one checkpoint per model. The 17 daily tables hold ~4.3 M rows each (~1 GB in
    # memory), so they are never concatenated: each is reduced on its own to its metrics and to its test
    # forecasts at the context selected on validation MAE (the rule of analysis/report.select_contexts).
    # The per-model checkpoints in _ckpt remain the complete archive (every phase and context).
    import gc

    met_parts, test_parts, chosen_rows = [], [], []
    for key in keys:
        fc = checkpointed(out / "_ckpt" / f"daily_{key}", lambda key=key: run_block(
            ds, horizons=horizons, contexts=contexts, keys=[key], cfg=cfg, args=args, phases=phases, step=step,
            lags=a["lear_lags_days"], tag_prefix="daily",
            val_step=int(knob(args, "val_step", a.get("val_step", step))), calib=4 * 365))
        fc["context_cfg"] = fc["context_cfg"].astype(str)
        met_parts.append(R.metrics_table(fc, ds))
        test = fc[fc["phase"] == "test"]
        val = fc[fc["phase"] == "validation"]
        if not val.empty:
            err = (val["y"] - val["q0.5"]).abs()
            score = err.groupby([val["model"], val["horizon_cfg"], val["context_cfg"]]).mean().reset_index()
            best = score.sort_values(0).groupby(["model", "horizon_cfg"]).head(1)
            chosen_rows.append(best.rename(columns={0: "val_MAE"}))
            keep = test.merge(best[["model", "horizon_cfg", "context_cfg"]], how="left", indicator=True)
            test = test[(keep["_merge"] == "both").to_numpy()]
        else:  # supervised / statistical models: a single context, nothing to select
            test = test[test.groupby("horizon_cfg")["context_cfg"].transform("nunique").eq(1)
                        | (test["context_cfg"] == "full")]
        test_parts.append(test.reset_index(drop=True))
        print(f"  reduced {key}: {len(fc)} -> {len(test)} test rows", flush=True)
        del fc, test, val
        gc.collect()
    if chosen_rows:
        pd.concat(chosen_rows, ignore_index=True).to_csv(out / "selected_context_daily.csv", index=False)
    R.save(pd.concat(test_parts, ignore_index=True), out / "forecasts_daily")  # test phase, selected context
    del test_parts
    met_a = pd.concat(met_parts, ignore_index=True)
    R.save(met_a, out / "metrics_daily")
    summary = R.aggregate(met_a, by=("model", "tag"))
    summary.to_csv(out / "summary_daily.csv", index=False)

    # ------------------------------------------------------- block B: intraday, 3 markets
    b = exp["intraday_3_markets"]
    if not args.quick and knob(args, "intraday", True):
        # Public substitute for the licensed EPEX-DE / N2EX-UK / NEMS-SG series: Ember hourly day-ahead
        # prices (CC BY 4.0) for Germany, Great Britain and Spain (Iberian market instead of Singapore).
        files = b.get("markets", {"DE": "Germany.csv", "GB": "United Kingdom.csv", "ES": "Spain.csv"})
        for market, fname in files.items():
            path = Path(args.data_dir) / "ember_hourly" / fname
            if not path.exists():
                print(f"Block B: {path} missing, skipping {market} (run scripts/get_data.py ember_hourly)")
                continue
            freq = "h"
            dsm = D.ember_hourly_prices(path.parent, fname, market, start=b["period"][0], end=b["period"][1])
            keys_b = model_keys(b, args) if not args.models else list(args.models)
            day = 24 if freq == "h" else 48
            lags = [1, 2, 3, day, 2 * day, 7 * day]
            # 1) validation sweep over the context grid (late 2022 + 2023, as in the single-fold scheme)
            val = {"validation": tuple(knob(args, "val_period", ["2022-07-01", "2023-12-31", "2022-06-30"]))}
            grid = P.context_grid("epf_grid")[:: max(1, int(knob(args, "grid_stride", b.get("grid_stride", 1))))]
            print(f"Block B {market}: {len(grid)} contexts on the validation grid")
            fc_val = pd.concat([checkpointed(out / "_ckpt" / f"{market}_val_{k}", lambda k=k: run_block(
                dsm, horizons=knob(args, "horizons_b", b["horizons_steps"]), contexts=grid + [None], keys=[k],
                cfg=cfg, args=args, phases=val, step=int(knob(args, "val_step_b", 7 * day)), lags=lags,
                tag_prefix=market)) for k in keys_b if k not in SUPERVISED and k not in STATS], ignore_index=True)
            met_val = R.metrics_table(fc_val, dsm).merge(
                fc_val[["tag", "model", "model_key", "horizon_cfg", "context_cfg"]].drop_duplicates(),
                on=["tag", "model"])
            met_val["context_cfg"] = met_val["context_cfg"].astype(str)
            best = (met_val.groupby(["model_key", "horizon_cfg", "context_cfg"])["MAE"].mean().reset_index()
                    .sort_values("MAE").groupby(["model_key", "horizon_cfg"]).head(1))
            best.to_csv(out / f"selected_context_{market}.csv", index=False)
            # 2) test (2024) and robustness (2025) with the selected context and the full history
            ph = {"test": (b["splits"]["test"][0], b["splits"]["test"][1], b["splits"]["tune"][1]),
                  "robustness": (b["splits"]["robustness"][0], b["splits"]["robustness"][1], b["splits"]["tune"][1])}
            pieces = []
            for key in keys_b:  # model outermost; one checkpoint per model and market

                def test_key(key=key):
                    parts = []
                    for horizon in knob(args, "horizons_b", b["horizons_steps"]):
                        if key in SUPERVISED or key in STATS:
                            ctxs = [None]
                        else:
                            sel = best[(best["horizon_cfg"] == horizon) & (best["model_key"] == key)]
                            chosen = [c for c in sel["context_cfg"] if c != "full"]
                            ctxs = ([int(chosen[0])] if chosen else []) + [None]
                        parts.append(run_block(dsm, horizons=[horizon], contexts=ctxs, keys=[key], cfg=cfg,
                                               args=args, phases=ph,
                                               step=int(knob(args, "test_step", b.get("test_step", day))),
                                               lags=lags, tag_prefix=market, calib=4 * 365 * day))
                    return pd.concat(parts, ignore_index=True)

                pieces.append(checkpointed(out / "_ckpt" / f"{market}_test_{key}", test_key))
            fc_b = pd.concat(pieces, ignore_index=True)
            R.save(fc_b, out / f"forecasts_{market}")
            R.save(R.metrics_table(fc_b, dsm), out / f"metrics_{market}")

    write_manifest(out, exp, args)
    test = summary[summary["tag"].str.contains("test")]
    print(test.pivot_table(index="tag", columns="model", values="MAE").round(2).to_string())


if __name__ == "__main__":
    main()
