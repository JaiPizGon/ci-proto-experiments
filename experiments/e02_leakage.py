"""E2 — Leakage control.

Part A: seeded synthetic processes that no pretraining corpus can contain — the original 8-process
suite (breaks, Lorenz, intermittent, ...) plus a 14-family bank of nonlinear, long-memory,
jump-diffusion, chaotic, count and structural-break generators (``ciexp.data.synthetic_bank``),
several seeds, two horizons, rolling origins. Aggregated per family and overall with the
fev-bench skill score (geometric mean vs Seasonal Naive), bootstrap CIs and Friedman–Nemenyi in
``analysis/report.py bench``.

Part B (temporal hold-out / prospective): Ember daily prices for the 23 markets with origins in
2026, i.e. after the training cut-off of every comparator whose cut-off is documented, plus the
strictly post-release window (target dates after the ci-proto release, 2026-11-12). Re-running the
script later extends Part B automatically as new Ember data are published.

ci-proto's training corpus is undisclosed ("real-world and synthetic time series"), so benchmark
scores alone cannot separate zero-shot skill from memorisation; Parts A and B can.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from common import checkpointed, knob, load_config, make_model, model_keys, out_dir, parse, write_manifest

from ciexp import data as D
from ciexp import protocols as P
from ciexp import runner as R

DEFAULT_MODELS = ["ci-proto", "chronos-2", "timesfm-3", "timesfm-2.5", "tirex-2", "tirex", "moirai-2",
                  "seasonal-naive", "naive", "autotheta"]


def part_a(cfg, args, exp, out, keys) -> pd.DataFrame:
    syn = exp["synthetic_suite"]
    seeds = syn["seeds"][:1] if args.quick else knob(args, "seeds", syn["seeds"])
    length = 2048 if args.quick else knob(args, "length", syn["length"])
    n_windows = 4 if args.quick else knob(args, "n_windows", syn["n_windows"])
    n_per_family = 2 if args.quick else int(knob(args, "n_per_family", syn.get("n_per_family", 10)))
    horizons = knob(args, "horizons", syn["horizons"])
    sets_by_seed = {}
    for seed in seeds:
        sets = [D.synthetic_suite(length=length, seed=seed)]
        if not knob(args, "suite_only", False):
            sets += D.synthetic_bank(n_per_family, length, seed)
        sets_by_seed[seed] = sets
    all_metrics = []
    for key in keys:  # model outermost: one load per model; one checkpoint per model

        def compute(key=key):
            parts = []
            for seed, sets in sets_by_seed.items():
                for ds in sets:
                    for horizon in horizons:
                        origins = P.rolling_origins(length, horizon, n_windows=n_windows)
                        model = make_model(key, cfg, season=ds.season, horizon=horizon, args=args)
                        fc = R.run_backtest(model, ds, origins, joint=False, calendar=False,
                                            tag=f"{ds.name}|seed{seed}|h{horizon}")
                        met = R.metrics_table(fc, ds)
                        met["seed"], met["horizon_cfg"], met["family"] = seed, horizon, ds.name
                        parts.append(met)
                print(f"  {key} seed {seed}: done", flush=True)
            return pd.concat(parts, ignore_index=True)

        all_metrics.append(checkpointed(out / "_ckpt" / f"partA_{key}", compute))
    metrics = pd.concat(all_metrics, ignore_index=True)
    R.save(metrics, out / "metrics")
    summary = R.aggregate(metrics, by=("model", "family", "horizon_cfg"))
    summary.to_csv(out / "summary_by_family.csv", index=False)
    old = metrics[metrics["family"] == "synthetic_suite"]
    R.aggregate(old, by=("model", "series", "horizon_cfg")).to_csv(out / "summary_by_process.csv", index=False)
    return summary


def part_b(cfg, args, exp, out, keys) -> pd.DataFrame | None:
    pb = exp.get("prospective", {})
    path = Path(args.data_dir) / "ember_daily.csv"
    if args.quick or not path.exists() or not knob(args, "prospective", True):
        return None
    ds = D.ember_daily_prices(path, end="2100-01-01", min_coverage=0)
    # keep the 23 markets of E3 (complete 2015-2025) and drop the trailing days not yet published
    full = D.ember_daily_prices(path)
    ds = ds.subset([ds.ids.index(c) for c in full.ids])
    last = int(pd.notna(ds.values).all(axis=0).nonzero()[0].max())
    ds.values, ds.index = ds.values[:, : last + 1], ds.index[: last + 1]
    tables = []
    data_end = str(ds.index[-1].date())
    windows = {"holdout_2026": (pb.get("holdout_start", "2026-01-01"), data_end),
               "post_release": (pb.get("start", "2026-11-13"), data_end)}
    print(f"  prospective: Ember data end {data_end}")
    for key in keys:

        def compute(key=key):
            parts = []
            for horizon in knob(args, "prospective_horizons", pb.get("horizons", [1, 7])):
                for label, (start, end) in windows.items():
                    origins = P.date_split_origins(ds.index, horizon, test_start=start, test_end=end, step=1)
                    if not origins:
                        print(f"  prospective {label} h={horizon}: no complete target window yet")
                        continue
                    model = make_model(key, cfg, season=ds.season, horizon=horizon, args=args)
                    fc = R.run_backtest(model, ds, origins, joint=False, calendar=True, covariates=False,
                                        tag=f"{label}|h{horizon}")
                    fc["window"], fc["horizon_cfg"] = label, horizon
                    parts.append(fc)
                    print(f"  {key} prospective {label} h={horizon}: {len(origins)} origins", flush=True)
            return pd.concat(parts, ignore_index=True) if parts else None

        fc_k = checkpointed(out / "_ckpt" / f"partB_{key}_{data_end}", compute)
        if fc_k is not None:
            tables.append(fc_k)
    if not tables:
        return None
    fc = pd.concat(tables, ignore_index=True)
    R.save(fc, out / "prospective_forecasts")
    met = R.metrics_table(fc, ds)
    summary = R.aggregate(met, by=("model", "tag"))
    summary.to_csv(out / "prospective_summary.csv", index=False)
    return summary


def audit_sets(args) -> list[tuple[D.SeriesSet, int, str]]:
    """Public datasets used elsewhere in the suite, with the audit horizon and the test-period start."""
    root = Path(args.data_dir)
    out = []
    for name in ("ETTh1", "ETTh2", "ETTm1", "ETTm2"):
        f = root / "ltsf" / f"{name}.csv"
        if f.exists():
            freq = "15min" if name.startswith("ETTm") else "h"
            ds = D.generic_csv(f, "date", ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"], freq, name=name)
            out.append((ds, 96 if freq == "15min" else 48, "2017-10-24"))   # ETT test split (12/4/4 x 30 days)
    if (root / "beijing_air_quality").exists():
        out.append((D.beijing_air_quality(root / "beijing_air_quality"), 48, "2016-03-01"))
    if (root / "bike_sharing_openml42712.csv").exists():
        bk = D.bike_sharing_uci(pd.read_csv(root / "bike_sharing_openml42712.csv"))
        bk.past_covariates = bk.future_covariates = None
        out.append((bk, 48, "2012-07-01"))
    if (root / "ember_daily.csv").exists():
        out.append((D.ember_daily_prices(root / "ember_daily.csv"), 14, "2024-01-01"))
    return out


def part_c(cfg, args, exp, out, keys) -> pd.DataFrame | None:
    """Contamination audit: error growth with lead time on public datasets.

    An honest forecaster's error grows with the lead; a model that recalls a series seen in
    pre-training keeps a flat, low error curve. Per model x dataset: RMSE at lead 1 and at the
    last lead (full context, weekly origins over the test period), growth ratio G = RMSE_H / RMSE_1,
    and relative long-lead RMSE versus the median of all TSFMs. Probe found on Beijing PM2.5:
    TimesFM-2.5 RMSE 16.5 -> 17.6 (lead 1 -> 48), ci-proto 20.4 -> 67.9, Chronos-2 21.3 -> 101.0.
    """
    if args.quick or not knob(args, "audit", True):
        return None
    from ciexp import metrics as MT

    n_max = int(knob(args, "audit_origins", 52))
    sets = audit_sets(args)
    parts = []
    for key in keys:  # model outermost, one checkpoint per model

        def compute(key=key):
            rows = []
            for ds, H, start in sets:
                T = ds.values.shape[1]
                first = int(np.searchsorted(ds.index.values, np.datetime64(pd.Timestamp(start))))
                step = max(1, (T - H - first) // n_max)
                origins = [P.Origin(c, H) for c in range(first, T - H, step)][:n_max]
                model = make_model(key, cfg, season=ds.season, horizon=H, args=args)
                fc = R.run_backtest(model, ds, origins, joint=False, calendar=False, covariates=False,
                                    tag=f"audit|{ds.name}")
                for lead in (1, H // 4, H // 2, H):
                    s = fc[fc["step"] == lead]
                    rows.append({"dataset": ds.name, "model": model.name, "lead": lead, "H": H,
                                 "RMSE": MT.rmse(s["y"].to_numpy(float), s["q0.5"].to_numpy(float)),
                                 "n": int(s["y"].notna().sum()),
                                 "fail_rate": float((s["status"] != "ok").mean())})
            print(f"  audit {key}: done", flush=True)
            return pd.DataFrame(rows)

        parts.append(checkpointed(out / "_ckpt" / f"partC_{key}", compute))
    t = pd.concat(parts, ignore_index=True)
    wide = t.pivot_table(index=["dataset", "model"], columns="lead", values="RMSE", aggfunc="first")
    summary = []
    for (dataset, model), r in wide.iterrows():
        H = int(t[t["dataset"] == dataset]["H"].iloc[0])
        summary.append({"dataset": dataset, "model": model, "RMSE_lead1": r[1], "RMSE_leadH": r[H],
                        "growth_G": r[H] / r[1] if r[1] else np.nan})
    s = pd.DataFrame(summary)
    tsfm = s[s["model"].isin(["ci-proto", "Chronos-2", "TimesFM-3", "TimesFM-2.5", "TiRex-2", "TiRex", "Moirai-2"])]
    med = tsfm.groupby("dataset")[["RMSE_leadH", "growth_G"]].median().rename(
        columns={"RMSE_leadH": "median_RMSE_leadH", "growth_G": "median_G"})
    s = s.merge(med, on="dataset")
    s["rel_RMSE_leadH"] = s["RMSE_leadH"] / s["median_RMSE_leadH"]
    # flag: error grows less than a third as much as the TSFM median and long-lead error < 60 % of median
    s["contamination_flag"] = ((s["growth_G"] - 1) < (s["median_G"] - 1) / 3) & (s["rel_RMSE_leadH"] < 0.6)
    t.to_csv(out / "audit_by_lead.csv", index=False)
    s.to_csv(out / "contamination_audit.csv", index=False)
    return s


def main() -> None:
    args = parse(__doc__)
    cfg = load_config(args.config)
    exp = cfg["experiments"]["E2_leakage_control"]
    out = out_dir(cfg, args, "E2_leakage_control")
    keys = model_keys({"models": exp.get("models", DEFAULT_MODELS)}, args)
    only = set(args.only or ["A", "B", "C"])  # e.g. --only B : refresh the prospective window only
    summary = part_a(cfg, args, exp, out, keys) if "A" in only else None
    pros = part_b(cfg, args, exp, out, keys) if "B" in only else None
    audit = part_c(cfg, args, exp, out, [k for k in keys if k not in ("autotheta", "autoets")]) \
        if "C" in only else None
    if audit is not None:
        print(audit[audit["contamination_flag"]].to_string())
    write_manifest(out, exp, args)
    if summary is not None:
        print(summary.pivot_table(index="family", columns="model", values="SQL").round(3).to_string())
    if pros is not None:
        print(pros.pivot_table(index="tag", columns="model", values="MAE").round(2).to_string())


if __name__ == "__main__":
    main()
