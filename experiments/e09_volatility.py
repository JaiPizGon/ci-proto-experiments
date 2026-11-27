"""E9 (optional) — Negative control: next-day volatility of equities.

Replicates the design of Zhong, Fu & Zhu (Finance Research Letters 110, 2026, 110606):
target |r_{t+1}| (absolute daily return as the realised-volatility proxy), one-step-ahead,
strictly out-of-sample with lagged information only, evaluated per calendar year with
RMSE and QLIKE; HAR-RV as the econometric benchmark; DM tests and MCS in the analysis.
Input: a CSV with columns ``date, ticker, close`` (licensed data are not redistributed).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from common import knob, load_config, make_model, model_keys, out_dir, parse, write_manifest

from ciexp import data as D
from ciexp import metrics as MT
from ciexp import protocols as P
from ciexp import runner as R


def abs_returns(path: Path, calendar_ticker: str = "SP500", drop: tuple[str, ...] = ("CBBTCUSD",)) -> D.SeriesSet:
    """|daily return| per asset on the NYSE trading calendar (dates on which ``calendar_ticker`` trades).

    Returns are computed on each asset's own calendar before alignment; 7-day assets are dropped.
    """
    df = pd.read_csv(path, parse_dates=["date"]).sort_values(["ticker", "date"])
    df = df[~df["ticker"].isin(drop)]
    df["absret"] = df.groupby("ticker")["close"].pct_change().abs()
    wide = df.pivot_table(index="date", columns="ticker", values="absret")
    if calendar_ticker in wide.columns:
        wide = wide[wide[calendar_ticker].notna()]
    wide = wide.loc[:, wide.notna().mean() > 0.9]
    return D.SeriesSet("abs_returns", wide.to_numpy(float).T, pd.DatetimeIndex(wide.index), "B", 5,
                       [str(c) for c in wide.columns])


def synthetic_garch(n_assets: int = 20, seed: int = 0) -> D.SeriesSet:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2019-01-01", "2025-12-31")
    rows = []
    for _ in range(n_assets):
        sigma2, r = np.empty(len(index)), np.empty(len(index))
        sigma2[0] = 1e-4
        for t in range(len(index)):
            if t:
                sigma2[t] = 2e-6 + 0.08 * r[t - 1] ** 2 + 0.9 * sigma2[t - 1]
            r[t] = np.sqrt(sigma2[t]) * rng.standard_t(4) / np.sqrt(2)
        rows.append(np.abs(r))
    return D.SeriesSet("synthetic_garch", np.stack(rows), index, "B", 5, [f"A{i}" for i in range(n_assets)])


def _losses(g: pd.DataFrame) -> pd.DataFrame:
    p = np.maximum(g["q0.5"].to_numpy(float), 1e-8)
    y = g["y"].to_numpy(float)
    ok = y > 1e-12
    ratio = np.where(ok, y / p, np.nan)
    return g.assign(sq=(y - p) ** 2, qlike=ratio - np.log(ratio) - 1)


def tests(fc: pd.DataFrame, ref: str = "HAR-RV") -> pd.DataFrame:
    """DM-HLN (h = 1) of every model against HAR-RV per asset and year, Holm across assets."""
    from ciexp import stats as S

    fc = _losses(fc)
    rows = []
    for (year, series), g in fc.groupby(["year", "series"]):
        base = g[g["model"] == ref].set_index("cutoff")
        for model, gm in g.groupby("model"):
            if model == ref:
                continue
            for loss in ("sq", "qlike"):
                a, b = gm.set_index("cutoff")[loss].align(base[loss], join="inner")
                res = S.diebold_mariano(a.to_numpy(), b.to_numpy(), 1, loss=lambda x: x)
                rows.append({"year": year, "series": series, "model": model, "loss": loss, "DM": res.statistic,
                             "p_model_better": res.p_less, "p_two": res.p_two_sided, "n": res.n})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm"] = out.groupby(["year", "model", "loss"])["p_model_better"].transform(S.holm)
    return out


def mcs(fc: pd.DataFrame) -> pd.DataFrame:
    """90 % Model Confidence Set per year on the cross-asset mean QLIKE and squared loss."""
    from ciexp import stats as S

    fc = _losses(fc)
    rows = []
    for year, g in fc.groupby("year"):
        for loss in ("sq", "qlike"):
            wide = g.pivot_table(index=["cutoff"], columns="model", values=loss, aggfunc="mean").dropna()
            if len(wide) < 30:
                continue
            res = S.model_confidence_set(wide.to_numpy(), list(wide.columns), alpha=0.10, reps=1000)
            rows += [{"year": year, "loss": loss, "model": m, "in_MCS90": m in res.included,
                      "mcs_p": res.pvalues.get(m)} for m in wide.columns]
    return pd.DataFrame(rows)


def main() -> None:
    args = parse(__doc__)
    cfg = load_config(args.config)
    exp = cfg["experiments"]["E9_volatility_negative_control"]
    out = out_dir(cfg, args, "E9_volatility")
    args._exp = "E9_volatility_negative_control"  # profile knobs live under the config key
    ds = synthetic_garch(5 if args.quick else 20) if args.quick or not (Path(args.data_dir) / "equities.csv").exists() \
        else abs_returns(Path(args.data_dir) / "equities.csv")
    keys = model_keys(exp, args) + (["har-rv"] if args.quick else [])
    tables = []
    for year in exp["years"][-1:] if args.quick else knob(args, "years", exp["years"]):
        step = 20 if args.quick else int(knob(args, "step", 1))
        origins = P.date_split_origins(ds.index, 1, test_start=f"{year}-01-01", test_end=f"{year}-12-31", step=step)
        for key in keys:
            model = make_model(key, cfg, season=ds.season, horizon=1, args=args)
            ctx = None if key in ("har-rv", "seasonal-naive", "naive") else 2048  # same window for every TSFM
            fc = R.run_backtest(model, ds, origins, context=ctx,
                                joint=False, covariates=False, calendar=False, tag=str(year))
            fc["year"] = year
            tables.append(fc)
    fc = pd.concat(tables, ignore_index=True)
    R.save(fc, out / "forecasts")
    rows = []
    for (model, year), g in fc.groupby(["model", "year"]):
        y, p = g["y"].to_numpy(float), np.maximum(g["q0.5"].to_numpy(float), 1e-8)
        rows.append({"model": model, "year": year, "RMSE": MT.rmse(y, p), "QLIKE": MT.qlike(y, p),
                     "MAE": MT.mae(y, p)})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    tests(fc).to_csv(out / "dm_vs_har.csv", index=False)
    mcs(fc).to_csv(out / "mcs.csv", index=False)
    write_manifest(out, exp, args)
    print(summary.pivot_table(index="year", columns="model", values=["RMSE", "QLIKE"]).round(5).to_string())


if __name__ == "__main__":
    main()
