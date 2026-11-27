"""Which model predicts *distributions* best?  Proper-score comparisons and calibration tests.

    uv run python analysis/probabilistic.py --profile full     # -> results/tables/<profile>/prob_*.csv
                                                                #    results/figures/<profile>/prob_dm_*.pdf

Applied to every experiment that keeps full forecast tables (E3 daily and hourly prices, the E2
prospective window, E9). For each source and horizon:

1. **Scores**: CRPS (quantile approximation, 2 x mean pinball over the 9 levels), 80 % interval width
   (sharpness), empirical coverage of every quantile level.
2. **Which distribution is better (significance)**:
   * Diebold–Mariano on CRPS per series (loss = per-origin mean CRPS over the h steps, HAC lag h-1,
     HLN correction), Holm across series -> share of series where model A is significantly better/worse
     than the reference;
   * pairwise DM matrix on the cross-series average CRPS (one test per ordered pair);
   * Model Confidence Set (Hansen, Lunde & Nason, 2011; T_max, 90 %) on per-origin CRPS.
3. **Calibration (is the distribution right?)**, on one-step-ahead forecasts (step 1) so that hits
   are not mechanically autocorrelated by overlapping horizons:
   * Kupiec (1995) unconditional coverage for every quantile level and the central 80 % interval;
   * Christoffersen (1998) independence and conditional coverage for the 80 % interval;
   * discretised PIT uniformity chi2 (10 bins defined by the 9 levels).
   Pooled observed rates plus the share of series rejected at 5 % after Holm correction.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from ciexp import runner as R  # noqa: E402
from ciexp import stats as S  # noqa: E402

LV = R.LEVELS
Q = R.QCOLS


def crps_rows(f: pd.DataFrame) -> pd.Series:
    qs = f[Q].to_numpy(float)
    d = f["y"].to_numpy(float)[:, None] - qs
    return pd.Series((2 * np.maximum(LV * d, (LV - 1) * d)).mean(axis=1), index=f.index)


def sources(res: Path, profile: str) -> dict[str, pd.DataFrame]:
    """Forecast tables restricted to the test period and one context per model (validation-selected)."""
    from report import select_contexts

    out = {}
    e3 = res / "E3_electricity_prices" / profile
    for f in sorted(e3.glob("forecasts_*")):
        block = f.name.split(".")[0].replace("forecasts_", "")
        fc = R.load(f.with_suffix(""))
        fc["context_cfg"] = fc["context_cfg"].astype(str)
        chosen = select_contexts(fc)
        test = fc[fc["phase"] == "test"]
        if chosen:
            want = [chosen.get((m, int(h)), c) for m, h, c in zip(test["model"], test["horizon_cfg"],
                                                                  test["context_cfg"], strict=True)]
            test = test[np.asarray(want) == test["context_cfg"].to_numpy()]
        else:  # no validation phase: full context for TSFMs
            n_ctx = test.groupby(["model", "horizon_cfg"])["context_cfg"].transform("nunique")
            test = test[(n_ctx == 1) | (test["context_cfg"] == "full")]
        out[f"E3_{block}"] = test
    p = R.load(res / "E2_leakage_control" / profile / "prospective_forecasts") \
        if (res / "E2_leakage_control" / profile / "prospective_forecasts.parquet").exists() else None
    if p is not None:
        for w, g in p.groupby("window"):
            out[f"E2_{w}"] = g
    e9 = res / "E9_volatility" / profile / "forecasts"
    if e9.with_suffix(".parquet").exists() or e9.with_suffix(".csv.gz").exists():
        f9 = R.load(e9)
        out["E9_volatility"] = f9.assign(horizon_cfg=1)
    return out


def per_origin_crps(g: pd.DataFrame) -> pd.DataFrame:
    """Wide [series x cutoff] -> model table of per-origin CRPS (mean over the steps)."""
    g = g.assign(crps=crps_rows(g))
    return g.pivot_table(index=["series", "cutoff"], columns="model", values="crps", aggfunc="mean")


def dm_vs_reference(wide: pd.DataFrame, ref: str, h: int) -> pd.DataFrame:
    rows = []
    for comp in [m for m in wide.columns if m != ref]:
        for series, g in wide.groupby(level="series"):
            a, b = g[ref].dropna().align(g[comp].dropna(), join="inner")
            r = S.diebold_mariano(a.to_numpy(), b.to_numpy(), h, loss=lambda x: x)
            rows.append({"comparator": comp, "series": series, "DM": r.statistic, "p_ref_better": r.p_less,
                         "p_ref_worse": 1 - r.p_less if np.isfinite(r.p_less) else np.nan, "n": r.n})
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t["holm_better"] = t.groupby("comparator")["p_ref_better"].transform(S.holm) < 0.05
    t["holm_worse"] = t.groupby("comparator")["p_ref_worse"].transform(S.holm) < 0.05
    return t


def pairwise_dm(wide: pd.DataFrame, h: int) -> pd.DataFrame:
    """DM statistic of row model vs column model on the cross-series mean per-origin CRPS
    (negative = row better); p-values one-sided for 'row better'."""
    avg = wide.groupby(level="cutoff").mean().dropna()
    models = list(avg.columns)
    stat = pd.DataFrame(np.nan, index=models, columns=models)
    pval = stat.copy()
    for a in models:
        for b in models:
            if a != b:
                r = S.diebold_mariano(avg[a].to_numpy(), avg[b].to_numpy(), h, loss=lambda x: x)
                stat.loc[a, b], pval.loc[a, b] = r.statistic, r.p_less
    return stat, pval


def calibration(g: pd.DataFrame) -> pd.DataFrame:
    """One-step-ahead calibration tests per model (pooled rates, share of series rejected after Holm)."""
    g1 = g[g["step"] == 1].sort_values("cutoff")
    rows = []
    for model, gm in g1.groupby("model"):
        rec = {"model": model, "n": int(gm["y"].notna().sum())}
        y = gm["y"].to_numpy(float)
        qs = gm[Q].to_numpy(float)
        for k, lv in enumerate(LV):  # pooled coverage of every quantile level
            rate, _, p = S.kupiec_pof(y[np.isfinite(y)] <= qs[np.isfinite(y), k], lv)
            rec[f"cov_q{lv:.1f}"] = rate
        outside = (y < qs[:, 0]) | (y > qs[:, -1])
        rec["cov80"] = 1 - float(np.mean(outside[np.isfinite(y)]))
        rec["width80"] = float(np.nanmean(qs[:, -1] - qs[:, 0]))
        pk, pc, pp = [], [], []
        for _, gs in gm.groupby("series"):
            ys = gs["y"].to_numpy(float)
            qq = gs[Q].to_numpy(float)
            ok = np.isfinite(ys)
            hits = ((ys < qq[:, 0]) | (ys > qq[:, -1]))[ok]
            pk.append(S.kupiec_pof(hits, 0.2)[2])
            pc.append(S.christoffersen(hits, 0.2)["p_cc"])
            pp.append(S.pit_uniformity(ys, qq, LV)["p"])
        for name, ps in (("kupiec80", pk), ("christoffersen_cc80", pc), ("pit_chi2", pp)):
            ps = np.asarray(ps, float)
            rec[f"reject_{name}_holm"] = float(np.nanmean(S.holm(ps) < 0.05)) if np.isfinite(ps).any() else np.nan
            rec[f"median_p_{name}"] = float(np.nanmedian(ps)) if np.isfinite(ps).any() else np.nan
        pooled = S.pit_uniformity(y, qs, LV)
        rec["pit_chi2_pooled"], rec["pit_p_pooled"] = pooled["chi2"], pooled["p"]
        rows.append(rec)
    return pd.DataFrame(rows)


def heatmap(stat: pd.DataFrame, pval: pd.DataFrame, title: str, path: Path) -> None:
    sys.path.insert(0, str(ROOT / "analysis"))
    from paper import INK, setup_mpl

    plt = setup_mpl()
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    cmap = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#3987e5", "#f0efec", "#e34948", "#c23a39"])
    n = len(stat)
    fig, ax = plt.subplots(figsize=(0.55 * n + 2.4, 0.5 * n + 1.4))
    lim = float(np.nanmax(np.abs(stat.to_numpy()))) or 1
    lim = min(lim, 6)
    im = ax.imshow(stat.clip(-lim, lim).to_numpy(), cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim))
    ax.grid(False)
    ax.set_xticks(range(n), stat.columns, rotation=50, ha="right")
    ax.set_yticks(range(n), stat.index)
    for i in range(n):
        for j in range(n):
            if i != j and np.isfinite(pval.iat[i, j]):
                mark = "**" if pval.iat[i, j] < 0.01 else "*" if pval.iat[i, j] < 0.05 else ""
                ax.text(j, i, f"{stat.iat[i, j]:.1f}{mark}", ha="center", va="center", fontsize=6, color=INK)
    fig.colorbar(im, ax=ax, shrink=0.8, label="DM statistic on CRPS (row − column; blue = row better)")
    ax.set_title(title, fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{ext}"))
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="full")
    p.add_argument("--results", default=str(ROOT / "results"))
    p.add_argument("--reference", default="ci-proto")
    a = p.parse_args()
    res = Path(a.results)
    tabs, figs = res / "tables" / a.profile, res / "figures" / a.profile
    tabs.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, fc in sources(res, a.profile).items():
        for h, g in fc.groupby("horizon_cfg"):
            h = int(h)
            g = g[g["status"] == "ok"] if "status" in g else g
            wide = per_origin_crps(g)
            if wide.empty:
                continue
            tag = f"{name}_h{h}"
            cal = calibration(g)
            cal.insert(0, "source", tag)
            cal = cal.merge(wide.mean().rename("CRPS").reset_index(), on="model")
            cal.to_csv(tabs / f"prob_calibration_{tag}.csv", index=False)
            summary.append(cal)
            if a.reference in wide.columns:
                dm = dm_vs_reference(wide, a.reference, h)
                dm.to_csv(tabs / f"prob_dm_crps_{tag}.csv", index=False)
                share = dm.groupby("comparator")[["holm_better", "holm_worse"]].mean()
                share.to_csv(tabs / f"prob_dm_crps_share_{tag}.csv")
            stat, pval = pairwise_dm(wide, h)
            stat.to_csv(tabs / f"prob_pairwise_dm_stat_{tag}.csv")
            pval.to_csv(tabs / f"prob_pairwise_dm_p_{tag}.csv")
            if len(stat) >= 3:
                heatmap(stat, pval, f"Pairwise DM on CRPS — {tag}", figs / f"prob_dm_{tag}")
            avg = wide.groupby(level="cutoff").mean().dropna()
            if len(avg) >= 30:
                m = S.model_confidence_set(avg.to_numpy(), list(avg.columns), alpha=0.10, reps=1000)
                pd.DataFrame({"model": avg.columns, "in_MCS90": [c in m.included for c in avg.columns],
                              "mcs_p": [m.pvalues.get(c) for c in avg.columns]}).to_csv(
                    tabs / f"prob_mcs_crps_{tag}.csv", index=False)
            print(f"  {tag}: {len(wide.columns)} models, {len(wide)} series-origins")
    if summary:
        pd.concat(summary, ignore_index=True).to_csv(tabs / "prob_calibration_all.csv", index=False)


if __name__ == "__main__":
    main()
