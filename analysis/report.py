"""Statistical analysis and paper tables from saved experiment results.

Examples
--------
python analysis/report.py epf   --results results/E3_electricity_prices/full --reference "ci-proto[joint]"
python analysis/report.py bench --results results/E2_leakage_control/full   --metric SQL --baseline SeasonalNaive
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ciexp import runner as R  # noqa: E402
from ciexp import stats as S  # noqa: E402


# --------------------------------------------------------------------------- helpers
def to_latex(df: pd.DataFrame, path: Path, caption: str, label: str, floatfmt: str = "%.3f") -> None:
    try:
        tex = df.to_latex(float_format=lambda x: floatfmt % x, caption=caption, label=label, escape=True)
    except ImportError:  # jinja2 missing
        tex = df.to_string()
    path.write_text(tex, encoding="utf-8")


def select_contexts(forecasts: pd.DataFrame, metric_by: str = "MAE") -> dict[tuple[str, int], str]:
    """Choose each model's context per horizon on the validation phase (no test-set snooping)."""
    val = forecasts[forecasts["phase"] == "validation"]
    if val.empty:
        return {}
    val = val.assign(abs_err=(val["y"] - val["q0.5"]).abs())
    score = val.groupby(["model", "horizon_cfg", "context_cfg"])["abs_err"].mean().reset_index()
    best = score.sort_values("abs_err").groupby(["model", "horizon_cfg"]).head(1)
    return {(r.model, int(r.horizon_cfg)): str(r.context_cfg) for r in best.itertuples()}


def per_origin_loss(g: pd.DataFrame, loss: str) -> pd.Series:
    e = g["err"] if "err" in g else g["y"] - g["q0.5"]
    v = e.abs() if loss == "abs" else e**2
    return v.groupby(g["cutoff"]).mean()


# ------------------------------------------------------------------------------- EPF
def epf_report(results: Path, reference: str, out: Path) -> None:
    files = sorted(results.glob("forecasts_*"))
    if not files:
        raise FileNotFoundError(f"no forecast tables in {results}")
    for f in files:
        block = f.name.split(".")[0].replace("forecasts_", "")
        ref_model = reference
        fc = R.load(f.with_suffix(""))
        fc["context_cfg"] = fc["context_cfg"].astype(str)
        chosen = select_contexts(fc)
        test = fc[fc["phase"].isin(["test"])]
        if chosen:
            wanted = [chosen.get((m, int(h)), c) for m, h, c in
                      zip(test["model"], test["horizon_cfg"], test["context_cfg"], strict=True)]
            test = test[np.asarray(wanted) == test["context_cfg"].to_numpy()]
        pd.Series({f"{k[0]}|h{k[1]}": v for k, v in chosen.items()}).to_csv(out / f"{block}_selected_contexts.csv")
        test = test.assign(err=test["y"] - test["q0.5"])
        if not chosen:  # no validation phase (laptop profile): treat each context as its own model
            # E3 daily tables are stored already reduced to one (validation-selected) context per model
            # and horizon; only tables that still hold several contexts per model/horizon are split
            multi = test.groupby(["model", "horizon_cfg"])["context_cfg"].transform("nunique") > 1
            test.loc[multi, "model"] = test.loc[multi, "model"] + "@" + test.loc[multi, "context_cfg"]
            if ref_model not in set(test["model"]):
                candidates = sorted(m for m in set(test["model"]) if m.startswith(ref_model + "@"))
                if candidates:
                    ref_model = next((m for m in candidates if m.endswith("@full")), candidates[0])

        # pooled accuracy table
        rows = []
        for (model, h), g in test.groupby(["model", "horizon_cfg"]):
            q = g[R.QCOLS].to_numpy(float)
            diff = g["y"].to_numpy(float)[:, None] - q
            pin = 2 * np.maximum(R.LEVELS * diff, (R.LEVELS - 1) * diff)
            rows.append({"model": model, "horizon": h, "MAE": g["err"].abs().mean(),
                         "RMSE": np.sqrt((g["err"] ** 2).mean()), "CRPS": pin.mean(),
                         "COV80": np.mean((g["y"] >= g["q0.1"]) & (g["y"] <= g["q0.9"]))})
        pooled = pd.DataFrame(rows)
        table = pooled.pivot_table(index="model", columns="horizon", values=["MAE", "RMSE", "CRPS", "COV80"])
        table.to_csv(out / f"{block}_pooled.csv")
        to_latex(table, out / f"{block}_pooled.tex", f"Pooled test accuracy ({block})", f"tab:{block}_pooled")

        if ref_model not in set(test["model"]):
            print(f"[{block}] ref_model {ref_model!r} not found; skipping tests")
            continue
        # relative MAE improvement of the ref_model
        ref = pooled[pooled["model"] == ref_model].set_index("horizon")["MAE"]
        rel = pooled.assign(improvement_pct=100 * (pooled["MAE"] - pooled["horizon"].map(ref)) / pooled["MAE"])
        rel.pivot_table(index="model", columns="horizon", values="improvement_pct").to_csv(
            out / f"{block}_ref_model_improvement_pct.csv")

        # Diebold–Mariano per series, Holm across series
        dm_rows = []
        for h, gh in test.groupby("horizon_cfg"):
            for comp in sorted(set(gh["model"]) - {ref_model}):
                for loss in ("abs", "sq"):
                    for series, gs in gh.groupby("series"):
                        # one loss per origin (mean over the h steps of that forecast); consecutive daily
                        # origins overlap by h-1 steps, hence the HAC lag h-1 inside diebold_mariano
                        a = per_origin_loss(gs[gs["model"] == ref_model], loss)
                        b = per_origin_loss(gs[gs["model"] == comp], loss)
                        a, b = a.align(b, join="inner")
                        res = S.diebold_mariano(a.to_numpy(), b.to_numpy(), int(h), loss=lambda x: x)
                        dm_rows.append({"horizon": h, "comparator": comp, "loss": loss, "series": series,
                                        "DM": res.statistic, "p_two": res.p_two_sided, "p_ref_better": res.p_less,
                                        "n": res.n})
        dm = pd.DataFrame(dm_rows)
        if not dm.empty:
            dm["p_holm"] = dm.groupby(["horizon", "comparator", "loss"])["p_ref_better"].transform(S.holm)
            dm["p_bh"] = dm.groupby(["horizon", "comparator", "loss"])["p_ref_better"].transform(S.benjamini_hochberg)
            dm.to_csv(out / f"{block}_dm_tests.csv", index=False)
            share = (dm.assign(sig=dm["p_holm"] < 0.05).groupby(["horizon", "comparator", "loss"])["sig"].mean()
                     .unstack("loss"))
            share.to_csv(out / f"{block}_dm_share_significant_holm.csv")

        # Model Confidence Set per horizon (absolute and pinball losses)
        mcs_rows = []
        for h, gh in test.groupby("horizon_cfg"):
            models = sorted(gh["model"].unique())
            for loss in ("abs", "pinball"):
                # MCS on the time series of per-origin losses (averaged over markets and steps), so the
                # moving-block bootstrap resamples contiguous blocks of origins
                wide = R.aligned_losses(gh, models, loss=loss).groupby(level="cutoff").mean()
                if len(wide) < 30:
                    continue
                res = S.model_confidence_set(wide.to_numpy(), list(wide.columns), alpha=0.10, reps=1000)
                for m in models:
                    mcs_rows.append({"horizon": h, "loss": loss, "model": m, "in_MCS90": m in res.included,
                                     "mcs_p": res.pvalues.get(m, np.nan)})
        pd.DataFrame(mcs_rows).to_csv(out / f"{block}_mcs.csv", index=False)
        print(f"[{block}] wrote pooled, DM/Holm and MCS tables to {out}")


# ------------------------------------------------------------- benchmark-style summary
def bench_report(results: Path, metric: str, baseline: str, out: Path, task_cols=("series", "tag")) -> None:
    met = R.load(results / "metrics")
    met["task"] = met[list(task_cols)].astype(str).agg("|".join, axis=1)
    errors = met.pivot_table(index="task", columns="model", values=metric, aggfunc="mean")
    if baseline not in errors.columns:
        raise KeyError(f"baseline {baseline!r} not among {list(errors.columns)}")
    names = list(errors.columns)
    b = names.index(baseline)
    E = S.fill_failures(errors.to_numpy(), b)
    skill, lo, hi = S.paired_bootstrap_ci(E, lambda x: S.skill_score(x, b), reps=1000)
    wr, wlo, whi = S.paired_bootstrap_ci(E, S.win_rate, reps=1000)
    fr = S.friedman_nemenyi(E)
    table = pd.DataFrame({"skill_score": skill, "skill_lo95": lo, "skill_hi95": hi, "win_rate": wr,
                          "win_lo95": wlo, "win_hi95": whi, "avg_rank": fr["avg_ranks"]}, index=names)
    table = table.sort_values("skill_score", ascending=False)
    table.to_csv(out / f"leaderboard_{metric}.csv")
    to_latex(table, out / f"leaderboard_{metric}.tex", f"Skill score and win rate ({metric})", f"tab:lb_{metric}")
    (out / f"friedman_{metric}.txt").write_text(
        f"Friedman chi2={fr['friedman_stat']:.3f}, p={fr['friedman_p']:.3g}, Nemenyi CD(0.05)={fr['cd']:.3f}, "
        f"n_tasks={fr['n']}\n")
    print(table.round(3).to_string())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("kind", choices=["epf", "bench"])
    p.add_argument("--results", required=True)
    p.add_argument("--reference", default="ci-proto[joint]")
    p.add_argument("--metric", default="SQL")
    p.add_argument("--baseline", default="SeasonalNaive")
    p.add_argument("--out", default=None)
    p.add_argument("--task-cols", nargs="+", default=["series", "tag"],
                   help="columns that identify a task for the benchmark aggregation")
    a = p.parse_args()
    results = Path(a.results)
    out = Path(a.out) if a.out else results / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    if a.kind == "epf":
        epf_report(results, a.reference, out)
    else:
        bench_report(results, a.metric, a.baseline, out, task_cols=tuple(a.task_cols))


if __name__ == "__main__":
    main()
