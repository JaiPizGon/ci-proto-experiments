"""Collect the key tables of every experiment into one Markdown file.

    uv run python analysis/summarize.py --profile laptop     # -> results/SUMMARY_laptop.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def md(df: pd.DataFrame, digits: int = 3) -> str:
    """Plain Markdown table (no tabulate dependency)."""
    if df is None or df.empty:
        return "_no rows_\n"
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join(str(c) for c in col if str(c)) for col in df.columns]
    df = df.reset_index()
    cols = [str(c) for c in df.columns]

    def fmt(v):
        if isinstance(v, float):
            return "" if pd.isna(v) else f"{v:.{digits}f}"
        return str(v)

    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def read(path: Path) -> pd.DataFrame | None:
    for candidate in (path, path.with_suffix(".parquet"), path.with_suffix(".csv.gz")):
        if candidate.exists():
            try:
                return pd.read_parquet(candidate) if candidate.suffix == ".parquet" else pd.read_csv(candidate)
            except (pd.errors.EmptyDataError, ValueError):
                return None
    return None


def env_line(folder: Path) -> str:
    manifest = folder / "manifest.json"
    if not manifest.exists():
        return ""
    env = json.loads(manifest.read_text(encoding="utf-8")).get("environment", {})
    keys = ("platform", "python", "torch", "gpu", "continuousintelligence")
    return "Environment: " + ", ".join(f"{k}={env.get(k)}" for k in keys) + "\n\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="laptop")
    p.add_argument("--results", default=str(ROOT / "results"))
    a = p.parse_args()
    res = Path(a.results)
    parts = [f"# ci-proto experiments — results ({a.profile} profile)\n"]

    def folder(name: str) -> Path:
        return res / name / a.profile

    f = folder("E1_general_benchmarks")
    t = read(f / "ltsf_zero_shot.csv")
    parts.append("## E1 — LTSF zero-shot (MSE on train-z-scored data)\n\n" + env_line(f))
    if t is not None:
        parts.append(md(t.pivot_table(index=["dataset", "horizon", "lookback", "mode"], columns="model", values="MSE"), 4))
        parts.append("\nMAE\n\n" + md(t.pivot_table(index=["dataset", "horizon", "lookback", "mode"], columns="model",
                                                      values="MAE"), 4))

    f = folder("E2_leakage_control")
    t = read(f / "summary_by_process.csv")
    parts.append("\n## E2 — Leakage-free synthetic suite (MASE, mean over horizons)\n\n" + env_line(f))
    if t is not None:
        parts.append(md(t.pivot_table(index="series", columns="model", values="MASE")))
        parts.append("\nSQL\n\n" + md(t.pivot_table(index="series", columns="model", values="SQL")))
    lb = read(f / "analysis" / "leaderboard_SQL.csv")
    if lb is not None:
        lb = lb.rename(columns={lb.columns[0]: "model"}).set_index("model")
        parts.append("\nLeaderboard (skill vs Seasonal Naive, 95 % bootstrap CI)\n\n" + md(lb))

    f = folder("E3_electricity_prices")
    parts.append("\n## E3 — Electricity prices, Ember 23 markets, test 2024–2025\n\n" + env_line(f))
    t = read(f / "analysis" / "daily_pooled.csv")
    if t is not None:
        parts.append("Pooled test accuracy (MAE, RMSE in EUR/MWh; CRPS; 80 % coverage)\n\n")
        parts.append(md(pd.read_csv(f / "analysis" / "daily_pooled.csv", header=[0, 1], index_col=0), 2))
    t = read(f / "analysis" / "daily_dm_share_significant_holm.csv")
    if t is not None:
        parts.append("\nShare of markets where the reference beats the comparator (DM-HLN, Holm 5 %)\n\n" + md(t.set_index(
            list(t.columns[:2]))))
    t = read(f / "analysis" / "daily_mcs.csv")
    if t is not None:
        parts.append("\nModel Confidence Set (90 %)\n\n" + md(t.pivot_table(index="model", columns=["horizon", "loss"],
                                                                          values="in_MCS90", aggfunc="first")))

    f = folder("E4_covariates_calendar")
    t = read(f / "summary.csv")
    parts.append("\n## E4 — Covariates and calendar (MASE)\n\n" + env_line(f))
    if t is not None:
        parts.append(md(t.pivot_table(index=["horizon_cfg", "condition"], columns="model", values="MASE")))
        parts.append("\nSQL\n\n" + md(t.pivot_table(index=["horizon_cfg", "condition"], columns="model", values="SQL")))

    f = folder("E5_context_scaling")
    t = read(f / "context_scaling.csv")
    parts.append("\n## E5 — Context scaling, UCI household power (1-min)\n\n" + env_line(f))
    if t is not None:
        parts.append("MASE\n\n" + md(t.pivot_table(index=["horizon", "context"], columns="model", values="MASE")))
        parts.append("\nMedian latency per call (s)\n\n" + md(t.pivot_table(index=["horizon", "context"], columns="model",
                                                                             values="latency_s"), 4))
    t = read(f / "regime_breaks.csv")
    if t is not None:
        parts.append("\nAfter a structural break (MASE by context)\n\n" + md(
            t.pivot_table(index=["series"], columns=["model", "context_cfg"], values="MASE")))

    f = folder("E6_data_scarcity")
    t = read(f / "summary.csv")
    parts.append("\n## E6 — Data scarcity, Beijing PM2.5 (RMSE by lead time)\n\n" + env_line(f))
    if t is not None:
        parts.append(md(t.pivot_table(index="lead", columns=["model", "ratio"], values="RMSE"), 2))

    f = folder("E7_ablations_robustness")
    t = read(f / "summary.csv")
    parts.append("\n## E7 — Ablations and missing data (SQL)\n\n" + env_line(f))
    if t is not None:
        parts.append(md(t.pivot_table(index=["study", "dataset", "variant"], columns="model", values="SQL")))

    f = folder("E8_efficiency")
    t = read(f / "efficiency.csv")
    parts.append("\n## E8 — Efficiency (median latency, s)\n\n" + env_line(f))
    if t is not None and "median_latency_s" in t:
        parts.append(md(t.pivot_table(index=["context", "batch"], columns=["model", "variant"],
                                      values="median_latency_s"), 4))

    out = res / f"SUMMARY_{a.profile}.md"
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
