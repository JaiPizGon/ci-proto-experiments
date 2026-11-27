"""Paper figures and cross-experiment tables from saved results (robust to missing experiments).

    uv run python analysis/paper.py --profile full      # -> results/figures/full/*.pdf|png, results/tables/full/

Colour: one fixed categorical slot per foundation model (never re-assigned when a figure shows a
subset), recessive greys for baselines, blue<->red diverging map with a grey midpoint for relative
skill. Every multi-series figure has a legend; lines are 2 px, markers >= 8 px (print sizes).
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ciexp import runner as R  # noqa: E402
from ciexp import stats as S  # noqa: E402

# fixed categorical order (validated reference palette, light mode)
MODEL_COLOURS = {
    "ci-proto": "#2a78d6", "Chronos-2": "#eb6834", "TimesFM-3": "#1baf7a", "TimesFM-2.5": "#eda100",
    "TiRex-2": "#e87ba4", "TiRex": "#008300", "Moirai-2": "#6250d6",
}
SUPERVISED_COLOUR = "#e34948"  # slot 8: the strongest supervised / econometric reference
GREYS = ["#52514e", "#8a8984", "#b5b4ae", "#d3d2cc"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
TSFM_ORDER = list(MODEL_COLOURS)


def base_name(model: str) -> str:
    """Strip mode/context decorations: 'ci-proto[cohort]@full' -> 'ci-proto'."""
    return model.split("@")[0].split("[")[0].strip()


def style(model: str, k_baseline: int = 0) -> dict:
    b = base_name(model)
    if b in MODEL_COLOURS:
        dashed = "[cohort]" in model or "[joint]" in model
        return {"color": MODEL_COLOURS[b], "lw": 2.0, "ls": "--" if dashed else "-", "zorder": 3}
    if b in ("LEAR-recal", "LEAR", "HAR-RV", "N-BEATS", "NBEATS", "PatchTST", "LSTM", "GBM-quantile"):
        return {"color": SUPERVISED_COLOUR, "lw": 1.6, "ls": "-", "zorder": 2}
    return {"color": GREYS[k_baseline % len(GREYS)], "lw": 1.4, "ls": ":", "zorder": 1}


def setup_mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "#fcfcfb",
        "savefig.bbox": "tight", "savefig.dpi": 200, "lines.markersize": 4,
    })
    return plt


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    print(f"  figure {name}")


def read(path: Path) -> pd.DataFrame | None:
    for c in (path, path.with_suffix(".parquet"), path.with_suffix(".csv.gz"), path.with_suffix(".csv")):
        if c.exists() and c.is_file():
            try:
                return pd.read_parquet(c) if c.suffix == ".parquet" else pd.read_csv(c)
            except Exception:  # noqa: BLE001
                return None
    return None


def ordered(models) -> list[str]:
    models = list(dict.fromkeys(models))
    key = {m: (TSFM_ORDER.index(base_name(m)) if base_name(m) in TSFM_ORDER else 99, m) for m in models}
    return sorted(models, key=key.get)


# ----------------------------------------------------------------------------- E1 fev-bench
def fev_tables(res: Path, tabs: Path, figs: Path, plt) -> None:
    ours_dir = res / "E1_general_benchmarks" / PROFILE / "fev"
    official = sorted(glob.glob(str(ROOT / "data" / "fev_results" / "*.csv")))
    if not official:
        return
    import fev

    off = pd.concat([pd.read_csv(f) for f in official], ignore_index=True)
    ours = [pd.read_csv(f) for f in sorted(glob.glob(str(ours_dir / "*.csv")))]
    if not ours:
        lb = fev.analysis.leaderboard(off, metric_column="SQL", baseline_model="Seasonal Naive", n_resamples=1000,
                                      missing_strategy="impute")
        lb.to_csv(tabs / "fev_official_leaderboard_SQL.csv")
        return
    ours = pd.concat(ours, ignore_index=True)
    ours["model_name"] = ours["model_name"] + " (ours)"
    tasks = sorted(set(ours["task_name"]))
    # reproduction fidelity: our per-task SQL vs the official submission of the same model
    name_map = {"ci-proto (ours)": "ci-proto", "Chronos-2 (ours)": "Chronos-2", "TiRex-2 (ours)": "TiRex-2",
                "TiRex (ours)": "TiRex", "TimesFM-2.5 (ours)": "TimesFM-2.5", "TimesFM-3 (ours)": "timesfm-3",
                "Moirai-2 (ours)": "Moirai-2.0", "ci-proto[cohort] (ours)": "ci-proto"}
    rows = []
    for m, g in ours.groupby("model_name"):
        ref = off[off["model_name"] == name_map.get(m, "")].set_index("task_name")["SQL"]
        a, b = g.set_index("task_name")["SQL"].align(ref, join="inner")
        if len(a) == 0:
            continue
        rel = a / b
        rows.append({"model": m, "official": name_map.get(m), "n_tasks": len(a),
                     "median_ratio_ours_over_official": float(np.median(rel)),
                     "geo_mean_ratio": float(np.exp(np.mean(np.log(rel.clip(1e-3))))),
                     "share_within_2pct": float(np.mean(np.abs(rel - 1) < 0.02)),
                     "spearman": float(pd.Series(a.values).corr(pd.Series(b.values), method="spearman"))})
    fid = pd.DataFrame(rows)
    fid.to_csv(tabs / "fev_reproduction_fidelity.csv", index=False)
    # leaderboard on the tasks we ran: official models + our re-runs
    sub = pd.concat([off[off["task_name"].isin(tasks)], ours], ignore_index=True)
    # fev rule: a task a model failed on is imputed with Seasonal Naive (missing_strategy="impute"), so models
    # with CUDA-driver failures (docs/PROTOCOL_CHANGES.md sections 11-13) are ranked rather than dropped
    lb = fev.analysis.leaderboard(sub, metric_column="SQL", missing_strategy="impute",
                                  baseline_model="Seasonal Naive", n_resamples=1000)
    lb.to_csv(tabs / "fev_leaderboard_SQL.csv")
    print(lb.round(3).head(20).to_string())
    # figure: official vs reproduced skill, per model
    shown = [m for m in lb.index if m.endswith("(ours)")]
    fig, ax = plt.subplots(figsize=(5.2, 0.42 * len(shown) + 1.2))
    for k, m in enumerate(shown):
        off_name = name_map.get(m)
        st = style(off_name if off_name != "timesfm-3" else "TimesFM-3")
        ax.errorbar(lb.loc[m, "skill_score"], k, xerr=[[lb.loc[m, "skill_score"] - lb.loc[m, "skill_score_lower"]],
                    [lb.loc[m, "skill_score_upper"] - lb.loc[m, "skill_score"]]], fmt="o", ms=6,
                    color=st["color"], capsize=2, label="reproduced" if k == 0 else None)
        if off_name in lb.index:
            ax.plot(lb.loc[off_name, "skill_score"], k, marker="D", ms=7, mfc="white", mec=INK, mew=1.2,
                    ls="none", zorder=4, label="official submission" if k == 0 else None)
    ax.set_yticks(range(len(shown)), [m.replace(" (ours)", "") for m in shown])
    ax.set_ylim(len(shown) - 0.5, -0.5)
    ax.set_xlabel("fev-bench SQL skill score vs Seasonal Naive (95 % bootstrap CI)")
    ax.legend(frameon=False, loc="lower right")
    save(fig, figs, "fev_reproduction")
    plt.close(fig)


# ------------------------------------------------------------------------ E2 synthetic bank
def e2(res: Path, tabs: Path, figs: Path, plt) -> None:
    met = read(res / "E2_leakage_control" / PROFILE / "metrics")
    if met is None or met.empty:
        return
    # one task = (family, series, horizon); skill vs Seasonal Naive per family (geometric mean of ratios)
    met["task"] = met["family"].astype(str) + "|" + met["series"].astype(str) + "|" + met["horizon_cfg"].astype(str)
    piv = met.pivot_table(index=["family", "task"], columns="model", values="SQL", aggfunc="mean")
    if "SeasonalNaive" not in piv:
        return
    models = ordered([m for m in piv.columns if m != "SeasonalNaive"])
    rel = np.log(piv[models].div(piv["SeasonalNaive"], axis=0).clip(0.01, 100))
    fam = (1 - np.exp(rel.groupby(level="family").mean())).loc[:, models]
    fam.loc["ALL"] = 1 - np.exp(rel.mean())
    fam.to_csv(tabs / "e2_skill_by_family_SQL.csv")
    E = S.fill_failures(piv[models + ["SeasonalNaive"]].to_numpy(), len(models))
    point, lo, hi = S.paired_bootstrap_ci(E, lambda x: S.skill_score(x, len(models)), reps=1000)
    fr = S.friedman_nemenyi(E)
    lb = pd.DataFrame({"skill": point[:-1], "lo95": lo[:-1], "hi95": hi[:-1], "avg_rank": fr["avg_ranks"][:-1]},
                      index=models).sort_values("skill", ascending=False)
    lb.to_csv(tabs / "e2_leaderboard_SQL.csv")
    (tabs / "e2_friedman.json").write_text(json.dumps({k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                                        for k, v in fr.items()}, indent=1))
    # heatmap of skill by family (diverging blue = better than Seasonal Naive, red = worse)
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    cmap = LinearSegmentedColormap.from_list("div", ["#c23a39", "#e34948", "#f0efec", "#3987e5", "#1c5cab"])
    data = fam.T
    fig, ax = plt.subplots(figsize=(0.5 * data.shape[1] + 2.2, 0.32 * data.shape[0] + 1.4))
    lim = 1.0  # colour clamped to [-1, 1] so differences among TSFMs stay visible; labels give exact values
    im = ax.imshow(np.clip(data.to_numpy(), -lim, lim), cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    ax.set_xticks(range(data.shape[1]), [c.replace("syn_", "").replace("_", " ") for c in data.columns],
                  rotation=50, ha="right")
    ax.set_yticks(range(data.shape[0]), data.index)
    ax.grid(False)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(v) > 0.6 * lim else INK)
    fig.colorbar(im, ax=ax, shrink=0.8, label="SQL skill vs Seasonal Naive (colour clamped at ±1)")
    ax.set_title("Leakage-free synthetic families")
    save(fig, figs, "e2_skill_heatmap")
    plt.close(fig)
    cd_diagram(lb["avg_rank"], fr["cd"], figs, "e2_critical_difference", plt)


def cd_diagram(ranks: pd.Series, cd: float, figs: Path, name: str, plt) -> None:
    """Demšar critical-difference diagram: average ranks, Nemenyi CD bar, best half labelled on the
    left (best on top), worst half on the right (worst on top) so connectors never cross labels."""
    ranks = ranks.sort_values()
    k = len(ranks)
    half = (k + 1) // 2
    left = list(ranks.items())[:half]
    right = list(ranks.items())[half:][::-1]
    rows = max(len(left), len(right))
    fig, ax = plt.subplots(figsize=(6.0, 0.32 * rows + 1.1))
    lo_x, hi_x = 1, max(k, int(np.ceil(ranks.max())))
    ax.set_xlim(lo_x - 0.1, hi_x + 0.1)
    ax.set_ylim(-(rows + 0.6) * 0.5, 0.9)
    ax.axhline(0, color=INK2, lw=1)
    ax.grid(False)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    for side, items in (("left", left), ("right", right)):
        for i, (m, r) in enumerate(items):
            y = -(i + 1) * 0.5
            x_end = lo_x - 0.1 if side == "left" else hi_x + 0.1
            colour = MODEL_COLOURS.get(base_name(m), INK2)
            ax.plot([r, r, x_end], [0, y, y], color=colour, lw=1.4, clip_on=False)
            ax.text(x_end + (-0.12 if side == "left" else 0.12), y, f"{m}  {r:.2f}" if side == "left"
                    else f"{r:.2f}  {m}", ha="right" if side == "left" else "left", va="center", fontsize=7.5,
                    color=INK)
    if np.isfinite(cd):
        ax.plot([lo_x, lo_x + cd], [0.55, 0.55], color=INK, lw=2)
        ax.text(lo_x + cd / 2, 0.7, f"CD = {cd:.2f} (Nemenyi, α = 0.05)", ha="center", fontsize=7)
    ax.set_xlabel("average rank (1 = best)")
    save(fig, figs, name)
    plt.close(fig)


# ---------------------------------------------------------------------------- E3 EPF
def e3(res: Path, tabs: Path, figs: Path, plt) -> None:
    an = res / "E3_electricity_prices" / PROFILE / "analysis"
    pooled = an / "daily_pooled.csv"
    if not pooled.exists():
        return
    t = pd.read_csv(pooled, header=[0, 1], index_col=0)
    mae = t["MAE"]
    base = "SeasonalNaive" if "SeasonalNaive" in mae.index else mae.index[0]
    rel = 1 - mae.div(mae.loc[base], axis=1)
    rel = rel.loc[ordered(rel.index)]
    rel.to_csv(tabs / "e3_daily_mae_skill_vs_snaive.csv")
    fig, ax = plt.subplots(figsize=(5.4, 0.25 * len(rel) + 1.2))
    horizons = list(rel.columns)
    marks = ["o", "s", "^"]
    for j, h in enumerate(horizons):
        ax.scatter(rel[h], np.arange(len(rel)) + (j - 1) * 0.18, s=22, marker=marks[j % 3],
                   color=[style(m)["color"] for m in rel.index], edgecolor="white", linewidth=0.5,
                   label=f"h = {h} d", zorder=3)
    ax.set_yticks(range(len(rel)), rel.index)
    ax.invert_yaxis()
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("MAE skill vs Seasonal Naive (test 2024–2025, 23 markets)")
    from matplotlib.lines import Line2D

    keys = [Line2D([], [], marker=marks[j % 3], ls="none", color=INK2, label=f"h = {h} d")
            for j, h in enumerate(horizons)]
    ax.legend(handles=keys, frameon=False, loc="lower left", title="horizon (marker shape)")
    save(fig, figs, "e3_daily_skill")
    plt.close(fig)
    # reliability diagram from the forecast table
    fc = read(res / "E3_electricity_prices" / PROFILE / "forecasts_daily")
    if fc is None:
        return
    fc = fc[fc["phase"] == "test"]
    n_ctx = fc.groupby(["model", "horizon_cfg"])["context_cfg"].transform("nunique")
    fc = fc[(n_ctx == 1) | (fc["context_cfg"].astype(str) == "full")]  # selected context when reduced
    fig, axs = plt.subplots(1, 2, figsize=(7, 3.2), sharey=True)
    for ax, h in zip(axs, sorted(fc["horizon_cfg"].unique())[:2], strict=False):
        g = fc[fc["horizon_cfg"] == h]
        for m in ordered(g["model"].unique()):
            if base_name(m) not in MODEL_COLOURS and m not in ("LEAR-recal", "SeasonalNaive"):
                continue
            gm = g[g["model"] == m]
            obs = [(gm["y"] <= gm[c]).mean() for c in R.QCOLS]
            ax.plot(R.LEVELS, obs, marker="o", ms=3, label=m, **style(m))
        ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls="--")
        ax.set_title(f"h = {h} day(s)")
        ax.set_xlabel("nominal quantile level")
    axs[0].set_ylabel("observed frequency  P(y ≤ q)")
    axs[-1].legend(frameon=False, fontsize=7, loc="upper left")
    save(fig, figs, "e3_reliability")
    plt.close(fig)


# ---------------------------------------------------------------------------- E4 covariates
def e4(res: Path, tabs: Path, figs: Path, plt) -> None:
    t = read(res / "E4_covariates_calendar" / PROFILE / "tests_vs_C0.csv")
    if t is None or t.empty:
        return
    t = t[t["metric"] == "SQL"]
    datasets = sorted(t["dataset"].unique())
    fig, axs = plt.subplots(1, len(datasets), figsize=(3.6 * len(datasets), 3.0), sharey=True, squeeze=False)
    for ax, d in zip(axs[0], datasets, strict=True):
        g = t[(t["dataset"] == d) & (t["horizon"] == t["horizon"].min())]
        conds = sorted(g["condition"].unique())
        for m in ordered(g["model"].unique()):
            gm = g[g["model"] == m].set_index("condition").reindex(conds)
            ax.plot(range(len(conds)), 100 * (gm["mean_cond"] / gm["mean_C0"] - 1), marker="o", label=m, **style(m))
            sig = gm["p_holm"] < 0.05
            ax.scatter(np.arange(len(conds))[sig.to_numpy()], (100 * (gm["mean_cond"] / gm["mean_C0"] - 1))[sig],
                       s=40, facecolor="none", edgecolor=INK, lw=0.8, zorder=4)
        ax.set_xticks(range(len(conds)), [c.replace("_l", " λ=") for c in conds], rotation=45, ha="right")
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_title(d.replace("_", " "))
    axs[0][0].set_ylabel("SQL change vs C0 (%)  ○ = Holm p < 0.05")
    axs[0][-1].legend(frameon=False, fontsize=7)
    save(fig, figs, "e4_covariates")
    plt.close(fig)


# ---------------------------------------------------------------------------- E5 context
def e5(res: Path, tabs: Path, figs: Path, plt) -> None:
    t = read(res / "E5_context_scaling" / PROFILE / "context_scaling.csv")
    if t is not None and not t.empty:
        for h, g in t.groupby("horizon"):
            fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0))
            for m in ordered(g["model"].unique()):
                gm = g[g["model"] == m].sort_values("context")
                st = style(m)
                inside = gm[~gm["exceeds_native_context"].astype(bool)]
                axs[0].plot(inside["context"], inside["MASE"], marker="o", label=m, **st)
                over = gm[gm["context"] >= (inside["context"].max() if len(inside) else 0)]
                axs[0].plot(over["context"], over["MASE"], color=st["color"], lw=1, ls="--", alpha=0.6)
                axs[1].plot(gm["context"], gm["latency_s"], marker="o", **st)
            for ax in axs:
                ax.set_xscale("log", base=2)
                ax.set_xlabel("context length (observations, 1-min data)")
            axs[1].set_yscale("log")
            axs[0].set_ylabel("MASE")
            axs[1].set_ylabel("median latency per forecast (s)")
            axs[0].axvline(16384, color=INK2, lw=0.6, ls=":")
            axs[0].text(16384, axs[0].get_ylim()[1], " long ckpt", fontsize=6, va="top", color=INK2)
            handles, labels = axs[0].get_legend_handles_labels()
            fig.legend(handles, labels, frameon=False, fontsize=7, ncol=4, loc="lower center",
                       bbox_to_anchor=(0.5, -0.12))
            fig.suptitle(f"Context scaling, UCI household power, horizon {h} min (dashed: beyond native window)",
                         fontsize=9)
            save(fig, figs, f"e5_context_h{h}")
            plt.close(fig)
    b = read(res / "E5_context_scaling" / PROFILE / "regime_breaks.csv")
    if b is not None and not b.empty:
        g = b.groupby(["model", "context_cfg", "days_after_break"])["MASE"].mean().reset_index()
        g.to_csv(tabs / "e5_regime_breaks.csv", index=False)
        models = [m for m in ordered(g["model"].unique()) if base_name(m) in MODEL_COLOURS]
        fig, axs = plt.subplots(1, len(models), figsize=(2.1 * len(models), 2.6), sharey=True, squeeze=False)
        ramp = ["#b7d3f6", "#86b6ef", "#5598e7", "#256abf", "#104281"]  # ordinal blue ramp (context length)
        for ax, m in zip(axs[0], models, strict=True):
            for k, (ctx, gc) in enumerate(g[g["model"] == m].groupby("context_cfg")):
                ax.plot(gc["days_after_break"], gc["MASE"], marker="o", ms=3, lw=1.6, color=ramp[k % len(ramp)],
                        label=f"{ctx}")
            ax.set_title(m, fontsize=8)
            ax.set_xlabel("days after break")
        axs[0][0].set_ylabel("MASE (break series)")
        axs[0][-1].legend(frameon=False, fontsize=6, title="context")
        save(fig, figs, "e5_regime_breaks")
        plt.close(fig)


# ---------------------------------------------------------------------------- E6 scarcity
def e6(res: Path, tabs: Path, figs: Path, plt) -> None:
    t = read(res / "E6_data_scarcity" / PROFILE / "summary.csv")
    if t is None or t.empty:
        return
    lead = 24 if 24 in set(t["lead"]) else int(t["lead"].max())
    g = t[t["lead"] == lead]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    sup = g[g["ratio"].astype(str).str.match(r"^[0-9.]+$")]
    for k, (m, gm) in enumerate(sup.groupby("model")):
        gm = gm.assign(r=gm["ratio"].astype(float)).sort_values("r")
        ax.plot(100 * gm["r"], gm["RMSE"], marker="s", label=f"{m} (trained)", **style(m, k))
    zs = g[~g.index.isin(sup.index)]
    for m in ordered(zs["model"].unique()):
        row = zs[(zs["model"] == m) & (zs["ratio"] == "full")]
        if len(row):
            ax.axhline(float(row["RMSE"].iloc[0]), lw=1.4, color=style(m)["color"], ls="--",
                       label=f"{m} zero-shot (full context)")
    ax.set_xlabel("training data used by supervised models (%)")
    ax.set_ylabel(f"RMSE, PM2.5, lead {lead} h")
    ax.legend(frameon=False, fontsize=6, ncol=2)
    save(fig, figs, "e6_scarcity")
    plt.close(fig)


# ---------------------------------------------------------------------------- E8 efficiency
def _flag(t: pd.DataFrame, col: str, fallback) -> pd.Series:
    """Boolean column of E8 (older tables without it: derive from the variant label)."""
    if col in t:
        return t[col].astype(str).str.lower().isin(["true", "1"])
    return t["variant"].astype(str).map(fallback)


def e8(res: Path, tabs: Path, figs: Path, plt) -> None:
    """Latency vs context, one figure per batch and per setting: out-of-the-box (each library's default
    GPU stack) and controlled (eager PyTorch, fp32, no compilation / custom kernels)."""
    t = read(res / "E8_efficiency" / PROFILE / "efficiency.csv")
    if t is None or t.empty or "median_latency_s" not in t:
        return
    t = t[(t["status"] == "ok") & (t["horizon"] == t["horizon"].min())]
    t = t[~t["variant"].astype(str).str.startswith("cpu")]
    settings = {"default": _flag(t, "is_default", lambda v: v in ("cuda", "default", "cuda/mamba3")),
                "controlled": _flag(t, "is_controlled", lambda v: "torch" in v)}
    for setting, mask in settings.items():
        for batch, g in t[mask].groupby("batch"):
            fig, ax = plt.subplots(figsize=(4.8, 3.1))
            for m, gm in g.groupby("model"):
                gm = gm.sort_values("context")
                ax.plot(gm["context"], gm["median_latency_s"], marker="o", label=m, **style(m))
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xlabel("context length")
            ax.set_ylabel("median latency per call (s), GPU")
            ax.set_title(f"{'out-of-the-box' if setting == 'default' else 'controlled: eager fp32'}, "
                         f"batch = {batch}", fontsize=8)
            ax.legend(frameon=False, fontsize=6)
            save(fig, figs, f"e8_latency_{setting}_b{batch}")
            plt.close(fig)
    every = read(res / "E8_efficiency" / PROFILE / "efficiency.csv")  # incl. CPU: TiRex-2 is counted there
    if "gflops_per_series" in every and every["gflops_per_series"].notna().any():
        f = every[every["gflops_per_series"].notna()]
        f.pivot_table(index="context", columns="model", values="gflops_per_series").to_csv(tabs / "e8_gflops.csv")


def gift(res: Path, tabs: Path, figs: Path, plt) -> None:
    lb = read(res / "E10_gift_eval" / PROFILE / "leaderboard.csv")
    if lb is None or lb.empty:
        return
    lb = lb.rename(columns={lb.columns[0]: "model"}).sort_values("CRPS_gmean")
    lb.to_csv(tabs / "gift_eval_leaderboard.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.4, 0.26 * len(lb) + 1.2))
    for k, r in enumerate(lb.itertuples()):
        name = r.model.replace(" (ours)", "")
        ours = r.model.endswith("(ours)")
        colour = MODEL_COLOURS.get(base_name(name), GREYS[1])
        ax.plot(r.CRPS_gmean, k, "o", ms=6, color=colour, mfc=colour if ours else "white", mew=1.4)
        ax.plot(r.MASE_gmean, k, "s", ms=4, color=colour, alpha=0.6)
    ax.set_yticks(range(len(lb)), lb["model"])
    ax.set_ylim(len(lb) - 0.5, -0.5)
    ax.axvline(1, color=INK2, lw=0.8)
    ax.set_xlabel("geometric mean relative to Seasonal Naive (● CRPS, ■ MASE; filled = our run)")
    ax.set_title(f"GIFT-Eval, {int(lb['n_configs'].max())} configurations")
    save(fig, figs, "gift_eval_leaderboard")
    plt.close(fig)


def cost_table(res: Path, tabs: Path, figs: Path, plt) -> None:
    """Inference cost per model: parameters, latency / throughput / peak memory (E8, GPU, batch 1 and 32
    at a 2,048 and the longest feasible context) and total benchmark inference time (fev-bench, GIFT-Eval)."""
    parts = []
    t = read(res / "E8_efficiency" / PROFILE / "efficiency.csv")
    if t is not None and not t.empty and "median_latency_s" in t:
        t = t[(t["status"] == "ok") & (t["horizon"] == t["horizon"].min())]
        gpu = t[~t["variant"].astype(str).str.startswith("cpu")]
        is_def = _flag(gpu, "is_default", lambda v: v in ("cuda", "default", "cuda/mamba3"))
        is_ctl = _flag(gpu, "is_controlled", lambda v: False)
        flops = t.dropna(subset=["gflops_per_series"]) if "gflops_per_series" in t else t.iloc[:0]
        for (m, v), g in gpu.groupby(["model", "variant"]):
            d, c = bool(is_def[g.index].any()), bool(is_ctl[g.index].any())
            name = m if d else f"{m} [{v}]"
            rec = {"model": name, "setting": "default" + ("+controlled" if c else "") if d else
                   ("controlled" if c else "extra"),
                   "params_M": g["params_M"].iloc[0] if "params_M" in g else np.nan,
                   "max_context_ok": int(g["context"].max())}
            for col in ("precision", "backend", "compiled", "process"):
                if col in g:
                    rec[col] = g[col].iloc[0]
            for b in (1, 32):
                gb = g[(g["batch"] == b) & (g["context"] == 2048)]
                if len(gb):
                    rec[f"latency_s_b{b}_ctx2048"] = float(gb["median_latency_s"].iloc[0])
                    rec[f"series_per_s_b{b}_ctx2048"] = float(gb["series_per_s"].iloc[0])
                    rec[f"gpu_peak_mb_b{b}_ctx2048"] = float(gb["peak_cuda_mb"].iloc[0])
                    if "first_call_s" in gb:
                        rec[f"first_call_s_b{b}_ctx2048"] = float(gb["first_call_s"].iloc[0])
                    if "ipc_overhead_s" in gb:
                        rec[f"ipc_overhead_s_b{b}_ctx2048"] = float(gb["ipc_overhead_s"].iloc[0])
            fm = flops[(flops["model"] == m) & (flops["context"] == 2048)]
            if len(fm):
                rec["gflops_per_series_ctx2048"] = float(fm["gflops_per_series"].iloc[0])
            gl = g[g["context"] == g["context"].max()].sort_values("batch").iloc[0]
            rec["latency_s_longest_b1"] = float(gl["median_latency_s"])
            rec["gpu_peak_mb_longest"] = float(gl["peak_cuda_mb"])
            if "peak_host_rss_mb" in gl:
                rec["host_rss_peak_mb_longest"] = float(gl["peak_host_rss_mb"])
            parts.append(rec)
    cost = pd.DataFrame(parts)
    fev_dir = res / "E1_general_benchmarks" / PROFILE / "fev"
    fev_t = {}
    for f in glob.glob(str(fev_dir / "*.csv")):
        d = pd.read_csv(f)
        fev_t[d["model_name"].iloc[0]] = (float(d["inference_time_s"].sum()) / 3600, int(len(d)))
    gift = read(res / "E10_gift_eval" / PROFILE / "our_results.csv")
    gift_t = {} if gift is None or "inference_time_s" not in gift else \
        {m: (float(g["inference_time_s"].sum()) / 3600, int(len(g))) for m, g in gift.groupby("model")}
    names = sorted(set(cost.get("model", pd.Series(dtype=str))) | set(fev_t) | set(gift_t))
    cost = cost.set_index("model").reindex(names) if len(cost) else pd.DataFrame(index=names)
    cost["fev_inference_h"] = [fev_t.get(n, (np.nan, 0))[0] for n in names]
    cost["fev_tasks"] = [fev_t.get(n, (np.nan, 0))[1] for n in names]
    cost["gift_inference_h"] = [gift_t.get(n, (np.nan, 0))[0] for n in names]
    cost["gift_configs"] = [gift_t.get(n, (np.nan, 0))[1] for n in names]
    cost.to_csv(tabs / "inference_cost.csv")
    print(cost.round(3).to_string())


def main() -> None:
    global PROFILE
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="full")
    p.add_argument("--results", default=str(ROOT / "results"))
    a = p.parse_args()
    PROFILE = a.profile
    res = Path(a.results)
    figs, tabs = res / "figures" / PROFILE, res / "tables" / PROFILE
    tabs.mkdir(parents=True, exist_ok=True)
    plt = setup_mpl()
    for fn in (fev_tables, e2, e3, e4, e5, e6, e8, gift, cost_table):
        try:
            fn(res, tabs, figs, plt)
        except Exception as exc:  # noqa: BLE001  one broken section must not hide the others
            import traceback

            print(f"[warn] {fn.__name__}: {type(exc).__name__}: {exc}")
            traceback.print_exc()


PROFILE = "full"

if __name__ == "__main__":
    main()
