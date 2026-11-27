"""E7 — Ablations of ci-proto's inference components and robustness to missing data.

ci-proto is inference-only, so the ablations act on the documented ``tta.json`` switches
(differencing, running totals, sign-flip averaging, band calibration, categorical target
encoding, non-negativity clipping, integer rounding, calendar features, bfloat16), on the
three multivariate modes (joint / cohort / independent) and on missing data injected at
random or in blocks. This mirrors the component ablations reported for Mamba forecasters
in Neurocomputing and Knowledge-Based Systems, adapted to a pretrained model.
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import pandas as pd
from common import checkpointed, knob, load_config, make_model, model_keys, out_dir, parse, write_manifest

from ciexp import data as D
from ciexp import protocols as P
from ciexp import runner as R

BOOLEAN_FLAGS = ["postprocess", "tta_diff", "tta_running_totals", "tta_sign_flip", "target_encode_categoricals",
                 "clip_nonnegative", "integer_round", "calendar_covariates", "bfloat16"]


def default_tta(source: str, revision: str | None) -> dict:
    from huggingface_hub import hf_hub_download

    return json.loads(Path(hf_hub_download(source, "tta.json", revision=revision)).read_text(encoding="utf-8"))


CUDA_FATAL_EXIT = 75
FATAL_CUDA = r"handles_\.at|CUDA driver error|illegal memory access|unspecified launch failure"


def evaluation_sets(args) -> list[D.SeriesSet]:
    """Five domains: synthetic (leak-free), electricity prices, transformer load, air quality, mobility."""
    sets = [D.synthetic_suite(length=4096 if args.quick else 8192, seed=11)]
    if args.quick:
        return sets
    root = Path(args.data_dir)
    wanted = knob(args, "datasets", ["synthetic_suite", "ember_daily", "ETTh1", "beijing_air_quality", "bike"])
    if "ember_daily" in wanted and (root / "ember_daily.csv").exists():
        sets.append(D.ember_daily_prices(root / "ember_daily.csv"))
    if "ETTh1" in wanted and (root / "ltsf" / "ETTh1.csv").exists():
        sets.append(D.generic_csv(root / "ltsf" / "ETTh1.csv", "date", ["HUFL", "HULL", "MUFL", "MULL", "LUFL",
                                                                        "LULL", "OT"], "h", name="ETTh1"))
    if "beijing_air_quality" in wanted and (root / "beijing_air_quality").exists():
        bj = D.beijing_air_quality(root / "beijing_air_quality")
        bj.past_covariates = None
        sets.append(bj)
    if "bike" in wanted and (root / "bike_sharing_openml42712.csv").exists():
        bk = D.bike_sharing_uci(pd.read_csv(root / "bike_sharing_openml42712.csv"))
        bk.past_covariates = bk.future_covariates = None
        sets.append(bk)
    return sets


def horizon_for(ds: D.SeriesSet) -> int:
    return 7 if ds.freq == "D" else 24


def run_set(model, ds, horizon, n_windows, tag, **kw) -> pd.DataFrame:
    horizon = horizon_for(ds)
    origins = P.rolling_origins(ds.values.shape[1], horizon, n_windows=n_windows, step=horizon)
    fc = R.run_backtest(model, ds, origins, joint=kw.pop("joint", False), tag=tag, **kw)
    return R.metrics_table(fc, ds)


def paired_tests(table: pd.DataFrame) -> pd.DataFrame:
    """Wilcoxon signed-rank of every variant against its reference, per dataset; Holm within study."""
    from ciexp import stats as S

    refs = {"tta": "default", "mode": "ci-proto-joint", "missing": None}
    rows = []
    for (study, dataset, model), g in table.groupby(["study", "dataset", "model"]):
        ref = refs[study]
        if study == "missing":
            ref = next((v for v in g["variant"].unique() if v.endswith("@0.0")), None)
        if study == "mode":
            g = g.assign(model="ci-proto")
        base = g[g["variant"] == ref].set_index(["series", "cutoff"])
        if base.empty:
            continue
        for variant, gv in g.groupby("variant"):
            if variant == ref:
                continue
            for metric in ("SQL", "MASE"):
                a, b = gv.set_index(["series", "cutoff"])[metric].align(base[metric], join="inner")
                rows.append({"study": study, "dataset": dataset, "model": model, "variant": variant,
                             "reference": ref, "metric": metric, "mean_ref": float(b.mean()),
                             "mean_variant": float(a.mean()),
                             "rel_change_pct": float(100 * (a.mean() / b.mean() - 1)), "n": int(len(a)),
                             "p_wilcoxon": S.wilcoxon_pair(a.to_numpy(), b.to_numpy())})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm"] = out.groupby(["study", "model", "metric"])["p_wilcoxon"].transform(S.holm)
    return out


def main() -> None:
    args = parse(__doc__)
    cfg = load_config(args.config)
    exp = cfg["experiments"]["E7_ablations_robustness"]
    out = out_dir(cfg, args, "E7_ablations_robustness")
    sets = evaluation_sets(args)
    horizon, n_windows = 24, (4 if args.quick else int(knob(args, "n_windows", 20)))
    tables = []

    has_ci = "ci-proto" in model_keys({"models": ["ci-proto"]}, args)
    # 1) tta.json switches — ci-proto only
    if not args.quick and has_ci:
        g = cfg["global"]["ci_proto"]
        defaults = default_tta(g["source"], g.get("revision"))
        variants = {"default": {}}
        variants.update({f"{flag}={not defaults[flag]}": {flag: not defaults[flag]}
                         for flag in knob(args, "flags", BOOLEAN_FLAGS) if isinstance(defaults.get(flag), bool)})
        for label, overrides in variants.items():

            def compute(label=label, overrides=overrides):
                model = make_model("ci-proto", cfg, season=24, args=args, tta_overrides=overrides or None,
                                   name=f"ci-proto[{label}]")
                df = pd.concat([run_set(model, ds, horizon, n_windows, f"tta|{label}|{ds.name}").assign(
                    study="tta", variant=label, dataset=ds.name) for ds in sets], ignore_index=True)
                if df["status"].astype(str).str.contains(FATAL_CUDA).any():  # §14: never checkpoint fault rows
                    print(f"[fatal] CUDA fault in tta/{label}; not checkpointed, restart the process")
                    raise SystemExit(CUDA_FATAL_EXIT)
                return df

            tables.append(checkpointed(out / "_ckpt" / f"tta_{label}", compute))

    # 2) multivariate modes on genuinely related series (Ember markets, synthetic suite as control)
    if not args.quick and has_ci:
        for key in ("ci-proto-joint", "ci-proto-cohort", "ci-proto-indep"):

            def compute(key=key):
                model = make_model(key, cfg, season=24, args=args)
                return pd.concat([run_set(model, ds, horizon, n_windows, f"mode|{key}|{ds.name}", joint=True).assign(
                    study="mode", variant=key, dataset=ds.name) for ds in sets], ignore_index=True)

            tables.append(checkpointed(out / "_ckpt" / f"mode_{key}", compute))

    # 3) missing data
    keys = model_keys({"models": exp.get("models", ["ci-proto", "chronos-2", "tirex-2", "seasonal-naive"])}, args)
    for key in keys:
        if key == "ci-proto":  # per-dataset checkpoints: one WSL CUDA fault must not void the other domains (§14)
            for ds in sets:

                def compute_ds(ds=ds):
                    model = make_model("ci-proto", cfg, season=ds.season, horizon=horizon, args=args)
                    parts = []
                    for pattern in exp["missing_data"]["patterns"]:
                        for rate in knob(args, "rates", exp["missing_data"]["rates"]):
                            hook = partial(P.inject_missing, rate=rate, pattern=pattern, block=ds.season, seed=3)
                            parts.append(run_set(model, ds, horizon, n_windows, f"missing|{pattern}|{rate}",
                                                 history_transform=hook).assign(
                                study="missing", variant=f"{pattern}@{rate}", dataset=ds.name))
                    df = pd.concat(parts, ignore_index=True)
                    if df["status"].astype(str).str.contains(FATAL_CUDA).any():
                        print(f"[fatal] CUDA fault in missing/ci-proto/{ds.name}; not checkpointed, restart the process")
                        raise SystemExit(CUDA_FATAL_EXIT)
                    return df

                tables.append(checkpointed(out / "_ckpt" / f"missing_ci-proto__{ds.name}", compute_ds))
            continue

        def compute(key=key):
            parts = []
            for ds in sets:
                model = make_model(key, cfg, season=ds.season, horizon=horizon, args=args)
                for pattern in exp["missing_data"]["patterns"]:
                    for rate in knob(args, "rates", exp["missing_data"]["rates"]):
                        hook = partial(P.inject_missing, rate=rate, pattern=pattern, block=ds.season, seed=3)
                        parts.append(run_set(model, ds, horizon, n_windows, f"missing|{pattern}|{rate}",
                                             history_transform=hook).assign(
                            study="missing", variant=f"{pattern}@{rate}", dataset=ds.name))
            return pd.concat(parts, ignore_index=True)

        tables.append(checkpointed(out / "_ckpt" / f"missing_{key}", compute))
    table = pd.concat(tables, ignore_index=True)
    R.save(table, out / "metrics")
    paired_tests(table).to_csv(out / "tests_vs_default.csv", index=False)
    summary = table.groupby(["study", "dataset", "variant", "model"])[["MASE", "SQL", "COV80"]].mean().reset_index()
    summary.to_csv(out / "summary.csv", index=False)
    write_manifest(out, exp, args)
    print(summary.pivot_table(index=["study", "variant"], columns="model", values="SQL").round(3).to_string())


if __name__ == "__main__":
    main()
