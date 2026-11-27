"""E10 — GIFT-Eval (Aksu et al., 2024): 97 dataset/frequency/term configurations, official splits and metrics.

Data splits, test windows and every metric come from the official ``gift-eval`` package, run in the
isolated ``envs/gifteval`` env (it pins numpy 1.26 / gluonts 0.15); this script only produces the nine
quantile forecasts with our wrappers and hands them over as arrays. Protocol details mirror the
official notebooks: multivariate datasets are kept multivariate (``to_univariate=False``); models without
multivariate support forecast each variate separately; target only (no timestamps / calendar features
for any model); contexts stored up to 32,768 points per window.

Leaderboard: per configuration, MASE and CRPS (mean weighted sum quantile loss) are divided by Seasonal
Naive's (official file) and aggregated by geometric mean; models are also ranked by CRPS per
configuration. Our re-runs of comparators are compared with their official result files (reproduction
fidelity); ci-proto has no official GIFT-Eval entry, so this is its first independent GIFT-Eval result.
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from common import ROOT, knob, load_config, make_model, model_keys, out_dir, parse, write_manifest

ENV = ROOT / "envs" / "gifteval"
PY = ENV / ".venv" / "bin" / "python"
OFFICIAL = {"Seasonal_Naive": "Seasonal_Naive", "Chronos-2": "Chronos-2", "TimesFM-3": "TimesFM-3",
            "TimesFM-2.5": "TimesFM-2.5", "TiRex-2": "TiRex-2-Zeroshot", "TiRex": "TiRex", "Moirai-2": "Moirai2",
            "Naive": "Naive", "Auto_ETS": "Auto_ETS", "Auto_Theta": "Auto_Theta", "Auto_Arima": "Auto_Arima",
            "N-BEATS": "N-BEATS", "PatchTST": "PatchTST", "Toto-2.0-313m": "Toto-2.0-313m",
            "Sundial": "sundial_base_128m", "TabPFN-TS": "TabPFN-TS", "Chronos-Bolt": "chronos_bolt_base"}
RAW = "https://raw.githubusercontent.com/SalesforceAIResearch/gift-eval/main/results/{}/all_results.csv"


def worker(*args: str, data_dir: Path) -> str:
    env = {**os.environ, "GIFT_EVAL": str(data_dir)}
    res = subprocess.run([str(PY), "-I", str(ENV / "worker.py"), *args], capture_output=True, text=True, env=env)
    if res.returncode:
        raise RuntimeError(res.stderr[-2000:])
    return res.stdout.strip().splitlines()[-1] if res.stdout.strip() else ""


def official_results(dest: Path) -> pd.DataFrame:
    dest.mkdir(parents=True, exist_ok=True)
    frames = []
    for label, folder in OFFICIAL.items():
        f = dest / f"{folder}.csv"
        if not f.exists():
            try:
                f.write_bytes(urllib.request.urlopen(RAW.format(urllib.request.quote(folder)), timeout=60).read())
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] official GIFT-Eval results for {folder}: {exc}")
                continue
        df = pd.read_csv(f)
        df["model"] = label
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def predict_config(model, meta: dict, chunk: int = 512) -> np.ndarray:
    """Quantiles for every test entry: [N, Q, H] (univariate) or [N, Q, H, V] (multivariate)."""
    H, entries = meta["prediction_length"], meta["entries"]
    if meta["target_dim"] == 1:
        out = []
        for k in range(0, len(entries), chunk):
            qs = model.predict_many([e["target"] for e in entries[k : k + chunk]], H)
            out += [np.asarray(q).reshape(len(q) if np.ndim(q) == 3 else 1, 9, H)[0] for q in qs]
        return np.stack(out)
    out = []
    for e in entries:
        x = np.asarray(e["target"])  # [V, T]
        if model.multivariate:
            q = np.asarray(model.predict(x, H))  # [V, Q, H]
        else:
            q = np.concatenate([np.asarray(r).reshape(1, 9, H) for r in model.predict_many(list(x), H)])
        out.append(np.transpose(q, (1, 2, 0)))  # [Q, H, V]
    return np.stack(out)


def leaderboard(rows: pd.DataFrame, official: pd.DataFrame, out: Path) -> pd.DataFrame:
    allr = pd.concat([official, rows], ignore_index=True)
    allr = allr.rename(columns={"eval_metrics/MASE[0.5]": "MASE",
                                "eval_metrics/mean_weighted_sum_quantile_loss": "CRPS"})
    sn = allr[allr["model"] == "Seasonal_Naive"].set_index("dataset")[["MASE", "CRPS"]]
    configs = sorted(set(rows["dataset"]))
    allr = allr[allr["dataset"].isin(configs)]
    complete = allr.groupby("model")["dataset"].nunique()
    allr = allr[allr["model"].isin(complete[complete == len(configs)].index)]
    rel = allr.join(sn, on="dataset", rsuffix="_sn")
    rel["MASE_rel"] = rel["MASE"] / rel["MASE_sn"]
    rel["CRPS_rel"] = rel["CRPS"] / rel["CRPS_sn"]
    rel["CRPS_rank"] = rel.groupby("dataset")["CRPS"].rank(method="average")
    lb = rel.groupby("model").agg(MASE_gmean=("MASE_rel", lambda x: float(np.exp(np.log(x.clip(1e-6)).mean()))),
                                  CRPS_gmean=("CRPS_rel", lambda x: float(np.exp(np.log(x.clip(1e-6)).mean()))),
                                  avg_CRPS_rank=("CRPS_rank", "mean"), n_configs=("dataset", "nunique"))
    lb = lb.sort_values("CRPS_gmean")
    lb.to_csv(out / "leaderboard.csv")
    for col in ("term", "domain"):
        key = rel["dataset"].str.split("/").str[2] if col == "term" else rel["domain"]
        rel.assign(_k=key).groupby(["_k", "model"])["CRPS_rel"].apply(
            lambda x: float(np.exp(np.log(x.clip(1e-6)).mean()))).unstack(0).to_csv(out / f"crps_by_{col}.csv")
    rel.to_csv(out / "per_config_relative.csv", index=False)
    return lb


def main() -> None:
    args = parse(__doc__)
    cfg = load_config(args.config)
    exp = cfg["experiments"].get("E10_gift_eval", {})
    out = out_dir(cfg, args, "E10_gift_eval")
    data_dir = Path(args.data_dir) / "gift_eval"
    cache = Path(args.data_dir) / "gift_eval_cache"
    cache.mkdir(exist_ok=True)
    if not PY.exists() or not data_dir.exists():
        raise SystemExit("GIFT-Eval needs envs/gifteval (cd envs/gifteval && uv sync) and data/gift_eval "
                         "(huggingface Salesforce/GiftEval)")
    configs = json.loads(worker("configs", data_dir=data_dir))
    configs = configs[: int(knob(args, "max_configs", len(configs)))]
    keys = model_keys({"models": exp.get("models", ["ci-proto", "chronos-2", "tirex-2", "seasonal-naive"])}, args)
    rows_dir = out / "rows"
    for key in keys:
        for c in configs:
            safe = c["config"].replace("/", "_")
            row_path = rows_dir / key / f"{safe}.json"
            if row_path.exists():
                continue
            from ciexp.adapters import fatal_list
            fatal_path = rows_dir / f"_cuda_fatal_{key}.txt"
            if c["config"] in fatal_list(fatal_path):
                print(f"  [skip] {key} {c['config']}: crashed the CUDA driver in an earlier attempt", flush=True)
                continue
            pkl = cache / f"{c['ds_name'].replace('/', '_')}_{c['term']}.pkl"
            if not pkl.exists():
                worker("export", c["ds_name"], c["term"], str(pkl), data_dir=data_dir)
            meta = pickle.loads(pkl.read_bytes())
            season = int(meta.get("season", 1))  # gluonts get_seasonality(freq), as in the official MASE
            model = make_model(key, cfg, season=season, horizon=meta["prediction_length"], args=args)
            tic = time.perf_counter()
            try:
                q = predict_config(model, meta)
            except Exception as exc:  # noqa: BLE001  recorded; fev/GIFT rule: impute with Seasonal Naive
                print(f"  [fail] {key} {c['config']}: {type(exc).__name__}: {exc}"[:300], flush=True)
                from ciexp.adapters import CUDA_FATAL_EXIT, is_fatal_cuda
                if is_fatal_cuda(exc):
                    from ciexp.adapters import record_fatal
                    fatal_path.parent.mkdir(parents=True, exist_ok=True)
                    record_fatal(fatal_path, c["config"])
                    print("  CUDA context lost; aborting so a fresh process can resume", flush=True)
                    raise SystemExit(CUDA_FATAL_EXIT) from exc
                continue
            elapsed = time.perf_counter() - tic
            npz = rows_dir / key / f"{safe}.npz"
            npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(npz, quantiles=q.astype(np.float32))
            tmp = row_path.with_suffix(".tmp")
            worker("evaluate", c["ds_name"], c["term"], str(npz), model.name, str(tmp), data_dir=data_dir)
            row = json.loads(tmp.read_text())
            row.update(inference_time_s=elapsed, n_entries=len(meta["entries"]))
            row_path.write_text(json.dumps(row))
            tmp.unlink()
            npz.unlink()  # keep only the metrics (quantile arrays can be large)
            print(f"  {model.name:>14s} {c['config']:<42s} MASE={row['eval_metrics/MASE[0.5]']:.3f} "
                  f"CRPS={row['eval_metrics/mean_weighted_sum_quantile_loss']:.3f} ({elapsed:.0f}s)", flush=True)
    rows = pd.DataFrame([json.loads(p.read_text()) for p in sorted(rows_dir.glob("*/*.json"))])
    rows.to_csv(out / "our_results.csv", index=False)
    official = official_results(Path(args.data_dir) / "gift_results")
    if not rows.empty and not official.empty:
        lb = leaderboard(rows.assign(model=rows["model"] + " (ours)"), official, out)
        print(lb.round(3).to_string())
        fid = []
        ours_to_official = {"SeasonalNaive": "Seasonal_Naive"}
        for m, g in rows.groupby("model"):
            ref = official[official["model"] == ours_to_official.get(m, m)]
            if ref.empty:
                continue
            rec = {"model": m, "official": ours_to_official.get(m, m)}
            for metric, col in (("CRPS", "eval_metrics/mean_weighted_sum_quantile_loss"),
                                ("MASE", "eval_metrics/MASE[0.5]")):
                a, b = g.set_index("dataset")[col].align(ref.set_index("dataset")[col], join="inner")
                rec.update({"n": len(a), f"median_ratio_{metric}": float(np.median(a / b)),
                            f"share_within_5pct_{metric}": float(np.mean(np.abs(a / b - 1) < 0.05))})
            fid.append(rec)
        pd.DataFrame(fid).to_csv(out / "reproduction_fidelity.csv", index=False)
    write_manifest(out, exp, args)


if __name__ == "__main__":
    main()
