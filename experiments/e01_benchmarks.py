"""E1 — General zero-shot benchmarks.

Suites
------
``fev``   fev-bench, 100 tasks (46 with covariates, 35 multivariate); SQL/MASE; skill score
          vs Seasonal Naive (geometric mean, clipped to [0.01, 100]); win rate; paired
          bootstrap 95 % CIs (B = 1000); failures replaced by Seasonal Naive.
``gift``  GIFT-Eval, 97 configurations; MASE and CRPS normalised by Seasonal Naive;
          average CRPS rank; breakdowns by domain, term, frequency and variates.
``ltsf``  The long-term forecasting protocol of the Mamba papers in Q1 journals
          (S-Mamba, Neurocomputing 619, 2025; RLMamba, ESWA 278, 2025; WaveletMamba,
          Neural Networks 2026): ETT/Electricity/Weather/Traffic/Exchange/Solar, chronological
          splits, train-split z-scoring, sliding test windows, MSE/MAE at H ∈ {96,192,336,720},
          run zero-shot with lookbacks 96/512/2048/full so ci-proto can be placed next to the
          published supervised numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from common import checkpointed, knob, load_config, make_model, model_keys, out_dir, parse, write_manifest

LTSF_FILES = {"ETTh1": ("ETTh1.csv", "h"), "ETTh2": ("ETTh2.csv", "h"), "ETTm1": ("ETTm1.csv", "15min"),
              "ETTm2": ("ETTm2.csv", "15min"), "Electricity": ("electricity.csv", "h"),
              "Weather": ("weather.csv", "10min"), "Traffic": ("traffic.csv", "h"),
              "Exchange": ("exchange_rate.csv", "D"), "Solar": ("solar_AL.csv", "10min")}
LTSF_LARGE = {"Electricity", "Weather", "Traffic", "Exchange", "Solar"}  # E1 part "ltsf_large"; ETT = "ltsf"
FEV_BASE = "https://raw.githubusercontent.com/autogluon/fev/main/benchmarks/fev_bench/"


def ltsf_borders(name: str, n: int, seq_len: int) -> tuple[int, int, int, int]:
    """Train end, test start (minus lookback handled by caller), test end — Time-Series-Library splits."""
    if name.startswith("ETTh"):
        train, val, test = 12 * 30 * 24, 4 * 30 * 24, 4 * 30 * 24
    elif name.startswith("ETTm"):
        train, val, test = 12 * 30 * 96, 4 * 30 * 96, 4 * 30 * 96
    else:
        train, test = int(n * 0.7), int(n * 0.2)
        val = n - train - test
    return train, train + val, train + val + test, n


def _free_cuda() -> None:
    import gc

    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:
        pass


# Joint (multivariate) mode only up to this many variates: ci-proto's joint pass over Electricity (321)
# at full context needs ~20 GB, which spills into WSL shared memory and then fails on an 8 GB GPU.
JOINT_MAX_VARIATES = 32


def ltsf_eval(model, values: np.ndarray, name: str, horizons: list[int], lookbacks: list, stride: int,
              joint: bool) -> list[dict]:
    train_end, test_start, test_end, _ = ltsf_borders(name, values.shape[1], 96)
    mu = values[:, :train_end].mean(axis=1, keepdims=True)
    sd = values[:, :train_end].std(axis=1, keepdims=True) + 1e-8
    z = (values - mu) / sd
    rows = []
    for horizon in horizons:
        cutoffs = range(test_start, test_end - horizon + 1, stride)
        for lb in lookbacks:
            se = ae = count = 0.0
            status = "ok"
            try:
                for c in cutoffs:
                    start = 0 if lb == "full" else max(0, c - int(lb))
                    hist = z[:, start:c]
                    if joint:
                        q = model.predict(hist, horizon)
                    else:
                        q = np.concatenate([np.asarray(x).reshape(1, 9, horizon)
                                            for x in model.predict_many(list(hist), horizon)])
                    err = q[:, 4, :] - z[:, c : c + horizon]
                    se += float(np.sum(err**2))
                    ae += float(np.sum(np.abs(err)))
                    count += err.size
            except Exception as exc:  # noqa: BLE001 - CUDA OOM on 8 GB: record the cell as failed, keep going
                if "out of memory" not in str(exc).lower():
                    raise
                status, count = "oom", 0
                _free_cuda()
            rows.append({"dataset": name, "model": model.name, "horizon": horizon, "lookback": lb,
                         "mode": "joint" if joint else "channel-independent",
                         "MSE": se / count if count else np.nan, "MAE": ae / count if count else np.nan,
                         "n_windows": len(cutoffs), "stride": stride, "n_variates": values.shape[0],
                         "status": status})
            print(f"  {model.name:>16s} {name:>11s} H={horizon:<4d} L={lb!s:<5s} MSE={rows[-1]['MSE']:.4f} {status}",
                  flush=True)
    return rows


def synthetic_ett(length: int = 17420, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    base = [np.sin(2 * np.pi * t / 24 + k) + 0.3 * np.sin(2 * np.pi * t / 168) for k in range(7)]
    return np.stack(base) + 0.3 * np.cumsum(rng.normal(size=(7, length)), axis=1) / np.sqrt(length) * 10


def main() -> None:
    args = parse(__doc__)
    cfg = load_config(args.config)
    exp = cfg["experiments"]["E1_general_benchmarks"]
    out = out_dir(cfg, args, "E1_general_benchmarks")
    suites = exp["suites"]
    # parts (--only): ltsf = the four ETT sets, ltsf_large = Electricity/Weather/Traffic/Exchange/Solar
    # (days of GPU time, so the queue runs them last), fev = fev-bench. Default: all parts.
    parts = set(args.only or ["ltsf", "ltsf_large", "fev"])

    # ---------------------------------------------------------------- LTSF zero-shot
    lt = suites["ltsf_zero_shot"]
    rows = []
    if args.quick:
        data = {"ETTh1": synthetic_ett()}
        horizons, lookbacks, stride = [96], [96, 512], 240
    else:
        data = {}
        wanted = knob(args, "ltsf_datasets", list(LTSF_FILES))
        for name, (fname, _freq) in LTSF_FILES.items():
            if name not in wanted or ("ltsf_large" if name in LTSF_LARGE else "ltsf") not in parts:
                continue
            path = Path(args.data_dir) / "ltsf" / fname
            if path.exists():
                df = pd.read_csv(path)
                data[name] = df.drop(columns=[df.columns[0]]).to_numpy(float).T
        horizons = knob(args, "horizons", lt["horizons"])
        lookbacks = knob(args, "lookbacks", lt["lookback"])
        stride = int(knob(args, "stride", lt.get("stride", 1)))
    ckpt = out / "_ckpt"
    for key in model_keys({"models": lt.get("models", ["ci-proto", "chronos-2", "tirex-2", "timesfm-2.5",
                                                       "seasonal-naive"])}, args):
        for name, values in data.items():
            season = (96 if name.startswith("ETTm") else 144 if name in ("Weather", "Solar")
                      else 7 if name == "Exchange" else 24)

            large = lt.get("large", {}) if name in LTSF_LARGE and not args.quick else {}
            lbs = [lb for lb in lookbacks
                   if not large or lb != "full" or key in large.get("full_lookback_models", [key])]
            st = int(large.get("stride", stride))

            def compute(key=key, name=name, values=values, season=season, lbs=lbs, st=st):
                model = make_model(key, cfg, season=season, args=args)
                part = []
                joint_ok = getattr(model, "multivariate", False) and values.shape[0] <= JOINT_MAX_VARIATES
                for joint in ([True, False] if joint_ok else [False]):
                    part += ltsf_eval(model, values, name, horizons, lbs, st, joint)
                return pd.DataFrame(part)

            # one checkpoint per model and dataset: a crash or a restart loses at most one cell
            rows.append(compute() if args.quick else checkpointed(ckpt / f"ltsf_{key}_{name}", compute))
    if not args.quick:  # the table always holds every finished cell, whichever part produced it
        rows = [pd.read_csv(f) for f in sorted(ckpt.glob("ltsf_*.csv.gz"))] + \
               [pd.read_parquet(f) for f in sorted(ckpt.glob("ltsf_*.parquet"))]
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(out / "ltsf_zero_shot.csv", index=False)

    # -------------------------------------------------------------- fev-bench (independent re-run)
    if not args.quick and "fev" in parts and "fev" in knob(args, "suites", ["ltsf", "fev"]):
        from ciexp.adapters import run_fev_benchmark

        fb = suites["fev_bench"]
        tasks_yaml = FEV_BASE + knob(args, "fev_tasks", "tasks") + ".yaml"
        fev_dir = out / "fev"
        fev_dir.mkdir(exist_ok=True)
        for key in model_keys(fb, args):
            path = fev_dir / f"{key}.csv"
            done_df = pd.read_csv(path) if path.exists() else pd.DataFrame()
            done = set(done_df["task_name"]) if not done_df.empty else set()
            model = make_model(key, cfg, season=1, args=args)
            kw = fb.get("model_options", {}).get(key, {})

            def save(summary, path=path):
                row = pd.DataFrame([summary])
                row.to_csv(path, mode="a", header=not path.exists(), index=False)

            print(f"fev-bench {key}: {len(done)} tasks already done")
            run_fev_benchmark(model, tasks_yaml, max_tasks=knob(args, "fev_max_tasks", None), done=done,
                              on_result=save, fatal_path=fev_dir / f"_cuda_fatal_{key}.txt", **kw)
        (out / "gift_eval_instructions.json").write_text(json.dumps({
            "repo": "https://github.com/SalesforceAIResearch/gift-eval",
            "predictor": "ciexp.adapters.GluonTSAdapter(model, prediction_length=ds.prediction_length, freq=ds.freq)",
            "metrics": ["MASE", "MeanWeightedSumQuantileLoss(quantile_levels=0.1..0.9)"],
            "aggregation": "normalise by Seasonal Naive per config, geometric mean; average CRPS rank",
        }, indent=2))
    write_manifest(out, exp, args)


if __name__ == "__main__":
    main()
