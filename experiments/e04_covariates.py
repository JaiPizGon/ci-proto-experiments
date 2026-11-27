"""E4 — Value of calendar features and covariates, and robustness to noisy future covariates.

Cumulative conditions: C0 history only · C1 + calendar (timestamps + freq) · C2 + known calendar
flags · C3 + weather as *past* covariates · C4 + weather as *future-known* covariates (oracle
actuals) · C5 C4 with Gaussian noise of λ·σ added to the future weather inside the forecast
window (λ = 0.5, 1, 5), emulating weather-forecast error.

Datasets: UCI Bike Sharing (ci-proto's walkthrough; flags holiday/workingday, weather temp, past
humidity/windspeed) and UCI Beijing multi-site air quality (12 stations, PM2.5, per-station weather
TEMP/PRES/DEWP/RAIN/WSPM; no calendar flags, so C2 is skipped).

Covariate-capable TSFMs (ci-proto, Chronos-2, TiRex-2) run every condition; univariate TSFMs run
C0/C1 as references. Paired Wilcoxon signed-rank tests (per window) of each condition against C0,
Holm-adjusted within model, are written to ``tests_vs_C0.csv``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from common import checkpointed, knob, load_config, make_model, model_keys, out_dir, parse, write_manifest

from ciexp import data as D
from ciexp import protocols as P
from ciexp import runner as R
from ciexp import stats as S

COV_KINDS = {"ci-proto", "chronos-2", "tirex-2"}


@dataclass
class CovData:
    ds: D.SeriesSet          # target panel (covariates set per condition)
    flags: np.ndarray | None  # known calendar flags [Kf, T] (shared)
    weather: np.ndarray      # oracle-able weather [Kw, T] or [V, Kw, T]
    other_past: np.ndarray | None  # past-only covariates [Kp, T] or [V, Kp, T]


def _stack(parts: list[np.ndarray | None], n_series: int) -> np.ndarray | None:
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    if any(p.ndim == 3 for p in parts):
        parts = [p if p.ndim == 3 else np.broadcast_to(p, (n_series, *p.shape)) for p in parts]
        return np.concatenate(parts, axis=1)
    return np.concatenate(parts, axis=0)


def synthetic_demand(seed: int = 0, days: int = 730) -> CovData:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2022-01-01", periods=24 * days, freq="h")
    hour, dow = index.hour.to_numpy(), index.dayofweek.to_numpy()
    holiday = np.isin(index.strftime("%m-%d"), ["01-01", "05-01", "12-25", "07-04"]).astype(float)
    working = ((dow < 5) & (holiday == 0)).astype(float)
    temp = 15 + 10 * np.sin(2 * np.pi * (index.dayofyear.to_numpy() - 100) / 365) + rng.normal(0, 2, len(index))
    humidity = 60 + rng.normal(0, 10, len(index))
    peaks = working * (np.exp(-((hour - 8) ** 2) / 4) + np.exp(-((hour - 17) ** 2) / 4)) * 300
    leisure = (1 - working) * np.exp(-((hour - 14) ** 2) / 12) * 200
    y = np.maximum(0, 50 + peaks + leisure + 6 * (temp - 15) + rng.normal(0, 20, len(index)))
    ds = D.SeriesSet("synthetic_demand", y[None], index, "h", 24, ["demand"])
    return CovData(ds, np.stack([holiday, working]), temp[None], humidity[None])


def load_bike(data_dir: Path) -> CovData:
    path = data_dir / "bike_sharing_openml42712.csv"
    if path.exists():
        raw = pd.read_csv(path)
    else:
        from sklearn.datasets import fetch_openml

        raw = fetch_openml(data_id=42712, as_frame=True, parser="liac-arff").frame
        raw.to_csv(path, index=False)
    ds = D.bike_sharing_uci(raw)  # past: temp, feel_temp, humidity, windspeed; future: holiday, workingday
    past = ds.past_covariates
    return CovData(replace(ds, past_covariates=None, future_covariates=None), ds.future_covariates,
                   past[0:1], past[2:4])


def load_beijing(data_dir: Path) -> CovData:
    ds = D.beijing_air_quality(data_dir / "beijing_air_quality")  # past: [12, 5, T] weather per station
    return CovData(replace(ds, past_covariates=None, future_covariates=None), None, ds.past_covariates, None)


def conditions(cd: CovData) -> list[str]:
    base = ["C0", "C1"] + (["C2"] if cd.flags is not None else []) + ["C3", "C4"]
    return base + [f"C5_l{lam}" for lam in (0.5, 1.0, 5.0)]


def condition(cd: CovData, name: str) -> tuple[D.SeriesSet, bool]:
    """Dataset carrying the covariates of a condition, and whether calendar features are on."""
    V = cd.ds.values.shape[0]
    if name == "C0":
        return cd.ds, False
    if name == "C1":
        return cd.ds, True
    if name == "C2":
        return replace(cd.ds, future_covariates=cd.flags), True
    if name == "C3":
        return replace(cd.ds, future_covariates=cd.flags,
                       past_covariates=_stack([cd.other_past, cd.weather], V)), True
    if name.startswith("C4") or name.startswith("C5"):
        return replace(cd.ds, future_covariates=_stack([cd.flags, cd.weather], V),
                       past_covariates=_stack([cd.other_past], V)), True
    raise ValueError(name)


def noise_hook(cd: CovData, lam: float, seed: int):
    """Adds N(0, (λ σ_k)^2) to the weather rows of the future covariates, inside the horizon only."""
    n_flags = 0 if cd.flags is None else cd.flags.shape[0]
    w = cd.weather if cd.weather.ndim == 2 else cd.weather.transpose(1, 0, 2).reshape(cd.weather.shape[1], -1)
    sigma = lam * np.nanstd(w, axis=-1)  # one σ per weather variable

    def f(future: np.ndarray, horizon_start: int, cutoff: int) -> np.ndarray:
        rng = np.random.default_rng(seed + cutoff)
        tail = future.shape[-1] - horizon_start
        for k, s in enumerate(sigma):
            future[..., n_flags + k, horizon_start:] += rng.normal(0.0, s, size=(*future.shape[:-2], tail))
        return future

    return f


def wilcoxon_vs_c0(metrics: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    key = ["series", "cutoff"]
    for (dataset, h, model), g in metrics.groupby(["dataset", "horizon_cfg", "model"]):
        base = g[g["condition"] == "C0"].set_index(key)[metric]
        for cond, gc in g.groupby("condition"):
            if cond == "C0":
                continue
            a, b = gc.set_index(key)[metric].align(base, join="inner")
            rows.append({"dataset": dataset, "horizon": h, "model": model, "condition": cond, "metric": metric,
                         "median_rel_change_pct": float(100 * np.nanmedian(a / b - 1)),
                         "mean_C0": float(b.mean()), "mean_cond": float(a.mean()), "n": int(len(a)),
                         "p_wilcoxon": S.wilcoxon_pair(a.to_numpy(), b.to_numpy())})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm"] = out.groupby(["dataset", "horizon", "model", "metric"])["p_wilcoxon"].transform(S.holm)
    return out


def main() -> None:
    args = parse(__doc__)
    cfg = load_config(args.config)
    exp = cfg["experiments"]["E4_covariates_calendar"]
    out = out_dir(cfg, args, "E4_covariates_calendar")
    if args.quick:
        datasets = [synthetic_demand()]
    else:
        loaders = {"bike_sharing_uci": load_bike, "beijing_air_quality": load_beijing}
        datasets = [loaders[n](Path(args.data_dir)) for n in knob(args, "datasets", exp["datasets"])]
    horizons = [24] if args.quick else knob(args, "horizons", exp["horizons"])
    n_windows = 6 if args.quick else knob(args, "n_windows", exp["n_windows"])
    keys = model_keys(exp, args)
    zoo = cfg["models"]
    tables = []
    for key in keys:  # model outermost: each pretrained model is loaded once (cached)
        kind = zoo.get(key, {}).get("kind", key)

        for cd in datasets:
            for horizon in horizons:

                def compute(key=key, kind=kind, cd=cd, horizon=horizon):
                    parts = []
                    origins = P.rolling_origins(cd.ds.values.shape[1], horizon, n_windows=n_windows, step=168)
                    model = make_model(key, cfg, season=cd.ds.season, horizon=horizon, args=args)
                    for cond in conditions(cd):
                        if kind not in COV_KINDS and cond not in ("C0", "C1"):
                            continue  # univariate TSFMs ignore covariates: C0/C1 are their references
                        hook = noise_hook(cd, float(cond.split("_l")[1]), seed=7) if cond.startswith("C5") else None
                        dsc, cal = condition(cd, cond)
                        fc = R.run_backtest(model, dsc, origins, joint=False, covariates=True, calendar=cal,
                                            future_transform=hook, tag=f"{cd.ds.name}|h{horizon}|{cond}",
                                            train_end=origins[0].cutoff)
                        parts.append(R.metrics_table(fc, dsc).assign(condition=cond, horizon_cfg=horizon,
                                                                     dataset=cd.ds.name))
                        print(f"  {key} {cd.ds.name} h={horizon} {cond}: done", flush=True)
                    return pd.concat(parts, ignore_index=True)

                # one checkpoint per (model, dataset, horizon): a ci-proto cell takes up to ~2.5 h
                tables.append(checkpointed(out / "_ckpt" / f"{key}_{cd.ds.name}_h{horizon}", compute))
    metrics = pd.concat(tables, ignore_index=True)
    R.save(metrics, out / "metrics")
    summary = R.aggregate(metrics, by=("dataset", "horizon_cfg", "condition", "model"))
    summary.to_csv(out / "summary.csv", index=False)
    tests = pd.concat([wilcoxon_vs_c0(metrics, m) for m in ("MASE", "SQL")], ignore_index=True)
    tests.to_csv(out / "tests_vs_C0.csv", index=False)
    write_manifest(out, exp, args)
    print(summary.pivot_table(index=["dataset", "horizon_cfg", "condition"], columns="model", values="MASE")
          .round(3).to_string())


if __name__ == "__main__":
    main()
