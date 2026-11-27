"""Dataset loaders and leakage-free synthetic series.

Every loader returns a :class:`SeriesSet`: an aligned ``[V, T]`` target panel, optional
covariates and the metadata the protocols need (timestamps, pandas frequency, seasonal
period). Public raw files are not redistributed; ``download_hint`` says where each comes
from so the experiments can be reproduced from the original sources.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class SeriesSet:
    name: str
    values: np.ndarray  # [V, T]
    index: pd.DatetimeIndex
    freq: str
    season: int
    ids: list[str]
    past_covariates: np.ndarray | None = None  # [P, T] (shared) or [V, P, T]
    future_covariates: np.ndarray | None = None  # [K, T] known for the whole axis
    static: dict[str, list[object]] = field(default_factory=dict)
    covariate_names: dict[str, list[str]] = field(default_factory=dict)
    download_hint: str = ""

    @property
    def timestamps_ns(self) -> np.ndarray:
        return self.index.asi8.astype(np.int64)

    def subset(self, rows: list[int]) -> SeriesSet:
        static = {k: [v[i] for i in rows] for k, v in self.static.items()}
        past = self.past_covariates
        if past is not None and past.ndim == 3:
            past = past[rows]
        return SeriesSet(self.name, self.values[rows], self.index, self.freq, self.season,
                         [self.ids[i] for i in rows], past, self.future_covariates, static,
                         self.covariate_names, self.download_hint)


SEASON = {"10s": 360, "min": 1440, "5min": 288, "10min": 144, "15min": 96, "30min": 48, "h": 24,
          "D": 7, "W": 52, "MS": 12, "ME": 12, "QS": 4, "QE": 4, "YS": 1, "YE": 1}


# ------------------------------------------------------------------------- bundled example
def bike_sharing_example() -> SeriesSet:
    """The six days of hourly rentals bundled with the ci-proto package (CC BY 4.0)."""
    try:
        path = resources.files("continuousintelligence") / "_data" / "bike_sharing.csv"
        frame = pd.read_csv(path)
    except ModuleNotFoundError:
        local = Path(__file__).resolve().parents[1] / "data" / "bike_sharing.csv"
        frame = pd.read_csv(local)
    index = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"]))
    return SeriesSet("bike_sharing_example", frame["rentals"].to_numpy(float)[None], index, "h", 24, ["rentals"],
                     download_hint="bundled with continuousintelligence")


def bike_sharing_uci(raw: pd.DataFrame) -> SeriesSet:
    """UCI Bike Sharing (OpenML 42712) on a complete hourly grid, as in the ci-proto walkthrough.

    ``raw = sklearn.datasets.fetch_openml(data_id=42712, as_frame=True, parser="liac-arff").frame``
    Future-known covariates: holiday, workingday (calendar). Past covariates: weather.
    """
    day = np.r_[0, np.diff(raw["weekday"].astype(int).to_numpy()) % 7].cumsum()
    stamps = pd.Timestamp("2011-01-01") + pd.to_timedelta(day, unit="D") + pd.to_timedelta(
        raw["hour"].astype(int).to_numpy(), unit="h")
    grid = pd.date_range(stamps[0], stamps[-1], freq="h")
    frame = raw.set_index(stamps).reindex(grid)
    known = np.stack([frame["holiday"].map({"True": 1, "False": 0, True: 1, False: 0}).astype(float).to_numpy(),
                      frame["workingday"].map({"True": 1, "False": 0, True: 1, False: 0}).astype(float).to_numpy()])
    past = np.stack([frame[c].astype(float).to_numpy() for c in ("temp", "feel_temp", "humidity", "windspeed")])
    return SeriesSet("bike_sharing_uci", frame["count"].astype(float).to_numpy()[None], grid, "h", 24, ["count"],
                     past_covariates=past, future_covariates=known,
                     covariate_names={"past": ["temp", "feel_temp", "humidity", "windspeed"],
                                      "future": ["holiday", "workingday"]},
                     download_hint="https://doi.org/10.24432/C5W894 (OpenML id 42712)")


# --------------------------------------------------------------------- electricity prices
def ember_daily_prices(path: str | Path, countries: list[str] | None = None, *, start: str = "2015-01-01",
                       end: str = "2025-12-31", min_coverage: float = 0.99) -> SeriesSet:
    """Ember European wholesale electricity prices, daily, EUR/MWh (Energy and AI 100914 data).

    Expects Ember's CSV with columns ``Country``, ``ISO3 Code``, ``Date``, ``Price (EUR/MWhe)``.
    Negative and near-zero prices are kept; no imputation is performed. Markets observed on
    fewer than ``min_coverage`` of the days in [start, end] are dropped: with the defaults this
    leaves exactly the 23 markets of Energy and AI 100914 (the current Ember file also lists
    Albania, Moldova, Kosovo, … with no or partial 2015–2025 data).
    """
    df = pd.read_csv(path)

    def first(*needles: str) -> str | None:
        for needle in needles:
            for c in df.columns:
                if needle in c.lower():
                    return c
        return None

    date_col = first("date", "time")
    price_col = first("price")
    key = first("iso3", "iso", "country", "area")
    if date_col is None or price_col is None or key is None:
        raise ValueError(f"unexpected Ember columns: {list(df.columns)}")
    dates = pd.to_datetime(df[date_col])
    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)
    df[date_col] = dates
    wide = df.pivot_table(index=date_col, columns=key, values=price_col)
    wide = wide.asfreq("D").loc[start:end]
    if countries:
        wide = wide[[c for c in countries if c in wide.columns]]
    elif min_coverage:
        wide = wide.loc[:, wide.notna().mean() >= min_coverage]
    return SeriesSet("ember_daily", wide.to_numpy(float).T, pd.DatetimeIndex(wide.index), "D", 7,
                     [str(c) for c in wide.columns],
                     download_hint="https://ember-energy.org/data/european-wholesale-electricity-price-data/")


def ember_hourly_prices(directory: str | Path, country_file: str, name: str, *, start: str = "2019-01-01",
                        end: str = "2025-12-31") -> SeriesSet:
    """Ember hourly day-ahead prices (UTC grid, EUR/MWh), the public substitute for the licensed
    EPEX-DE / N2EX-UK / NEMS-SG series of Energy and AI 100886 (2026). CC BY 4.0."""
    df = pd.read_csv(Path(directory) / country_file)
    stamps = pd.to_datetime(df["Datetime (UTC)"])
    series = pd.Series(df["Price (EUR/MWhe)"].astype(float).to_numpy(), index=stamps).sort_index()
    series = series[~series.index.duplicated()].asfreq("h").loc[start:end]
    return SeriesSet(name, series.to_numpy()[None], pd.DatetimeIndex(series.index), "h", 168, [name],
                     download_hint="https://ember-energy.org/data/european-wholesale-electricity-price-data/")


def hourly_prices(path: str | Path, time_col: str, value_col: str, name: str, freq: str = "h") -> SeriesSet:
    """EPEX-DE, N2EX-UK (hourly) or NEMS-SG (30-min) price CSV (Energy and AI 100886 markets)."""
    df = pd.read_csv(path, parse_dates=[time_col]).set_index(time_col).sort_index()
    series = df[value_col].astype(float).asfreq(freq)
    return SeriesSet(name, series.to_numpy()[None], pd.DatetimeIndex(series.index), freq, SEASON[freq], [name])


# -------------------------------------------------------------- long-context electricity
def uci_household_power(path: str | Path, column: str = "Global_active_power") -> SeriesSet:
    """UCI 'Individual household electric power consumption': 1-minute data, ≈2.07M rows.

    Long enough to test ci-proto's 2M+ context claim on a single real series.
    """
    df = pd.read_csv(path, sep=";", na_values=["?"], low_memory=False)
    stamps = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
    series = pd.Series(df[column].astype(float).to_numpy(), index=stamps).sort_index().asfreq("min")
    return SeriesSet("uci_household_power", series.to_numpy()[None], pd.DatetimeIndex(series.index), "min", 1440,
                     [column], download_hint="https://doi.org/10.24432/C58K54")


def generic_csv(path: str | Path, time_col: str, value_cols: list[str], freq: str, *, name: str | None = None,
                past_cols: list[str] | None = None, future_cols: list[str] | None = None) -> SeriesSet:
    df = pd.read_csv(path, parse_dates=[time_col]).set_index(time_col).sort_index().asfreq(freq)
    past = df[past_cols].to_numpy(float).T if past_cols else None
    future = df[future_cols].to_numpy(float).T if future_cols else None
    return SeriesSet(name or Path(path).stem, df[value_cols].to_numpy(float).T, pd.DatetimeIndex(df.index), freq,
                     SEASON.get(freq, 1), list(value_cols), past, future,
                     covariate_names={"past": past_cols or [], "future": future_cols or []})


# ------------------------------------------------------------------- synthetic, leak-free
def synthetic_suite(length: int = 4096, seed: int = 0) -> SeriesSet:
    """Processes no pretraining corpus can contain, generated with a fixed seed.

    Includes structural breaks (mean, variance, seasonality, trend) to probe the
    'context inertia' failure mode of long contexts (Neurocomputing, 2026), plus
    chaotic and intermittent series used in zero-shot TSFM stress tests.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(length, dtype=float)
    series, ids = [], []

    # 1. AR(2) + daily/weekly seasonality
    e = rng.normal(size=length)
    ar = np.zeros(length)
    for i in range(2, length):
        ar[i] = 0.6 * ar[i - 1] - 0.2 * ar[i - 2] + e[i]
    series.append(10 + 3 * np.sin(2 * np.pi * t / 24) + 1.5 * np.sin(2 * np.pi * t / 168) + ar)
    ids.append("ar2_multiseasonal")

    # 2. random walk with drift
    series.append(np.cumsum(0.05 + rng.normal(size=length)))
    ids.append("rw_drift")

    # 3–6. structural breaks at 75 % of the series
    b = int(0.75 * length)
    base = 5 * np.sin(2 * np.pi * t / 24) + rng.normal(scale=1.0, size=length)
    series.append(base + np.where(t >= b, 8.0, 0.0))
    ids.append("break_mean")
    series.append(5 * np.sin(2 * np.pi * t / 24) + rng.normal(size=length) * np.where(t >= b, 4.0, 1.0))
    ids.append("break_variance")
    series.append(np.where(t >= b, 5 * np.sin(2 * np.pi * t / 12), 5 * np.sin(2 * np.pi * t / 24))
                  + rng.normal(size=length))
    ids.append("break_seasonality")
    series.append(base + np.where(t >= b, 0.05 * (t - b), 0.0))
    ids.append("break_trend")

    # 7. Lorenz x-component (chaotic)
    x, y, z, dt = 1.0, 1.0, 1.0, 0.01
    lorenz = np.empty(length)
    for i in range(length):
        for _ in range(5):
            dx, dy, dz = 10 * (y - x), x * (28 - z) - y, x * y - 8 / 3 * z
            x, y, z = x + dt * dx, y + dt * dy, z + dt * dz
        lorenz[i] = x
    series.append(lorenz)
    ids.append("lorenz_x")

    # 8. intermittent demand (Bernoulli × Poisson)
    series.append((rng.random(length) < 0.3) * rng.poisson(4, size=length).astype(float))
    ids.append("intermittent")

    index = pd.date_range("2030-01-01", periods=length, freq="h")  # deliberately future-dated
    return SeriesSet("synthetic_suite", np.stack(series), index, "h", 24, ids,
                     download_hint="generated (seeded); no external data")


def regime_break_location(length: int) -> int:
    return int(0.75 * length)


def beijing_air_quality(directory: str | Path, target: str = "PM2.5") -> SeriesSet:
    """UCI Beijing Multi-Site Air Quality (12 stations, hourly, 2013-03 – 2017-02).

    Twelve monitoring stations, mirroring the 12-station design of Zheng et al.
    (Water Research X 30, 2026). Weather columns are kept as past covariates per station.
    """
    files = sorted(Path(directory).glob("PRSA_Data_*_20130301-20170228.csv"))
    if not files:
        raise FileNotFoundError(f"no PRSA_Data_*.csv files in {directory}")
    values, past, ids = [], [], []
    index = None
    for f in files:
        df = pd.read_csv(f)
        stamps = pd.to_datetime(df[["year", "month", "day", "hour"]])
        df = df.set_index(stamps).asfreq("h")
        index = pd.DatetimeIndex(df.index) if index is None else index
        values.append(df[target].to_numpy(float))
        past.append(df[["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]].to_numpy(float).T)
        ids.append(str(df["station"].dropna().iloc[0]))
    return SeriesSet("beijing_air_quality", np.stack(values), index, "h", 24, ids, past_covariates=np.stack(past),
                     covariate_names={"past": ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]},
                     download_hint="https://doi.org/10.24432/C5RK5G")


# ------------------------------------------------------------- synthetic bank (E2, leakage-free)
def _stable_ar(rng: np.random.Generator, p: int) -> np.ndarray:
    """Random stationary AR(p) coefficients (roots drawn inside the unit circle)."""
    roots = rng.uniform(0.2, 0.95, p) * rng.choice([-1, 1], p)
    return -np.poly(roots)[1:]


def _arfima_noise(rng, n: int, d: float) -> np.ndarray:
    k = np.arange(1, n)
    w = np.r_[1.0, np.cumprod((k - 1 + d) / k)]  # MA(inf) weights of (1-B)^-d
    e = rng.normal(size=n)
    return np.convolve(e, w)[:n]


def _ode(f, x0, dt: float, n: int, sub: int = 5) -> np.ndarray:
    x = np.asarray(x0, float)
    out = np.empty((n, len(x)))
    for i in range(n):
        for _ in range(sub):  # RK2 (midpoint) with sub-steps
            k1 = f(x)
            x = x + dt * f(x + 0.5 * dt * k1)
        out[i] = x
    return out


def _family(name: str, rng: np.random.Generator, n: int) -> tuple[np.ndarray, int]:
    """One series of a synthetic family and its natural seasonal period (for MASE scaling)."""
    t = np.arange(n, dtype=float)
    if name == "arma_seasonal":
        a = _stable_ar(rng, int(rng.integers(1, 4)))
        e = rng.normal(size=n)
        x = np.zeros(n)
        for i in range(len(a), n):
            x[i] = a @ x[i - len(a):i][::-1] + e[i]
        m = int(rng.choice([12, 24, 48]))
        s = rng.uniform(0.5, 4) * np.sin(2 * np.pi * t / m + rng.uniform(0, 6.3)) \
            + rng.uniform(0, 2) * np.sin(2 * np.pi * t / (7 * m))
        return 10 + s + x, m
    if name == "ou_jumps":
        theta, sigma, mu = rng.uniform(0.01, 0.2), rng.uniform(0.3, 1.5), rng.uniform(-5, 5)
        jumps = (rng.random(n) < rng.uniform(0.002, 0.02)) * rng.normal(0, 8, n)
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = x[i - 1] + theta * (mu - x[i - 1]) + sigma * rng.normal() + jumps[i]
        return x, 1
    if name == "markov_switching":
        p_stay = rng.uniform(0.97, 0.999, 2)
        mus, sds, phi = rng.normal(0, 5, 2), rng.uniform(0.3, 3, 2), rng.uniform(0.3, 0.95)
        s, x = 0, np.zeros(n)
        for i in range(1, n):
            s = s if rng.random() < p_stay[s] else 1 - s
            x[i] = mus[s] + phi * (x[i - 1] - mus[s]) + sds[s] * rng.normal()
        return x, 1
    if name == "threshold_ar":
        c, a1, a2 = rng.normal(), rng.uniform(-0.9, 0.9), rng.uniform(-0.9, 0.9)
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = (a1 if x[i - 1] < c else a2) * x[i - 1] + (1.5 if x[i - 1] < c else -1.5) * 0.3 + rng.normal()
        return x, 1
    if name == "arfima":
        return _arfima_noise(rng, n, rng.uniform(0.1, 0.45)) + rng.normal(0, 3), 1
    if name == "garch_level":
        w, a, b = 1e-5, rng.uniform(0.03, 0.15), 0.0
        b = rng.uniform(0.7, 0.98 - a)
        r, s2 = np.zeros(n), np.full(n, w / (1 - a - b))
        for i in range(1, n):
            s2[i] = w + a * r[i - 1] ** 2 + b * s2[i - 1]
            r[i] = np.sqrt(s2[i]) * rng.standard_t(5) / np.sqrt(5 / 3)
        return 100 * np.exp(np.cumsum(r)), 1
    if name == "logistic_seasonal":
        k, r0, mid = rng.uniform(50, 500), rng.uniform(2, 10) / n, rng.uniform(0.2, 0.8) * n
        m = int(rng.choice([7, 12, 24]))
        level = k / (1 + np.exp(-r0 * (t - mid)))
        return level * (1 + rng.uniform(0.05, 0.3) * np.sin(2 * np.pi * t / m)) + rng.normal(0, 0.02 * k, n), m
    if name == "lorenz":
        s, r, b = 10 * rng.uniform(0.8, 1.2), 28 * rng.uniform(0.9, 1.3), 8 / 3
        out = _ode(lambda v: np.array([s * (v[1] - v[0]), v[0] * (r - v[2]) - v[1], v[0] * v[1] - b * v[2]]),
                   rng.normal(1, 0.5, 3), rng.uniform(0.005, 0.015), n)
        return out[:, int(rng.integers(0, 3))], 1
    if name == "rossler":
        a, b, c = 0.2, 0.2, rng.uniform(4.5, 7)
        out = _ode(lambda v: np.array([-v[1] - v[2], v[0] + a * v[1], b + v[2] * (v[0] - c)]),
                   rng.normal(1, 0.5, 3), rng.uniform(0.05, 0.15), n)
        return out[:, 0], 1
    if name == "mackey_glass":
        tau, beta, gamma, nn = int(rng.integers(17, 30)), 0.2, 0.1, 10
        x = np.full(n + tau, 1.2) + rng.normal(0, 0.01, n + tau)
        for i in range(tau, n + tau - 1):
            x[i + 1] = x[i] + beta * x[i - tau] / (1 + x[i - tau] ** nn) - gamma * x[i]
        return x[tau:], 1
    if name == "intermittent":
        p = rng.uniform(0.05, 0.5)
        return (rng.random(n) < p) * rng.gamma(rng.uniform(1, 5), rng.uniform(1, 10), n), 1
    if name == "poisson_seasonal":
        m = int(rng.choice([7, 24]))
        lam = np.exp(rng.uniform(0, 3) + rng.uniform(0.2, 1.0) * np.sin(2 * np.pi * t / m))
        return rng.poisson(lam).astype(float), m
    if name == "nonsinusoidal":
        m = int(rng.choice([12, 24, 36]))
        phase = (t + np.cumsum(rng.normal(0, 0.02, n))) % m / m
        shape = np.where(phase < rng.uniform(0.2, 0.6), 1.0, -1.0) if rng.random() < 0.5 else phase
        return rng.uniform(2, 10) * shape + 0.01 * rng.normal(size=n).cumsum() + rng.normal(0, 0.3, n), m
    if name == "structural_break":
        m, b = 24, int(rng.uniform(0.5, 0.9) * n)
        base = 5 * np.sin(2 * np.pi * t / m) + rng.normal(size=n)
        kind = rng.choice(["mean", "var", "season", "trend"])
        if kind == "mean":
            return base + np.where(t >= b, rng.uniform(4, 12) * rng.choice([-1, 1]), 0), m
        if kind == "var":
            return 5 * np.sin(2 * np.pi * t / m) + rng.normal(size=n) * np.where(t >= b, rng.uniform(2, 5), 1), m
        if kind == "season":
            return np.where(t >= b, 5 * np.sin(2 * np.pi * t / 12), 5 * np.sin(2 * np.pi * t / m)) \
                + rng.normal(size=n), m
        return base + np.where(t >= b, rng.uniform(0.01, 0.08) * (t - b), 0), m
    raise ValueError(name)


SYNTHETIC_FAMILIES = ["arma_seasonal", "ou_jumps", "markov_switching", "threshold_ar", "arfima", "garch_level",
                      "logistic_seasonal", "lorenz", "rossler", "mackey_glass", "intermittent", "poisson_seasonal",
                      "nonsinusoidal", "structural_break"]


def synthetic_bank(n_per_family: int = 10, length: int = 4096, seed: int = 0,
                   families: list[str] | None = None) -> list[SeriesSet]:
    """Seeded bank of leakage-free series: one :class:`SeriesSet` per family (shared seasonal period).

    Families cover nonlinear (threshold AR, Markov switching), long-memory (ARFIMA), jump-diffusion
    (OU + compound Poisson), heteroskedastic (GARCH price level), chaotic (Lorenz, Rössler,
    Mackey–Glass), count, intermittent, non-sinusoidal periodic and structural-break processes.
    Note: Chronos-family models were pre-trained on Gaussian-process synthetic data (KernelSynth);
    none of these generators is a GP kernel composition. Index is dated 2030 (future) on purpose.
    """
    out = []
    for f_i, fam in enumerate(families or SYNTHETIC_FAMILIES):
        rng = np.random.default_rng([seed, f_i])
        rows, seasons = [], []
        for _ in range(n_per_family):
            x, m = _family(fam, rng, length)
            rows.append(x)
            seasons.append(m)
        season = int(np.bincount(seasons).argmax())
        index = pd.date_range("2030-01-01", periods=length, freq="h")
        out.append(SeriesSet(f"syn_{fam}", np.stack(rows), index, "h", season,
                             [f"{fam}_{seed}_{k}" for k in range(n_per_family)],
                             download_hint="generated (seeded)"))
    return out
