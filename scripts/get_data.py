"""Download the public datasets used by the experiments into ``data/``.

Skips files that already exist. Every source is public (CC BY 4.0); see README.
Datasets that need a licence (EPEX/N2EX/NEMS intraday prices, equities) are not downloaded.

    uv run python scripts/get_data.py            # everything
    uv run python scripts/get_data.py ember ett  # a subset
"""

from __future__ import annotations

import io
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

ETT = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{}.csv"
SOURCES = {
    "ember": "https://files.ember-energy.org/public-downloads/price/outputs/european_wholesale_electricity_price_data_daily.csv",
    "ember_hourly": "https://files.ember-energy.org/public-downloads/price/outputs/european_wholesale_electricity_price_data_hourly.zip",
    "household": "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip",
    "beijing": "https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip",
}


def fetch(url: str, retries: int = 3) -> bytes:
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url)  # default Python-urllib agent (FRED stalls custom agents)
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001
            print(f"  attempt {attempt} failed: {exc}")
            time.sleep(3 * attempt)
    raise RuntimeError(f"could not download {url}")


def extract_matching(blob: bytes, predicate, dest: Path) -> int:
    """Extract files whose basename satisfies ``predicate``, descending into nested zips."""
    count = 0
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for info in zf.infolist():
            name = Path(info.filename).name
            if info.filename.endswith(".zip"):
                count += extract_matching(zf.read(info), predicate, dest)
            elif name and predicate(name):
                dest.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(dest / name, "wb") as out:
                    shutil.copyfileobj(src, out)
                count += 1
    return count


def get_ember() -> None:
    target = DATA / "ember_daily.csv"
    if target.exists() and "--refresh" in sys.argv:  # new days for the prospective (post-release) window
        target.rename(target.with_suffix(".previous.csv"))
    if target.exists():
        print("ember: present")
        return
    print("ember: downloading daily prices (Ember, CC BY 4.0)")
    target.write_bytes(fetch(SOURCES["ember"]))


def get_household() -> None:
    target = DATA / "household_power_consumption.txt"
    if target.exists():
        print("household power: present")
        return
    print("household power: downloading UCI 235 (~20 MB zip)")
    n = extract_matching(fetch(SOURCES["household"]), lambda n: n == "household_power_consumption.txt", DATA)
    print(f"  extracted {n} file")


def get_beijing() -> None:
    dest = DATA / "beijing_air_quality"
    if dest.exists() and any(dest.glob("PRSA_Data_*.csv")):
        print("beijing air quality: present")
        return
    print("beijing air quality: downloading UCI 501")
    n = extract_matching(fetch(SOURCES["beijing"]), lambda n: n.startswith("PRSA_Data_") and n.endswith(".csv"), dest)
    print(f"  extracted {n} station files")


def get_ett() -> None:
    dest = DATA / "ltsf"
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("ETTh1", "ETTh2", "ETTm1", "ETTm2"):
        target = dest / f"{name}.csv"
        if target.exists():
            continue
        print(f"ltsf: downloading {name}")
        target.write_bytes(fetch(ETT.format(name)))
    print("ltsf: ETT files present (Electricity/Weather/Traffic/Exchange/Solar: scripts/get_data.py ltsf_large)")


TSLIB = "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/{}"
SOLAR = "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/solar-energy/solar_AL.txt.gz"


def get_ltsf_large() -> None:
    """Electricity, Weather, Traffic and Exchange from the Time-Series-Library release (thuml, the files
    used by the LTSF papers), and Solar-Energy (solar_AL, LSTNet repository; 137 plants, 10-min, 2006),
    rewritten with a date column so every LTSF file has the same layout."""
    import gzip

    dest = DATA / "ltsf"
    dest.mkdir(parents=True, exist_ok=True)
    for sub, fname in (("electricity", "electricity.csv"), ("weather", "weather.csv"),
                       ("traffic", "traffic.csv"), ("exchange_rate", "exchange_rate.csv")):
        target = dest / fname
        if target.exists():
            continue
        print(f"ltsf: downloading {fname}")
        target.write_bytes(fetch(TSLIB.format(f"{sub}/{fname}")))
    target = dest / "solar_AL.csv"
    if not target.exists():
        import pandas as pd

        print("ltsf: downloading solar_AL")
        raw = pd.read_csv(io.BytesIO(gzip.decompress(fetch(SOLAR))), header=None)
        raw.columns = [str(c) for c in raw.columns]
        raw.insert(0, "date", pd.date_range("2006-01-01", periods=len(raw), freq="10min"))
        raw.to_csv(target, index=False)
    print("ltsf: Electricity/Weather/Traffic/Exchange/Solar present")


def get_bike() -> None:
    target = DATA / "bike_sharing_openml42712.csv"
    if target.exists():
        print("bike sharing: present")
        return
    print("bike sharing: downloading OpenML 42712 via scikit-learn")
    from sklearn.datasets import fetch_openml

    frame = fetch_openml(data_id=42712, as_frame=True, parser="liac-arff").frame
    frame.to_csv(target, index=False)


def get_ember_hourly() -> None:
    dest = DATA / "ember_hourly"
    wanted = ["Germany.csv", "United Kingdom.csv", "Spain.csv"]
    if all((dest / w).exists() for w in wanted):
        print("ember hourly: present")
        return
    print("ember hourly: downloading (Ember, CC BY 4.0, ~43 MB zip)")
    n = extract_matching(fetch(SOURCES["ember_hourly"]), lambda n: n in wanted, dest)
    print(f"  extracted {n} market files")


FRED_SERIES = ["SP500", "NASDAQCOM", "NASDAQ100", "DJIA",                     # US equity indices
               "DEXUSEU", "DEXJPUS", "DEXUSUK", "DEXCHUS", "DEXCAUS", "DEXSZUS", "DEXUSAL", "DEXMXUS",
               "DEXKOUS", "DEXINUS",                                             # FX (H.10)
               "DCOILWTICO", "DCOILBRENTEU", "DHHNGSP", "CBBTCUSD"]              # commodities, bitcoin


def get_fred() -> None:
    """Daily closes from FRED for the volatility negative control (E9): ``date,ticker,close``.

    Public substitute for the licensed equity panel of Finance Research Letters 110 (2026) 110606.
    Index series (S&P, Dow Jones, Nasdaq) are copyrighted by their owners: use, don't redistribute.
    """
    import csv

    target = DATA / "equities.csv"
    if target.exists():
        print("fred daily closes: present")
        return
    rows = []
    for sid in FRED_SERIES:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd=2015-01-01&coed=2025-12-31"
        text = fetch(url).decode("utf-8").splitlines()
        for line in text[1:]:
            date, value = line.split(",")
            if value not in ("", "."):
                rows.append((date, sid, float(value)))
        print(f"  fred {sid}: {len(text) - 1} days")
        time.sleep(0.5)
    with open(target, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "ticker", "close"])
        w.writerows(rows)


TASKS = {"ember": get_ember, "fred": get_fred, "ember_hourly": get_ember_hourly, "household": get_household, "beijing": get_beijing, "ett": get_ett, "ltsf_large": get_ltsf_large, "bike": get_bike}


def main() -> None:
    DATA.mkdir(exist_ok=True)
    wanted = [a for a in sys.argv[1:] if not a.startswith("--")] or list(TASKS)
    failed = []
    for key in wanted:
        try:
            TASKS[key]()
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] {key}: {exc}")
            failed.append(key)
    if failed:
        print(f"\nNot downloaded: {', '.join(failed)}. The matching experiments will skip or fail for them.")
        sys.exit(1)


if __name__ == "__main__":
    main()
