"""Official GIFT-Eval splits and metrics, run in an isolated env (gift-eval pins numpy 1.26 / gluonts 0.15).

    python worker.py configs                         -> JSON list of the 97 (dataset, term) configurations
    python worker.py export   <ds_name> <term> <out.pkl>
    python worker.py evaluate <ds_name> <term> <preds.npz> <model_name> <out.json>

Follows the official notebooks (e.g. notebooks/chronos-2.ipynb): ``Dataset(name, term, to_univariate=False)``,
``evaluate_forecasts(..., axis=None, mask_invalid_label=True, allow_nan_forecast=False,
seasonality=get_seasonality(freq))`` with the same 11 metrics; the configuration naming and the
``dataset_properties.json`` domain / variate counts are taken from the repository. The forecasts are
produced in the main project env (``experiments/e10_gifteval.py``) and passed back as quantile arrays.
GIFT_EVAL must point at the downloaded Salesforce/GiftEval folder.
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SHORT = ("m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly electricity/15T electricity/H electricity/D "
         "electricity/W solar/10T solar/H solar/D solar/W hospital covid_deaths us_births/D us_births/M us_births/W "
         "saugeenday/D saugeenday/M saugeenday/W temperature_rain_with_missing kdd_cup_2018_with_missing/H "
         "kdd_cup_2018_with_missing/D car_parts_with_missing restaurant hierarchical_sales/D hierarchical_sales/W "
         "LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D SZ_TAXI/15T SZ_TAXI/H M_DENSE/H M_DENSE/D ett1/15T ett1/H "
         "ett1/D ett1/W ett2/15T ett2/H ett2/D ett2/W jena_weather/10T jena_weather/H jena_weather/D "
         "bitbrains_fast_storage/5T bitbrains_fast_storage/H bitbrains_rnd/5T bitbrains_rnd/H bizitobs_application "
         "bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H").split()
MED_LONG = ("electricity/15T electricity/H solar/10T solar/H kdd_cup_2018_with_missing/H LOOP_SEATTLE/5T "
            "LOOP_SEATTLE/H SZ_TAXI/15T M_DENSE/H ett1/15T ett1/H ett2/15T ett2/H jena_weather/10T jena_weather/H "
            "bitbrains_fast_storage/5T bitbrains_rnd/5T bizitobs_application bizitobs_service bizitobs_l2c/5T "
            "bizitobs_l2c/H").split()
PRETTY = {"saugeenday": "saugeen", "temperature_rain_with_missing": "temperature_rain",
          "kdd_cup_2018_with_missing": "kdd_cup_2018", "car_parts_with_missing": "car_parts"}
LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MAX_CONTEXT = 32768  # stored context per window (every comparator truncates far below this)


def config_name(ds_name: str, term: str) -> tuple[str, str, int]:
    props = json.load(open(HERE / "dataset_properties.json"))
    if "/" in ds_name:
        key, freq = ds_name.split("/")
        key = PRETTY.get(key.lower(), key.lower())
    else:
        key = PRETTY.get(ds_name.lower(), ds_name.lower())
        freq = props[key]["frequency"]
    return f"{key}/{freq}/{term}", props[key]["domain"], props[key]["num_variates"]


def configs() -> list[dict]:
    out = []
    for ds in dict.fromkeys(SHORT + MED_LONG):  # official order: union of both lists
        for term in ("short", "medium", "long"):
            if term != "short" and ds not in MED_LONG:
                continue
            name, domain, nvar = config_name(ds, term)
            out.append({"ds_name": ds, "term": term, "config": name, "domain": domain, "num_variates": nvar})
    return out


def load(ds_name: str, term: str):
    from gift_eval.data import Dataset

    return Dataset(name=ds_name, term=term, to_univariate=False)


def export(ds_name: str, term: str, path: str) -> None:
    ds = load(ds_name, term)
    entries = []
    for e in ds.test_data.input:
        target = np.asarray(e["target"], dtype=np.float32)
        entries.append({"target": target[..., -MAX_CONTEXT:], "full_length": int(target.shape[-1]),
                        "item_id": str(e.get("item_id"))})
    from gluonts.time_feature import get_seasonality

    meta = {"prediction_length": int(ds.prediction_length), "freq": ds.freq, "windows": int(ds.test_data.windows),
            "season": int(get_seasonality(ds.freq)),
            "target_dim": int(ds.target_dim), "entries": entries}
    with open(path, "wb") as fh:
        pickle.dump(meta, fh, protocol=4)
    print(json.dumps({"n_entries": len(entries), "prediction_length": meta["prediction_length"],
                      "freq": meta["freq"], "windows": meta["windows"], "target_dim": meta["target_dim"]}))


def evaluate(ds_name: str, term: str, preds_path: str, model_name: str, out_path: str) -> None:
    from gluonts.ev.metrics import (MAE, MAPE, MASE, MSE, MSIS, ND, NRMSE, RMSE, SMAPE,
                                    MeanWeightedSumQuantileLoss)
    from gluonts.model import evaluate_forecasts
    from gluonts.model.forecast import QuantileForecast
    from gluonts.time_feature import get_seasonality

    metrics = [MSE(forecast_type="mean"), MSE(forecast_type=0.5), MAE(), MASE(), MAPE(), SMAPE(), MSIS(), RMSE(),
               NRMSE(), ND(), MeanWeightedSumQuantileLoss(quantile_levels=LEVELS)]
    ds = load(ds_name, term)
    q = np.load(preds_path)["quantiles"]  # [N, Q, H] (univariate) or [N, Q, H, V]
    keys = [str(lv) for lv in LEVELS] + ["mean"]
    forecasts = []
    for i, e in enumerate(ds.test_data.input):
        arr = q[i]
        arr = np.concatenate([arr, arr[4:5]], axis=0)  # 'mean' := median (point forecast)
        start = e["start"] + np.asarray(e["target"]).shape[-1]
        forecasts.append(QuantileForecast(forecast_arrays=arr, start_date=start, forecast_keys=keys,
                                          item_id=e.get("item_id")))
    res = evaluate_forecasts(forecasts, test_data=ds.test_data, metrics=metrics, batch_size=1024, axis=None,
                             mask_invalid_label=True, allow_nan_forecast=False,
                             seasonality=get_seasonality(ds.freq)).reset_index(drop=True).to_dict(orient="records")
    name, domain, nvar = config_name(ds_name, term)
    row = {"dataset": name, "model": model_name, **{f"eval_metrics/{k}": v for k, v in res[0].items()},
           "domain": domain, "num_variates": nvar}
    Path(out_path).write_text(json.dumps(row, default=float))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "configs":
        print(json.dumps(configs()))
    elif cmd == "export":
        export(*sys.argv[2:5])
    elif cmd == "evaluate":
        evaluate(*sys.argv[2:7])
    else:
        raise SystemExit(f"unknown command {cmd}")
