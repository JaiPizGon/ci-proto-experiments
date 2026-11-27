"""Shared CLI, configuration and model construction for the experiment scripts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ciexp import models as M  # noqa: E402

QUICK_MODELS = ["seasonal-naive", "naive", "historical-mean"]

# Python module each model kind needs; models whose package is missing are skipped with a warning.
REQUIRED_MODULE = {"ci-proto": "continuousintelligence", "chronos-2": "chronos", "timesfm-2.5": "timesfm",
                   "timesfm-3": "timesfm3", "tirex-2": "tirex2", "tirex": "tirex", "statsforecast": "statsforecast",
                   "neuralforecast": "neuralforecast"}
# Pretrained (zero-shot) kinds: instances are cached and re-used across conditions of a run.
TSFM_KINDS = {"ci-proto", "chronos-2", "timesfm-2.5", "timesfm-3", "tirex-2", "tirex", "moirai-2"}
_MODEL_CACHE: dict[str, M.BaseForecaster] = {}
CACHE_LIMIT = [4]  # resident pretrained models (8 GB GPU: weights + activations); long-context studies use 1


def set_cache_limit(n: int) -> None:
    CACHE_LIMIT[0] = n
    while len(_MODEL_CACHE) > n:
        _MODEL_CACHE.pop(next(iter(_MODEL_CACHE)))
    _free_gpu()
FORECAST_REPOS = {"ContinuousIntelligence/ci-proto", "amazon/chronos-2", "google/timesfm-2.5-200m-pytorch",
                  "google/timesfm-3.0-pytorch", "NX-AI/TiRex", "NX-AI/TiRex-2", "Salesforce/moirai-2.0-R-small"}
NEURAL_KEYS = ("nbeats", "lstm", "patchtst", "itransformer", "dlinear", "timesnet")


def parse(description: str) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=str(ROOT / "configs" / "experiments.yaml"))
    p.add_argument("--models", nargs="*", help="override the model list (keys of the 'models' section)")
    p.add_argument("--data-dir", default=str(ROOT / "data"), help="directory with the raw public datasets")
    p.add_argument("--out", default=None, help="results directory (default: global.results_dir)")
    p.add_argument("--quick", action="store_true",
                   help="smoke run: synthetic/bundled data and CPU baselines only (no foundation models)")
    p.add_argument("--device", default=None, help="override ci-proto device (cpu|cuda|auto)")
    p.add_argument("--only", nargs="*", default=None,
                   help="run only these parts of an experiment (e.g. E2: A synthetic, B prospective, C audit)")
    p.add_argument("--profile", default="full", choices=["full", "laptop", "smoke"],
                   help="'laptop' = CPU-sized settings from the config's profiles.laptop section")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def out_dir(cfg: dict, args: argparse.Namespace, experiment: str) -> Path:
    """Create the results folder and remember the experiment so profile knobs can be looked up."""
    args._exp, args._cfg = experiment, cfg
    base = Path(args.out or ROOT / cfg["global"]["results_dir"])
    path = base / experiment / ("quick" if args.quick else args.profile)
    path.mkdir(parents=True, exist_ok=True)
    return path


def knob(args: argparse.Namespace, key: str, default):
    """Value of ``key`` for the current experiment under the active profile, else ``default``."""
    if getattr(args, "quick", False) or getattr(args, "profile", "full") == "full":
        return default
    profile = (getattr(args, "_cfg", {}) or {}).get("profiles", {}).get(args.profile, {})
    return profile.get(getattr(args, "_exp", ""), {}).get(key, default)


def _kind(key: str, cfg: dict) -> str:
    zoo = cfg.get("models", {})
    if key in zoo:
        return zoo[key]["kind"]
    return "neuralforecast" if key in NEURAL_KEYS else key


def available(key: str, cfg: dict) -> bool:
    kind = _kind(key, cfg)
    if kind == "moirai-2":  # isolated environment (uni2ts pins torch 2.4)
        return (ROOT / "envs" / "moirai" / ".venv").exists() or importlib.util.find_spec("uni2ts") is not None
    module = REQUIRED_MODULE.get(kind)
    return module is None or importlib.util.find_spec(module) is not None


def model_keys(cfg_exp: dict, args: argparse.Namespace, default: list[str] | None = None) -> list[str]:
    if args.models:
        keys = list(args.models)
    elif args.quick:
        return QUICK_MODELS
    else:
        keys = knob(args, "models", list(cfg_exp.get("models", default or [])))
    cfg = getattr(args, "_cfg", {}) or {}
    missing = [k for k in keys if not available(k, cfg)]
    for k in missing:
        print(f"[skip] model {k!r}: its Python package is not installed")
    return [k for k in keys if k not in missing]


def make_model(key: str, cfg: dict, *, season: int, horizon: int | None = None, lags: list[int] | None = None,
               args: argparse.Namespace | None = None, **override) -> M.BaseForecaster:
    """Build a forecaster from the shared zoo, filling dataset-dependent arguments."""
    zoo = cfg["models"]
    if key in zoo:
        spec = dict(zoo[key])
    elif key in ("lear", "gbm-quantile"):
        spec = {"kind": key, "lags": lags or [1, 2, 3, season, 2 * season], "horizon": horizon or 1,
                "per_series": key == "lear"}  # LEAR is a local (per-market) model; GBM is global
    elif key == "lear-recal":  # LEAR recalibrated at every origin on a rolling window (Lago et al., 2021)
        spec = {"kind": "lear", "lags": lags or [1, 2, 3, season, 2 * season], "horizon": horizon or 1,
                "calibration_window": override.pop("calibration_window", 4 * 365), "refit_every": 1,
                "name": "LEAR-recal", "per_series": True}
    elif key in ("nbeats", "lstm", "patchtst", "itransformer", "dlinear", "timesnet"):
        names = {"nbeats": "NBEATS", "lstm": "LSTM", "patchtst": "PatchTST", "itransformer": "iTransformer",
                 "dlinear": "DLinear", "timesnet": "TimesNet"}
        spec = {"kind": "neuralforecast", "model": names[key], "horizon": horizon or 1,
                "input_size": override.pop("input_size", max(2 * (horizon or 1), season * 7)), "name": names[key]}
    elif key == "har-rv":
        spec = {"kind": "har-rv"}
    else:
        raise KeyError(f"model {key!r} not in the zoo")
    if spec.get("season", "absent") is None:
        spec["season"] = season
    if spec["kind"] == "seasonal-naive":
        spec.setdefault("season", season)
    if spec["kind"] == "ci-proto":
        g = cfg["global"]["ci_proto"]
        spec.setdefault("source", g["source"])
        spec.setdefault("revision", g.get("revision"))
        spec.setdefault("device", (args.device if args and args.device else g.get("device", "auto")))
    spec.update(override)
    if spec["kind"] in TSFM_KINDS and not spec.get("tta_overrides"):
        key_ = json.dumps(spec, sort_keys=True, default=str)
        if key_ not in _MODEL_CACHE:
            if len(_MODEL_CACHE) >= CACHE_LIMIT[0]:  # bound GPU memory (E5/E8 keep one model resident)
                _MODEL_CACHE.pop(next(iter(_MODEL_CACHE)))
                _free_gpu()
            _MODEL_CACHE[key_] = M.build(spec)
        return _MODEL_CACHE[key_]
    return M.build(spec)


def _free_gpu() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def model_revisions() -> dict[str, str]:
    """Hugging Face snapshot hashes of every cached pretrained model (pin these in the paper)."""
    out = {}
    try:
        from huggingface_hub import scan_cache_dir

        for repo in scan_cache_dir().repos:
            if repo.repo_type == "model" and repo.repo_id in FORECAST_REPOS:
                out[repo.repo_id] = sorted(r.commit_hash for r in repo.revisions)[-1]
    except Exception:  # noqa: BLE001
        pass
    return out


def environment_report() -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        info["torch"] = None
    try:
        import continuousintelligence as ci

        info["continuousintelligence"] = getattr(ci, "__version__", "installed")
    except ImportError:
        info["continuousintelligence"] = None
    return info


def write_manifest(path: Path, cfg_exp: dict, args: argparse.Namespace, extra: dict | None = None) -> None:
    manifest = {"config": cfg_exp, "args": {k: v for k, v in vars(args).items() if not k.startswith("_")},
                "environment": environment_report(), "model_revisions": model_revisions()}
    if extra:
        manifest.update(extra)
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


def checkpointed(path: Path, compute):
    """Return the table cached at ``path`` (parquet or csv.gz) or compute it, save it and return it.

    Long experiments wrap each (model, block) unit in this so a crash or a killed session resumes
    where it stopped. Delete the ``_ckpt`` folder of an experiment to force a full re-run.
    """
    from ciexp import runner as R

    raw = Path(path)
    # dots in model keys ("timesfm-2.5") must not be read as a file suffix: R.save/R.load use
    # with_suffix, which turned "partB_timesfm-2.5_2026-11-14" into "partB_timesfm-2" (date lost)
    path = raw.with_name(raw.name.replace(".", "_"))
    legacy = raw.with_suffix("")  # name written before this fix
    dropped = raw.name[len(legacy.name):]
    candidates = [path] + ([legacy] if legacy != raw and "_" not in dropped else [])  # never reuse a lost date
    for base in candidates:
        for cand in (base.with_suffix(".parquet"), base.with_suffix(".csv.gz")):
            if cand.exists():
                print(f"  [resume] {cand.name}")
                return R.load(base)
    df = compute()
    if df is not None and len(df):
        R.save(df, path)
    return df


def run_isolated_per_model(args, keys: list[str]) -> bool:
    """Re-launch this experiment once per model in a fresh process (``--models <key>``).

    A CUDA out-of-memory raised inside a kernel (e.g. a 2M-point context) poisons the CUDA context of
    the whole process; isolating models keeps one model's failure from turning every later model's
    results into errors. Returns True when the per-model runs were launched (the caller then only
    aggregates the checkpoints); False when this process is already a single-model run.
    """
    import subprocess

    if args.models or len(keys) <= 1:
        return False
    argv = [a for a in sys.argv[1:]]
    for key in keys:
        print(f"=== isolated run: {key}", flush=True)
        code = subprocess.call([sys.executable, sys.argv[0], *argv, "--models", key])
        if code:
            print(f"[warn] isolated run for {key} exited with {code}", flush=True)
    return True
