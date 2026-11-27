# ci-proto experiments

Experiment suite for a journal article evaluating **ci-proto** (Continuous Intelligence,
[github.com/continuousintelligence/ci](https://github.com/continuousintelligence/ci)), a
167M-parameter zero-shot probabilistic time-series foundation model that adapts TiRex-2 with a
convolution-free bidirectional Mamba-3 time mixer.

Every experiment copies a protocol that has already been published in a Q1 journal or in a
standard TSFM benchmark, so the results can be placed next to published numbers. The full
rationale, literature table and analysis plan are in the project document
"ci-proto paper — experimental protocol grounded in Q1 literature".

## Publication protocol (Linux + CUDA, recommended)

The suite was upgraded for a Q1 submission (all current zero-shot TSFMs as comparators, larger
leakage-free and prospective evaluations, fixed statistical tests); see
[`docs/PROTOCOL_CHANGES.md`](docs/PROTOCOL_CHANGES.md) for every change and the bugs it fixed.

```sh
uv sync --extra chronos --extra stats --extra fm --extra bench --extra neural
(cd envs/moirai && uv sync)          # Moirai-2 needs torch 2.4: isolated env + subprocess worker
./run_paper.sh smoke                 # ~1-2 h: every model and code path on tiny settings
./run_paper.sh full                  # the paper run (days on an 8 GB GPU); resumable per experiment
```

Figures land in `results/figures/<profile>/`, tables in `results/tables/<profile>/`, the
statistical tests next to each experiment's results, and a Markdown digest in
`results/SUMMARY_<profile>.md`.

## Run everything with uv (one command, original laptop profile)

Windows (PowerShell, from this folder):

```powershell
powershell -ExecutionPolicy Bypass -File .\run_experiments.ps1            # laptop profile, CPU
powershell -ExecutionPolicy Bypass -File .\run_experiments.ps1 -RunProfile full   # full protocol, GPU
```

Linux / macOS / WSL: `./run_experiments.sh` (or `./run_experiments.sh full`).

The script runs `uv sync --extra chronos` (Python 3.12, ci-proto, Chronos-2), downloads the
public datasets (`scripts/get_data.py`: Ember daily prices, UCI household power, UCI Beijing
air quality, ETT, UCI Bike Sharing), runs the tests, runs E2, E4, E3, E6, E5, E7, E8 and E1,
then the statistical analysis, and writes `results/SUMMARY_<profile>.md`. Logs are in
`results/logs/`. Models whose package is missing are skipped with a `[skip]` line.

The **laptop** profile keeps every protocol but uses fewer origins, contexts and models
(ci-proto, Chronos-2 and baselines; see `profiles.laptop` in `configs/experiments.yaml`).
E9 and the intraday-price block of E3 need licensed data and are not run.

GPU: `uv sync` installs the CPU build of PyTorch on Windows. For an NVIDIA GPU, after the sync run
`uv pip install torch --index-url https://download.pytorch.org/whl/cu128 --reinstall`
and keep using `uv run --no-sync` (the script already does).

## Layout

```
ciexp/            evaluation core
  metrics.py      MAE, RMSE, sMAPE, MASE, WQL, SQL, CRPS, coverage, interval score, QLIKE, KGE …
  stats.py        Diebold–Mariano (HAC + HLN), Holm, BH, Model Confidence Set, skill score,
                  win rate, paired bootstrap CIs, Friedman–Nemenyi
  protocols.py    rolling / date-split origins, embargo, context grids, scarcity subsets, NaN injection
  models.py       ci-proto wrapper (joint | cohort | independent, tta.json ablations), Chronos-2,
                  TimesFM 2.5, TiRex-2, TiRex, Moirai 2.0, statistical and supervised baselines
  data.py         loaders for every dataset + a seeded leakage-free synthetic suite
  runner.py       rolling-origin backtest → forecast table → metrics table
  adapters.py     fev-bench driver and GluonTS predictor for GIFT-Eval
configs/experiments.yaml   the whole design, one section per experiment
experiments/e01…e09        one script per experiment
analysis/report.py         statistical tests and LaTeX tables from saved results
tests/test_core.py         25 tests (CPU, no foundation models needed)
scripts/get_data.py        downloads the public datasets into data/
run_experiments.ps1 / .sh  one-command uv runner (profiles: laptop, full)
analysis/summarize.py      collects all key tables into results/SUMMARY_<profile>.md
data/                      bundled 6-day bike-sharing example (CC BY 4.0); put raw datasets here
```

## Experiments

| ID | Script | Replicated protocol |
| --- | --- | --- |
| E1 | `e01_benchmarks.py` | fev-bench, GIFT-Eval, LTSF protocol of S-Mamba (Neurocomputing 619, 2025) |
| E2 | `e02_leakage.py` | fev-bench leakage policy; TiRex-2 leakage-free variants |
| E3 | `e03_epf.py` | Energy and AI 100914 (2026) and 100886 (2026) |
| E4 | `e04_covariates.py` | fev-bench covariate tasks; ci-proto walkthrough |
| E5 | `e05_context.py` | S-Mamba lookback study; Energy and AI context sweeps; Neurocomputing (2026) context inertia |
| E6 | `e06_scarcity.py` | Water Research X 30 (2026) 100469 |
| E7 | `e07_ablations.py` | component ablations (S-Mamba, RecMamba, FSMamba) adapted to tta.json |
| E8 | `e08_efficiency.py` | S-Mamba efficiency study; TiRex-2 per-patch cost |
| E9 | `e09_volatility.py` | Finance Research Letters 110 (2026) 110606 (optional) |

## Running

```sh
pip install -r requirements-core.txt
pytest -q                                   # 25 tests, ~10 s on CPU

# smoke runs: synthetic/bundled data and CPU baselines, no foundation models
python experiments/e02_leakage.py --quick
python experiments/e03_epf.py --quick

# full runs (GPU machine, datasets in data/)
pip install -r requirements-full.txt
python experiments/e03_epf.py --data-dir data
python analysis/report.py epf --results results/E3_electricity_prices/full --reference "ci-proto[joint]"
python analysis/report.py bench --results results/E2_leakage_control/full --metric SQL
```

`--models` overrides the model list (keys of the `models` section of the config), `--device`
forces CPU or CUDA for ci-proto, `--out` changes the results directory. Every run writes a
`manifest.json` with the config, package versions, GPU and ci-proto backend.

Before the paper runs, pin the ci-proto Hugging Face commit in `configs/experiments.yaml`
(`global.ci_proto.revision`).

## What has and has not been executed

* Executed here (CPU, Python 3.13): all of `ciexp/` except the foundation-model wrappers,
  the statistical baselines (Seasonal Naive, Naive, historical quantiles, LEAR, gradient-boosted
  quantiles, HAR-RV), the analysis script, and `--quick` runs of all nine experiments.
* Not executed here: the ci-proto, Chronos-2, TimesFM 2.5, TiRex-2, TiRex and Moirai wrappers,
  `statsforecast` / `neuralforecast` baselines and the fev / GIFT-Eval adapters. Their calls follow
  each package's README; check them once on the GPU machine (a single E2 run with each model is
  enough) before launching the full suite.

## Data licences

The bundled `data/bike_sharing.csv` is from the ci-proto package (Fanaee-T, 2013, CC BY 4.0;
see `data/NOTICE_bike_sharing.txt`). Other raw data are not redistributed; see the
"Datasets and access" section of the protocol document for sources and terms.
