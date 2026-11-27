# Protocol upgrades for a Q1 submission (2026-11-15)

This file records what was changed relative to the original suite (v0.2.0), why, and what
was fixed along the way. Everything below is implemented and is run by `./run_paper.sh full`.

## 1. Bugs fixed (these would have invalidated published numbers)

| Where | Problem | Fix |
|---|---|---|
| `ciexp/data.py::ember_daily_prices` | Kept all 34 countries in the current Ember file, including Albania, Moldova and Kosovo with **no 2015–2025 data** and 8 partial markets; NaN series inflated failure rates and distorted pooled metrics. | Markets need ≥ 99 % coverage in the period → exactly the **23 markets** of Energy and AI 100914. |
| `analysis/report.py` DM test | Stacked every (origin, step) error of an h-step forecast into one series, which is not a valid DM loss series (steps of one forecast are not consecutive time points). | Loss per origin = mean over the h steps; HAC Bartlett lag h−1 for the overlapping daily origins; HLN correction kept. |
| `analysis/report.py` MCS | Moving-block bootstrap over rows sorted by (series, cutoff, step): blocks crossed series boundaries. | MCS on the time series of per-origin losses (averaged over markets and steps). |
| `ciexp/adapters.py` fev-bench | Predictions passed as dict-of-columns (fev expects one dict per item); categorical covariates crashed; one call per item. | Rewritten against fev 0.10: per-item dicts, integer-coded categoricals (flagged for ci-proto), batched univariate items, resumable per-task CSV. |
| `ciexp/models.py` neuralforecast | MQLoss median column dropped → **8 quantiles instead of 9** (misaligned levels in every metric). Interpretable N-BEATS crashes at h = 1. | Columns parsed by level; generic N-BEATS (identity stacks); seeded. |
| `ciexp/models.py` TimesFM-2.5 | `max_context=16384` + `max_horizon=1024` violates the model's 16,384 limit (crash); every input padded to the max context and to a batch of 1 → ~100× slower. | Lazily compiled per (context bucket, horizon, batch) with cached decoders. |
| `ciexp/models.py` LEAR | One pooled model for all markets (LEAR is a per-market model in Lago et al., 2021); O(H²) design construction. | `PerSeries` wrapper (one LEAR per market), design built once; added daily-recalibrated **LEAR-recal** (4-year window). |
| `ciexp/runner.py` | A failing `fit` aborted the whole experiment. | Recorded as an error status with NaN forecasts (fev rule: impute with Seasonal Naive). |
| `run_experiments.sh` | `uv sync --extra chronos` uninstalls every other extra. | `run_paper.sh` never re-syncs. |
| `experiments/e09_volatility.py` | ci-proto got a 2048-day context, other TSFMs the full history. | Same window for every TSFM. |
| `experiments/e08_efficiency.py` | ci-proto timed in cohort mode (joint chunks) vs independent series for the others. | Every model forecasts the batch as independent series. |

## 2. Comparators (the main gap for a TSFM paper)

The laptop profile compared ci-proto with Chronos-2 only. Now every experiment runs the current
zero-shot state of the art: **Chronos-2, TimesFM-3, TimesFM-2.5, TiRex-2, TiRex, Moirai-2.0**
(Moirai in an isolated env, `envs/moirai`, because uni2ts pins torch 2.4 / numpy 1.26 which
break ci-proto's Mamba-3 kernels), plus statistical (AutoETS, AutoTheta, Seasonal Naive, Naive),
econometric (LEAR, LEAR-recal, HAR-RV) and supervised deep baselines (N-BEATS, PatchTST, LSTM,
LightGBM quantiles) where the replicated protocol calls for them. All model revisions are pinned
in every `manifest.json` (ci-proto Hub commit `ee30313…`).

ci-proto is evaluated in three documented modes:
`ci-proto` (univariate per series; batched, identical to per-row calls up to bfloat16 rounding —
max |Δ| 0.006 EUR/MWh in fp32), `ci-proto[cohort]` (default cohort policy: chunked joint
forecasting = cross-learning) and `ci-proto[joint]` (multivariate).

## 3. Per-experiment upgrades

* **E1** — fev-bench is re-run **independently** for ci-proto and all six TSFMs with our harness
  and compared task-by-task with the official leaderboard files (ci-proto's own submission is
  among them). Reproduction fidelity (ratio ours/official, Spearman) validates the harness on the
  comparators and tests ci-proto's self-reported numbers. LTSF: all TSFMs, joint and
  channel-independent, lookbacks 512/2048/full.
* **E2** — leakage control was 8 hand-made series. Now 8-process suite + **14 synthetic families
  × 10 series × 5 seeds** (threshold AR, Markov switching, ARFIMA, OU-jumps, GARCH price levels,
  Lorenz, Rössler, Mackey–Glass, count, intermittent, non-sinusoidal periodic, breaks, …; none
  is a GP-kernel composition, unlike KernelSynth used to pre-train Chronos), aggregated with
  the fev skill score, bootstrap CIs and Friedman–Nemenyi. **Part B: temporal hold-out** on
  Ember 2026 prices (after the documented cut-offs of all comparators) and the post-release
  window (targets after 2026-11-12), re-runnable as data accumulate.
* **E3** — daily origins (730 per market and horizon instead of 52), context selection on
  validation 2023, 17 models incl. LEAR-recal; Block B on public Ember hourly prices
  (DE / GB / ES substitute for the licensed EPEX-DE / N2EX-UK / NEMS-SG).
* **E4** — second dataset (Beijing, 12 stations, per-station weather), all TSFMs, paired
  Wilcoxon + Holm against C0.
* **E5** — regime breaks: 10 seeds × 9 post-break offsets × 4 break types (was 1 × 5 × 4);
  native-context limits flagged for every model.
* **E6** — all TSFMs; N-BEATS added; neural models seeded.
* **E7** — five domains (synthetic, Ember, ETTh1, Beijing, bike) instead of two; paired tests of
  every ablation against the default with Holm correction.
* **E8** — all TSFMs, both ci-proto backends (fused Mamba-3 vs portable torch), CPU and GPU.
* **E9** — public FRED panel (4 equity indices, 10 FX rates, WTI/Brent/Henry Hub) instead of
  synthetic GARCH only; DM vs HAR-RV per asset with Holm, and MCS.

## 4. Outputs for the manuscript

`analysis/paper.py` writes figures (PDF + PNG) to `results/figures/<profile>/` and tables to
`results/tables/<profile>/`: fev reproduction plot and leaderboard, E2 skill heatmap and
critical-difference diagram, E3 skill dot-plot and reliability diagrams, E4 covariate effects
with significance marks, E5 context-scaling and regime-break curves, E6 scarcity curves, E8
latency curves.

## 5. Remaining limitations to state in the paper

* Licensed intraday markets (EPEX, N2EX, NEMS) and single-stock equities are replaced by public
  substitutes; GEFCom2014 (registration) is replaced by Beijing air quality in E4.
* ci-proto's training corpus is undisclosed; E2 bounds but cannot exclude benchmark leakage on
  public datasets (ETT, bike sharing, Beijing, household power are all widely mirrored).
* The post-release window has only a few days of targets at the time of writing; re-run E2
  Part B before submission.
* LTSF uses every 24th test window (stride 24) to keep the 8-model × 3-lookback grid tractable.

## 6. Additions of 2026-11-16

* **E10 — GIFT-Eval** (`experiments/e10_gifteval.py`): all 97 configurations with the *official*
  `gift-eval` splits, windows and gluonts metrics, run in the isolated `envs/gifteval` env (the package
  pins numpy 1.26 / gluonts 0.15); our wrappers only produce the quantiles. ci-proto has no official
  GIFT-Eval entry, so this is its first GIFT-Eval result; Chronos-2, TiRex-2 and Seasonal Naive are re-run
  to validate the harness against their official files (smoke check: our Seasonal Naive reproduces the
  official MASE exactly, geometric mean 1.000). Leaderboard = geometric mean of MASE and CRPS relative to
  Seasonal Naive + average CRPS rank, with all official results; breakdowns by term and domain.
* **Probabilistic tests** (`analysis/probabilistic.py`, `ciexp/stats.py`): which model predicts the
  *distribution* best — Diebold–Mariano on CRPS per series (Holm) and as a pairwise matrix, MCS on CRPS —
  and whether distributions are calibrated — Kupiec (1995) per quantile and for the 80 % interval,
  Christoffersen (1998) independence / conditional coverage, discretised PIT chi2 — on E3 (daily and
  hourly), E2 prospective and E9.
* **Inference cost** (`ciexp/profiling.py`): every E5/E8 cell records wall time, peak GPU memory
  (allocated and reserved; Moirai-2 reports from its worker process), peak host RSS and parameter count;
  E8 measures each model at every context inside its native window plus the native maximum, one model
  per process (a CUDA OOM cannot contaminate other models); `results/tables/<profile>/inference_cost.csv`
  joins E8 with total fev-bench and GIFT-Eval inference time.
* **Robust long runs**: per-model checkpoints (`<experiment>/<profile>/_ckpt/`) in E2–E8 so a crash or a
  closed session resumes where it stopped; model loops outermost (one load per model); the runner is
  started detached (`setsid`).
* **Prospective refresh**: `scripts/refresh_prospective.sh` re-downloads Ember, re-runs E2 Part B only
  (`--only B`; checkpoints keyed by the data end date) and the probabilistic tests. Run it shortly before
  submission so the post-release window covers as many days as possible.
* **Contamination audit (E2 Part C)**: error growth with lead time per model x public dataset; first
  finding: TimesFM-2.5 keeps a flat PM2.5 error on Beijing (RMSE 16.5 -> 17.6, lead 1 -> 48 h) while
  ci-proto (20.4 -> 67.9) and Chronos-2 (21.3 -> 101.0) grow as expected — memorisation of the test period.

## 7. Comparable inference cost (E8, 2026-11-16)

Each library ships its own inference stack, so a single latency per model measures software as much as
architecture. E8 now measures every model in explicit **variants** (`VARIANTS` in
`experiments/e08_efficiency.py`), and every row records `precision`, `backend`, `compiled`, `process`,
`param_dtype`, `is_default`, `is_controlled`:

| Model | default (out of the box) | controlled (eager PyTorch, fp32) | extra |
|---|---|---|---|
| ci-proto | Mamba-3 fused kernels, bf16 autocast | torch backend, `bfloat16=False` | torch backend + bf16; CPU |
| Chronos-2 | eager fp32 (= controlled) | same | CPU |
| TimesFM-3 | eager fp32 (= controlled) | same | CPU |
| TimesFM-2.5 | `torch.compile` (library default) | `torch_compile=False` | — |
| TiRex-2 | custom CUDA recurrent kernels + CUDA graphs | not available on GPU (custom kernels always on CUDA) | no CUDA graphs; CPU (eager, FLOPs) |
| TiRex | eager fp32 (`tirex.load_model` default backend) | same | xLSTM sLSTM CUDA kernels (`tirex-ts[cuda]`, bf16 recurrent weights); CPU |
| Moirai-2 | eager fp32 in a worker process (torch 2.4) | same | `compute_median_s` inside the worker, `ipc_overhead_s` = wall − compute |

* **Cold start** is reported separately (`first_call_s`: first call of a shape, incl. compilation /
  CUDA-graph capture); latencies are medians over the following timed repeats.
* **FLOPs per series** (`gflops_per_series`, `torch.utils.flop_counter`, matmul/convolution/attention,
  2 FLOPs per MAC) on the eager variant at batch 1, contexts ≤ 16,384: a hardware- and software-
  independent cost. Element-wise work (gates, scan recurrences) is not counted, so it is a lower bound.
* Figures `e8_latency_default_b*` (what a user gets) and `e8_latency_controlled_b*` (architecture), table
  `e8_gflops.csv`; `inference_cost.csv` lists every variant with its settings.
* Accuracy experiments are unchanged: they keep each library's default path (TiRex on its torch backend;
  its CUDA kernels change quantiles by ≤ 0.15 on a random walk of scale ~45 — a bf16 effect).
* `torch.compile` of TiRex's sLSTM (`compile=True`) was evaluated and **excluded**: it changed the
  forecasts materially (max |Δq| ≈ 22 on the same test), so it is not the same model.
* Paper framing: absolute latencies are "single 8 GB laptop GPU, default inference stack"; the
  architectural claims rest on the controlled curves, the context-scaling slopes, memory and FLOPs.
* TiRex pads every context to its training length (2,048) before the forward pass
  (`tirex/models/tirex.py::_adjust_context_length`), so its latency and FLOPs are flat below 2,048 — an
  implementation property to state when comparing short-context cost.
* One variant on the GPU at a time (the model cache is cleared between variants) and every cell records
  `gpu_free_mb_after`: on WSL/WDDM, exceeding VRAM silently spills to host memory and slows a call ~100x
  (seen once in testing when E8 shared the GPU with another experiment). E8 must run alone on the GPU.

## 8. E3 daily forecast table stored reduced (memory)

The full E3 run was OOM-killed (exit 137) after all 17 daily models had finished: each per-model table holds
~4.3 M rows (every phase × 6 contexts × 3 horizons, ~1 GB in memory) and concatenating them needed more
than the 15 GB of RAM. E3 now reduces each model's table on its own: metrics for every phase and context
go to `metrics_daily`, and `forecasts_daily` keeps only the **test phase at the context selected on
validation MAE** (the same rule as `analysis/report.select_contexts`; choices in
`selected_context_daily.csv`). Results are unchanged; the complete per-model tables remain in `_ckpt/`.
`report.py`, `probabilistic.py` and `paper.py` now count contexts per (model, horizon) so they accept the
reduced table. E3 is rerun after the rest of the queue (`scripts/rerun_e03_then_finish.sh`): Block A resumes
from its checkpoints, then Block B (hourly DE/GB/ES) runs.

## 9. E1 long-horizon (LTSF) suite: all nine datasets, checkpoints, 8 GB limits

* **Data.** Only the four ETT files were present, so the first run silently evaluated 4 of the 9 LTSF
  datasets. `scripts/get_data.py ltsf_large` now fetches Electricity (321 series), Weather (21),
  Traffic (862) and Exchange (8) from the Time-Series-Library release (`thuml/Time-Series-Library` on
  Hugging Face, the files used by the LTSF papers) and Solar-Energy (`solar_AL`, 137 plants, LSTNet
  repository; rewritten with a 10-min date column). Splits follow Time-Series-Library (0.7/0.1/0.2).
* **Checkpoints.** The LTSF table used to be written only at the end of E1; every (model, dataset) cell
  is now checkpointed in `_ckpt/ltsf_<model>_<dataset>`, and `ltsf_zero_shot.csv` is rebuilt from all
  finished cells. Rows now carry `n_variates` and `status`.
* **Parts.** `--only ltsf` (ETT) / `ltsf_large` / `fev`. The queue runs ETT + fev-bench first and the
  large datasets after every other experiment.
* **Cost-driven settings for the large datasets** (measured on the RTX 8 GB laptop GPU, one window at
  H = 336): with stride 24 and the full history for every model the five sets needed ~400-600 GPU-hours
  (e.g. Chronos-2 on Traffic at full context: 397 s per window, VRAM spill). For Electricity/Weather/
  Traffic/Exchange/Solar we therefore use **stride 96** (33-425 windows per horizon) and lookbacks
  **512 and 2048 for every model plus the full history for ci-proto** (and Seasonal Naive): the other
  TSFMs truncate inputs to their native context (2-16 k), so "full" adds cost but no information for
  them. ETT keeps stride 24 and all three lookbacks for every model.
* **Joint mode** (ci-proto multivariate) only for datasets with <= 32 variates (ETT, Weather, Exchange):
  on Electricity at full context it needs ~20 GB and fails on 8 GB. Any CUDA OOM is now recorded as a
  failed cell (`status = oom`, NaN metrics) instead of aborting the experiment.

## 10. TiRex-2 horizons beyond 320 steps: autoregressive rollout

The TiRex-2 checkpoint outputs at most `future_len` = 320 steps and the library silently truncates longer
requests (warning only). E1 crashed on the first LTSF cell with H = 336 (the queue log showed exit 0
because of an exit-code bug in `scripts/resume_queue.sh`, since fixed). The wrapper (`ciexp/models.py`,
`TiRex2._forecast`) now rolls out longer horizons autoregressively: forecast 320 steps, append the median
to the context, forecast the remainder. The first 320 steps are bit-identical to a direct forecast.
This affects TiRex-2 at H = 336/720 in E1 (LTSF) and the long-term GIFT-Eval configurations in E10, and
is noted next to those numbers (the official TiRex-2 GIFT-Eval entries may treat long horizons
differently). Rollout with covariates is refused (no experiment needs it: covariate horizons are <= 168).
Moirai-2 (`uni2ts` predict-token loop) and TiRex (patch rollout) already handle long horizons internally.
E1 (ETT + fev-bench) is rerun at the start of `scripts/after_full_run.sh`, resuming from its 16 finished cells.

## 11. fev-bench task failures and retry (2026-11-22)

The E1 fev-bench run (finished 2026-11-22) recorded 42 failed task runs, which are not written to the per-model CSVs:
Moirai-2 12 (its worker environment runs pandas 2.1, which rejects the pandas 2.2 frequency aliases `ME`/`QE`/`YE` used by
fev; `envs/moirai/worker.py` now maps them to `M`/`Q`/`A`), and 30 transient CUDA driver errors under WSL
(`INTERNAL ASSERT FAILED !handles_.at(i)`, `device not ready`) across TiRex 14, ci-proto[cohort] 11, ci-proto 3 and
Chronos-2 2. `scripts/after_full_run.sh` reruns `e01_benchmarks --only fev`, which skips finished tasks and retries only the
failures, before the final analysis. Any task still failing after the retry is imputed with Seasonal Naive, as fev's rules require.

E10 (GIFT-Eval, finished 2026-11-23) hit the same fault: 75 failed configs (TiRex-2 52, Chronos-2 23), all after one
"CUDA driver error: device not ready". Once that happens the CUDA context is unusable and every later task fails. fev-bench
and E10 now stop with exit 75 on such a fault (`ciexp.adapters.is_fatal_cuda`) instead of logging every later task as
failed. `after_full_run.sh` retries E1 fev and E10 up to four times in fresh processes, and also E3 and the large LTSF sets
if their last run exited non-zero. Saved results are never recomputed.

## 12. E8: per-cell time budget and per-variant checkpoints (2026-11-23)

The first full E8 run reached ci-proto's PyTorch-eager bf16 variant at context 262 144 × batch 32 (8.4 M points)
and spent > 4.5 h on that single cell. The cells before it, up to 65 536 × 32, took minutes. The working set outgrows
the 8 GB card, and WSL/WDDM then silently pages GPU memory to host RAM (50-100× slower), so the latency would not
have been valid anyway. Changes: (i) a cell whose first (cold) call takes longer than `CELL_BUDGET_S` = 300 s is not
repeated. It keeps its single-call time and memory with status `slow`, and larger contexts at the same (horizon, batch)
are recorded as `not run`. (ii) Checkpoints are written per (model, variant) instead of per model. The interrupted
ci-proto run had no checkpoint and is recomputed in full (≈ 7 h). The default fused-kernel grid had completed
without hitting the budget, so its cells are expected to be unaffected.

## 13. Tasks that crash the CUDA driver every time (2026-11-26)

The fev/E10 retries in `after_full_run.sh` all stopped on their first task: fev `entsoe_15T` (ci-proto) and GIFT-Eval
`m4_daily/D/short` (Chronos-2). Both fail with "CUDA driver error: device not ready / Failed to create GPU mapping".
On these inputs the WSL driver fault is deterministic, not transient, so stopping on it blocked every other retry. Any
task that crashes the driver is now written to `_cuda_fatal_<model>.txt` (fev: `results/E1_general_benchmarks/full/fev/`;
E10: `results/E10_gift_eval/full/rows/`) and skipped in later attempts. `scripts/retry_then_finish.sh` reruns until no new
crash occurs and then rebuilds the analysis. Skipped tasks count as failures (Seasonal Naive imputation, fev/GIFT-Eval
rule) and are listed in the results.

## 14. Gap reruns of E5 and E7 (2026-11-26/27)

* **E5, ci-proto at long contexts.** The first gap rerun reproduced the failures at ≥ 524,288 (h = 60)
  and ≥ 262,144 (h = 1440). A single-call probe showed both cells are feasible in isolation. The
  failures are **CUDA out-of-memory** errors (reserved memory grows across the 20 origins; WSL lets the
  allocator spill to shared memory until it fails). Contexts ≥ 1,048,576 with calendar features
  exceed the 8 GB card even in isolation. Fix: `experiments/e05_context.py` frees the CUDA cache
  (`gc.collect()` + `torch.cuda.empty_cache()`) before each forecast above 131,072 steps. This changes
  memory behaviour only, not forecasts. Cells that still fail are reported as hardware-limited (OOM on 8 GB).
* **E7, ci-proto missing-data study.** The first gap rerun hit a WSL CUDA fault (`!handles_.at(i)`) at
  the start and voided every dataset. The study is now checkpointed per dataset
  (`missing_ci-proto__<dataset>`) and aborts with exit 75 on a fatal CUDA status so that
  `scripts/rerun_e7_missing.sh` restarts it in a fresh process. The checkpoints for bike, Ember and synthetic are the
  valid rows (all `ok`) of the main run (`results/_ckpt_backup_20261126/E7/missing_ci-proto.parquet`).
  ETTh1 and Beijing are recomputed.
* `tta_bfloat16=False` (fp32) on ETTh1 remains an error cell (CUDA fault); the fp32 ablation is
  reported on the other four domains.
