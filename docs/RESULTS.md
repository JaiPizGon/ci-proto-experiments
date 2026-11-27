# ci-proto — findings report for the paper (full run, 18–26 Nov 2026)

This report covers the full experimental run of ci-proto, a 167 M-parameter zero-shot probabilistic TSFM: TiRex-2 with a bidirectional Mamba-3 time mixer. It covers E1–E10, the probabilistic-forecast tests and the inference-cost study. Every number below comes from a file in `results/`. File paths are given so each table can be traced. Protocol deviations and bug fixes made during the run are documented in [`PROTOCOL_CHANGES.md`](PROTOCOL_CHANGES.md) (sections 1–13).


> **Update 2026-11-27: gap reruns complete.** Every cell that the CUDA-driver fault had voided in E5 and E7 has been recomputed (PROTOCOL_CHANGES §14):
> - E5: ci-proto at 262 k and 524 k context (both horizons).
> - E7: ci-proto missing data on ETTh1 and Beijing, the independent mode, the six TTA flags, and fp32 on ETTh1.
>
> The only remaining gap is ci-proto at ≥ 1 M context in E5, which runs out of memory on the 8 GB GPU when calendar features are on. No number in this report is imputed or estimated.

**Set-up.**
- Hardware and software: one RTX 4070 Laptop GPU (8 GB) under WSL2, torch 2.14 + CUDA 13.0.
- Comparators: Chronos-2, TimesFM-3, TimesFM-2.5, TiRex-2, TiRex, Moirai-2, plus task-specific statistical and supervised baselines.
- All TSFMs run zero-shot. Where a model's native window is shorter than the requested context, inputs are truncated to it.
- Lower is better for every error metric. COV80 should be 0.80.

---

## 1. Headline findings

1. **Point and probabilistic accuracy is at the top of the field, but the margin is small and test-dependent.**
   - ci-proto ranks first on the leakage-free synthetic suite (E2), the long-horizon benchmarks (E1 ETT/large LTSF) and fev-bench (among our re-runs). It is second on GIFT-Eval and best on most hourly electricity-price cells (E3).
   - Against the strongest comparators (TimesFM-3, Chronos-2, TiRex-2), the gap is usually 1–4 % and often inside the confidence set.
2. **Single-model claims that hold up statistically:**
   - **E2 synthetic suite:** best average rank (2.65 vs 3.06 for TimesFM-3). The Nemenyi critical difference is 0.36, so the rank gap is significant. ci-proto also has the highest skill vs Seasonal Naive (0.601 [0.584, 0.617]).
   - **fev-bench, our re-run:** win rate 0.855 [0.823, 0.885] and skill 0.497. Both are above every other re-run model, including TimesFM-3 (ours: 0.734 / 0.458).
   - **E3 hourly prices:** best CRPS in 6 of 9 market × horizon cells and in the 90 % CRPS model confidence set in all 9.
3. **The cohort mode is ci-proto's best configuration on multi-series data.** On the 23-market daily price panel, `ci-proto[cohort]` has the lowest MAE and CRPS of all 17 models at every horizon, and the lowest RMSE at 7 and 30 days. Plain univariate ci-proto ranks mid-pack among the TSFMs there. Joint mode lowers ci-proto's MSE by 0–9 % on the multivariate LTSF sets.
4. **Long context: it helps up to ~16 k steps, then stops helping or hurts.**
   - On 1-min household power (E5), MASE is flat beyond 2 k (h = 60). For h = 1440 it is best at 16 k (0.718) and 3–8 % worse at 65–524 k (0.742–0.773).
   - On Beijing PM2.5 (E6), a history of 20–80 % of the training period beats the full history by 25–28 % RMSE at leads of 24–48 h.
   - The architecture *runs* at 2 M steps (E8: 3.2 s, 5.7 GB). The run does *not* show that accuracy keeps improving with context.
5. **Prediction intervals are too narrow.**
   - Univariate ci-proto has the worst 80 %-interval calibration among the TSFMs on the leakage-free and prospective tests: COV80 0.62–0.74 on E2, 0.72–0.73 on E3 daily.
   - The cohort mode is the best-calibrated model overall (mean |COV80 − 0.8| = 0.032).
   - The paper should report this, and post-hoc conformal calibration is a natural add-on.
6. **Prospective 2026 window (after every model's training cut-off):**
   - ci-proto ties for best point accuracy at h = 1 and h = 7: MAE within 0.05 % of the best model.
   - Its CRPS is 0.5 % behind the best at h = 1 (TimesFM-3) and 1.2 % behind at h = 7 (Chronos-2).
   - In the short post-release window (13 days, 2026-11-13 → 25) it is 5th–6th of the 7 TSFMs. This window is too short to support a conclusion.
7. **Covariates:**
   - Known future covariates help ci-proto most among the covariate-capable models (bike sharing: −15 % / −18 % MASE, both Holm-significant). On bike sharing it degrades far less than Chronos-2 under noisy future covariates.
   - **Past-only covariates hurt ci-proto significantly on Beijing** (+20 % / +36 % MASE). This is a limitation to report.
8. **Cost:**
   - At 2 k context and batch 1, ci-proto runs at 0.059 s, the same order as Chronos-2 (0.049 s), 5× faster than TimesFM-3 and 4× slower than TiRex-2.
   - Latency and memory grow linearly with context. TiRex-2 is nearly flat: 0.054 s and 0.36 GB at 2 M steps, against 3.16 s and 5.7 GB for ci-proto.
9. **Contamination warning for comparators.**
   - The E2 contamination audit flags TimesFM-2.5 and TimesFM-3 on Beijing air quality. Their long-lead error is 37 % and 59 % of the median model's.
   - Their implausibly low E4/E6/E7 Beijing errors (e.g. RMSE ≈ 19 at every lead 1–48 h for TimesFM-2.5) should be excluded from, or flagged in, any comparison.

---

## 2. E1 — General benchmarks

### 2a. LTSF zero-shot (ETT ×4, Electricity, Traffic, Weather, Exchange, Solar)

MSE on train-z-scored data, horizons 96/192/336/720, lookbacks 512/2048/full. There are 944 cells in total, all with status `ok` (`results/E1_general_benchmarks/full/ltsf_zero_shot.csv`).

**Rank over 36 dataset × horizon cells, channel-independent mode, 7 TSFMs:**

| Model | Mean rank, L = 512 | Wins, L = 512 | Mean rank, L = 2048 | Wins, L = 2048 | Mean rank, best lookback | Wins, best lookback |
|---|---|---|---|---|---|---|
| **ci-proto** | **2.64** | 9 | **2.69** | 10 | **1.89** | **19** |
| TiRex-2 | 3.28 | 9 | 2.94 | **12** | 3.50 | 6 |
| Chronos-2 | 4.72 | 1 | 3.28 | 4 | 3.67 | 0 |
| TimesFM-2.5 | 3.83 | 2 | 4.69 | 4 | 4.00 | 5 |
| TimesFM-3 | 4.72 | 4 | 4.39 | 0 | 4.08 | 1 |
| Moirai-2 | 4.61 | 3 | 5.14 | 3 | 5.39 | 4 |
| TiRex | 4.19 | 8 | 4.86 | 3 | 5.47 | 1 |

"Best lookback" picks each model's best lookback per cell on the test set, an oracle that favours models with more lookbacks. On the five large sets only ci-proto and Seasonal Naive have a `full` lookback (cost-driven, see PROTOCOL_CHANGES §9), so report it as a secondary view. The fixed-lookback columns are the fair comparison.

**Mean MSE over horizons, best lookback:**

| Dataset | ci-proto | Best other (model) |
|---|---|---|
| ETTh1 | 0.388 | 0.384 (TiRex-2) |
| ETTh2 | **0.327** | 0.328 (TiRex) |
| ETTm1 | **0.330** | 0.360 (TimesFM-3) |
| ETTm2 | **0.248** | 0.248 (TimesFM-3) |
| Electricity | **0.147** | 0.153 (Chronos-2) |
| Traffic | 0.343 | **0.324** (TimesFM-2.5) |
| Weather | **0.201** | 0.221 (TiRex-2) |
| Exchange | 0.370 | **0.230** (TiRex-2) |
| Solar | 0.206 | **0.164** (Moirai-2) |

**Weaknesses:**
- **Exchange:** a random-walk-like series where TiRex-2 is far ahead and ci-proto does not beat Seasonal Naive at H = 720.
- **Solar:** Chronos-2 and Moirai-2 are best there.
- **Traffic:** TimesFM-2.5 is best there.

**Joint (multivariate) mode** changes ci-proto's MSE by a mean ratio of 0.91 (Weather), 0.95 (ETTh2, Exchange), 0.98 (ETTh1, ETTm2) and 1.00 (ETTm1). In single cells it is at most 4 % worse.

**Full history vs 2048** for ci-proto gives a mean MSE ratio of 0.90–0.97 on most sets. The exception is ETTh2 (1.05), where the longer history hurts.

### 2b. fev-bench (100 tasks), our re-run

Source: `results/tables/full/fev_leaderboard_SQL.csv`.
- Failed tasks are imputed with Seasonal Naive, per fev's rule.
- Win rate and skill are vs Seasonal Naive, with 95 % bootstrap CIs.
- Rows without "(ours)" are the official leaderboard entries on the same tasks.

| Model | Win rate | Skill (SQL) | Failed tasks |
|---|---|---|---|
| ci-proto (official, self-reported) | 0.921 [0.900, 0.941] | 0.515 [0.471, 0.561] | 0 |
| **ci-proto[cohort] (ours)** | **0.860** [0.823, 0.893] | 0.493 | 5 |
| **ci-proto (ours)** | 0.855 [0.823, 0.885] | **0.497** [0.448, 0.546] | 2 |
| TimesFM-3 (official) | 0.823 | 0.490 | 0 |
| TimesFM-3 (ours) | 0.734 | 0.458 | 0 |
| TiRex-2 (ours) | 0.715 | 0.455 | 0 |
| Chronos-2 (ours) | 0.701 | 0.462 | 2 |
| TimesFM-2.5 (ours) | 0.654 | 0.467 | 0 |
| TiRex (ours) | 0.638 | 0.427 | 0 |
| Moirai-2 (ours) | 0.496 | 0.399 | 1 |

**Reproduction fidelity** (`fev_reproduction_fidelity.csv`): our numbers divided by the self-reported ones.

| Model | Geometric-mean SQL ratio (ours / official) | Share of tasks within 2 % | Notes |
|---|---|---|---|
| TiRex, TiRex-2, TimesFM-2.5 | ≈ 1.000 | ≥ 0.99 | Matched exactly |
| Moirai-2 | 1.000 | 0.83 | |
| Chronos-2 | 1.008 | 0.76 | |
| **ci-proto** | **1.022** | 0.65 | Median ratio 1.005, Spearman 0.995 |
| TimesFM-3 | 1.063 | 0.54 | |

Our ci-proto re-run is about 2 % worse than the self-reported entry, which partly explains its lower win rate (0.855 vs 0.921). The paper should cite our numbers and mention the gap.

**Failures:** the CUDA-driver faults (PROTOCOL_CHANGES §11–13) left a few tasks unrecoverable. They count as Seasonal Naive:
- ci-proto: `entsoe_15T`, `solar_with_weather_15T`
- cohort mode: the same two plus `LOOP_SEATTLE_5T`, `LOOP_SEATTLE_1H`, `ercot_1H`
- Chronos-2: `LOOP_SEATTLE_5T`, `LOOP_SEATTLE_1H`
- Moirai-2: 1 task

## 3. E10 — GIFT-Eval (97 configurations)

Source: `results/tables/full/gift_eval_leaderboard.csv`. CRPS and MASE are geometric means normalised by the official Seasonal Naive. "Avg rank" is the average CRPS rank.

| Model | CRPS | MASE | Avg rank |
|---|---|---|---|
| TimesFM-3 (official) | **0.456** | **0.667** | **2.99** |
| **ci-proto (ours)** | 0.462 | 0.676 | 4.03 |
| TiRex-2 (official) | 0.478 | 0.697 | 5.63 |
| Toto-2.0 (official) | 0.481 | 0.703 | 5.47 |
| Chronos-2 (official) | 0.485 | 0.698 | 5.52 |
| TiRex (official) | 0.488 | 0.716 | 6.09 |
| TimesFM-2.5 (official) | 0.490 | 0.705 | 6.60 |
| TiRex-2 (ours) | 0.501 | 0.713 | 7.13 |
| Moirai-2 (official) | 0.516 | 0.728 | 8.73 |

- ci-proto is second, 1.3 % behind TimesFM-3 in CRPS and ahead of every other published model.
- There is no official ci-proto GIFT-Eval entry, so this is new evidence.

**Caveats:**
- Our TiRex-2 re-run is 2.3 % worse than its official CRPS (median ratio 1.023). Treat about 2 % as the reproduction noise floor of our harness on GIFT-Eval.
- Some leaderboard models were trained on GIFT-Eval's pre-training split or test data. Check the leaderboard's leakage flags before comparing against TimesFM-3.
- Chronos-2 (ours) is incomplete: 23 configurations crash the WSL CUDA driver every time (electricity, solar, kdd_cup_2018, loop_seattle, m_dense, m4_daily). Its official entry is used instead.

## 4. E2 — Leakage control

### 4a. Synthetic suite

Source: `results/tables/full/e2_leaderboard_SQL.csv` and `e2_friedman.json`. The suite has 1416 series (8 processes plus 14 families × 10 series, 5 seeds) at h = 24/168. Skill is SQL vs Seasonal Naive.

| Model | Skill [95 % CI] | Win rate | Avg rank |
|---|---|---|---|
| **ci-proto** | **0.601** [0.584, 0.617] | **0.817** | **2.65** |
| TimesFM-3 | 0.590 [0.573, 0.605] | 0.771 | 3.06 |
| Chronos-2 | 0.584 [0.567, 0.600] | 0.739 | 3.35 |
| TimesFM-2.5 | 0.561 | 0.634 | 4.29 |
| TiRex-2 | 0.545 | 0.613 | 4.48 |
| TiRex | 0.555 | 0.553 | 5.02 |
| Moirai-2 | 0.537 | 0.422 | 6.20 |
| AutoTheta | 0.154 | 0.180 | 8.38 |

- The Friedman test gives χ² = 8061 (p ≈ 0) with Nemenyi CD = 0.36. **ci-proto's average rank is significantly better than every other model's.**
- Where it gains most:
  - Chaotic systems: Lorenz skill 0.68 vs ≤ 0.51 for the others; Mackey–Glass 0.80 vs ≤ 0.70.
  - Seasonality breaks: MASE 0.725 vs ≥ 0.78.
- Where it is worse: logistic-seasonal series (0.59 vs 0.70 for TimesFM-3/Chronos-2) and trend breaks (behind TimesFM-3/2.5).

**Contamination audit** (`contamination_audit.csv`): long-lead RMSE relative to the median model. The only flags are TimesFM-2.5 (0.37) and TimesFM-3 (0.59) on Beijing air quality, so those Beijing results are treated as likely training-set exposure.

### 4b. Prospective window (Ember daily prices, 23 markets)

Source: `prospective_summary.csv`. The hold-out covers 2026-01-01 → 2026-11-25 (7.6 k windows, after every comparator's training cut-off). The post-release window covers 2026-11-13 → 25 (299 h = 1 windows, 161 h = 7 windows).

| Model | MAE, hold-out h = 1 | MAE, hold-out h = 7 | CRPS, hold-out h = 1 | CRPS, hold-out h = 7 | COV80, hold-out h = 1 | MAE, post-release h = 1 | MAE, post-release h = 7 |
|---|---|---|---|---|---|---|---|
| ci-proto | 19.86 | 26.25 | 15.70 | 20.80 | 0.72 | 33.81 | 42.26 |
| TimesFM-3 | **19.85** | 26.30 | **15.63** | 20.66 | 0.75 | 32.89 | **40.13** |
| Chronos-2 | 19.92 | **26.24** | 15.65 | **20.55** | 0.75 | 34.58 | 41.02 |
| TiRex-2 | 19.94 | 26.45 | 15.64 | 20.71 | 0.78 | 33.31 | 40.83 |
| TimesFM-2.5 | 19.87 | 26.48 | 15.65 | 20.82 | 0.79 | **32.77** | 41.42 |
| TiRex | 20.13 | 27.00 | 15.79 | 21.01 | 0.78 | 32.88 | 40.85 |
| Moirai-2 | 20.42 | 27.04 | 15.99 | 21.08 | 0.79 | 34.48 | 42.48 |
| AutoTheta | 21.61 | 30.33 | 17.36 | 24.68 | 0.77 | 36.50 | 44.81 |
| Seasonal Naive | 35.33 | 35.19 | 30.33 | 30.21 | 0.51 | 77.62 | 78.05 |

Accuracy and CRPS differences among the top five models in the hold-out are under 0.5 %. ci-proto's intervals are the narrowest and its COV80 is the lowest.

## 5. E3 — Electricity prices

### 5a. Daily, 23 Ember markets

Test period 2024–2025. Contexts are selected on 2023 validation data. Sources: `results/E3_electricity_prices/full/summary_daily.csv` and `analysis/daily_*`.

| Model | MAE, h = 1 | MAE, h = 7 | MAE, h = 30 | CRPS, h = 1 | CRPS, h = 7 | CRPS, h = 30 | COV80, h = 1 |
|---|---|---|---|---|---|---|---|
| **ci-proto[cohort]** | **15.26** | **19.26** | **21.67** | **12.05** | **15.15** | **17.01** | 0.75 |
| TimesFM-3 | 15.30 | 19.36 | 22.02 | 12.09 | 15.23 | 17.34 | 0.77 |
| Chronos-2 | 15.36 | 19.38 | 21.76 | 12.14 | 15.26 | 17.11 | 0.75 |
| TiRex-2 | 15.41 | 19.40 | 22.08 | 12.14 | 15.27 | 17.40 | 0.79 |
| TimesFM-2.5 | 15.41 | 19.56 | 21.87 | 12.17 | 15.41 | 17.21 | 0.81 |
| ci-proto | 15.47 | 19.63 | 22.16 | 12.25 | 15.53 | 17.45 | 0.72 |
| LEAR-recal | 17.46 | 22.17 | 24.22 | 14.16 | 17.91 | 19.79 | 0.85 |

- Diebold–Mariano tests (HLN, Holm 5 %, per market) use univariate ci-proto as the reference.
  - With absolute loss, it beats AutoETS, AutoTheta, GBM, LEAR, LEAR-recal and the naive baselines in 96–100 % of markets at h = 1 and 7.
  - At h = 30 the shares are 78–100 %, except LEAR-recal and N-BEATS (both 43 %).
  - It beats no TSFM significantly in more than 17 % of markets.
- The 90 % model confidence set (MCS) on absolute and pinball loss contains ci-proto[cohort], TimesFM-3, TimesFM-2.5, TiRex-2 and Chronos-2 at every horizon. Univariate ci-proto drops out at h = 1 (pinball) and h = 7 (absolute loss).

### 5b. Hourly, Germany / Great Britain / Spain

Horizons 4/24/40 h. Test MAE / CRPS / COV80 at each model's validation-selected context. Source: `analysis/{DE,GB,ES}_pooled.csv`.

| Market, horizon | Best model on CRPS | Runner-up | ci-proto rank on CRPS |
|---|---|---|---|
| DE, 4 h | **ci-proto** 5.18 / 4.13 / 0.84 | TiRex-2 5.45 / 4.41 | 1 |
| DE, 24 h | **ci-proto** 17.52 / 14.01 / 0.76 | TimesFM-3 17.51 / 14.02 | 1 |
| DE, 40 h | TimesFM-3 19.55 / 15.62 | ci-proto 19.71 / 15.67 | 2 |
| GB, 4 h | **ci-proto** 5.39 / 4.27 / 0.79 | Chronos-2 5.63 / 4.47 | 1 |
| GB, 24 h | TiRex-2 11.64 / 9.36 | Chronos-2 11.97 / 9.46 | 4 |
| GB, 40 h | **ci-proto** 13.54 / 10.80 / 0.78 | TiRex-2 13.59 / 10.91 | 1 |
| ES, 4 h | **ci-proto** 5.08 / 4.02 / 0.86 | TiRex-2 5.23 / 4.16 | 1 |
| ES, 24 h | Chronos-2 11.34 / 8.95 | TimesFM-3 11.94 / 9.47 | 6 |
| ES, 40 h | **ci-proto** 13.73 / 10.78 / 0.83 | Chronos-2 14.35 / 11.30 | 1 |

- ci-proto's gain is largest at the 4-hour horizon: 3–6 % lower CRPS than the runner-up in all three markets.
- Its weak spot is the day-ahead (24 h) horizon in Great Britain and Spain.
- LEAR and LEAR-recalibrated are 30–45 % worse than the best TSFM in every hourly market.
- ci-proto is in the 90 % CRPS MCS for all 9 cells (§10).

## 6. E4 — Covariates and calendar features

Beijing air quality (12 stations) and UCI bike sharing, h = 24/168, 52 weekly origins.

Conditions:
- C0: no covariates.
- C1: calendar features.
- C2: plus known holiday/working-day flags.
- C3: plus past-only weather.
- C4: C2 plus actual future temperature, with past humidity and wind.
- C5: C4 with Gaussian noise of σ × 0.5 / 1 / 5 on the future covariates.

The table shows the median % change in MASE vs C0 (`tests_vs_C0.csv`). `*` marks Holm-adjusted Wilcoxon p < 0.05.

| Dataset, horizon | Model | C1 | C2 | C3 | C4 | C5, σ × 1 | C5, σ × 5 |
|---|---|---|---|---|---|---|---|
| Bike sharing, 24 | ci-proto | −1.0 | +1.6 | −6.3 | **−15.1\*** | +7.6 | +9.7 |
| | Chronos-2 | 0 | +8.2 | +5.9 | +2.1 | +51.9\* | +121.4\* |
| | TiRex-2 | 0 | −1.1 | −0.3 | −1.2\* | −2.7\* | −1.5 |
| Bike sharing, 168 | ci-proto | −0.4 | −1.7\* | −3.4\* | **−18.0\*** | +2.8 | +6.1\* |
| | Chronos-2 | 0 | +1.2 | +1.5 | −7.9\* | +4.2 | +25.4\* |
| Beijing, 24 | ci-proto | +1.7\* | — | **+20.1\*** | +9.3\* | +39.4\* | +68.9\* |
| | Chronos-2 | 0 | — | −0.7\* | **−21.0\*** | +49.2\* | +79.7\* |
| | TiRex-2 | 0 | — | −0.1 | −7.4\* | +131.1\* | +162.5\* |
| Beijing, 168 | ci-proto | +3.9\* | — | **+36.3\*** | +1.9 | +41.9\* | +54.5\* |
| | Chronos-2 | 0 | — | +0.3\* | −35.2\* | +4.6\* | +26.7\* |

- On bike sharing, ci-proto uses known future covariates best: −15 % and −18 %. With C4 it has the lowest MASE of all models (0.620 / 0.636, vs TimesFM-3 at C0: 0.725 / 0.731). Under noisy covariates it degrades far less than Chronos-2 (+10 % vs +121 % at σ × 5). TiRex-2 barely changes because it largely ignores covariates.
- On Beijing, ci-proto is significantly *hurt* by adding past-only weather covariates (C3) and by calendar features (C1). Chronos-2 gains most from future temperature (C4).
- Here only Chronos-2, TiRex-2 and ci-proto take covariates; the other models run C0/C1 only.
- The TimesFM-2.5 and TimesFM-3 Beijing numbers (MASE 0.20–0.34) are contamination-flagged (§4a) and should not be used as a reference.

## 7. E5 — Context scaling (UCI household power, 1-min, about 2.07 M points)

Source: `results/E5_context_scaling/full/context_scaling.csv`. 20 daily origins. The table shows MASE.

| Context | ci-proto, h = 60 | Best other, h = 60 | ci-proto, h = 1440 | TimesFM-3, h = 1440 | TiRex-2, h = 1440 |
|---|---|---|---|---|---|
| 512 | 0.160 | 0.157 (Moirai-2) | 0.889 | 1.004 | 0.977 |
| 2 048 | 0.151 | 0.150 (Moirai-2, TimesFM-2.5) | 0.772 | 0.760 | 0.830 |
| 8 192 | 0.155 | 0.149 (TimesFM-3) | 0.724 | **0.692** | 0.830 |
| 16 384 | 0.153 | 0.148 | **0.718** | 0.699 | 0.830 |
| 65 536 | 0.151 | 0.148 | 0.743 | (truncated) | 0.830 |
| 131 072 | 0.152 | 0.148 | 0.760 | | |
| 262 144 | 0.152 | 0.148 | 0.773 | | |
| 524 288 | 0.152 | 0.148 | 0.742 | | |
| ≥ 1 048 576 | OOM (8 GB) | 0.148 | OOM (8 GB) | | |

- At h = 60, every competent model saturates by about 2 k steps.
- At h = 1440, ci-proto improves up to 16 k and is then 3–8 % worse up to 524 k (non-monotone: 0.773 at 262 k, 0.742 at 524 k). TimesFM-3 (native 16 k) is best overall.
- The other models plateau because they truncate to their native window, so "full context" adds nothing for them.
- ci-proto at 262 k and 524 k context was recomputed in the gap reruns (PROTOCOL_CHANGES §14). At ≥ 1 M steps with calendar features, ci-proto runs out of memory on the 8 GB card (E8 shows the model itself runs at 2 M steps without calendar features), so these cells are reported as hardware-limited.
- TimesFM-2.5 fails at h = 1440: the horizon exceeds its maximum output length.

**After structural breaks** (`regime_breaks.csv`, MASE at 256 → 65 k context):
- ci-proto handles seasonality breaks best: 0.75–0.77 at every context, against 0.87–1.58 for the others.
- On trend breaks it collapses at the longest context: 0.97 → 1.44 at 65 k. Context inertia is real here.

## 8. E6 — Data scarcity (Beijing multi-site PM2.5)

Source: `results/E6_data_scarcity/full/summary.csv`. RMSE in µg/m³ by lead.
- Zero-shot TSFMs are compared with LSTM, PatchTST, N-BEATS and quantile GBM, each trained on 20–100 % of two years of history.
- ci-proto runs with the 120-step window, a history matched to the training ratio, and the full history.

| Lead (h) | ci-proto, matched 0.2 / 0.8 / full | TiRex-2, full | Chronos-2, full | Moirai-2, full | Best supervised (100 %) | Seasonal Naive |
|---|---|---|---|---|---|---|
| 1 | 18.1 / 18.0 / 19.2 | 18.5 | 19.7 | 18.7 | 19.0 (N-BEATS) | 91.5 |
| 6 | 38.0 / 37.5 / 43.0 | 42.5 | 49.7 | 49.6 | 50.1 (GBM) | 91.2 |
| 24 | 43.1 / 40.9 / 55.0 | 51.4 | 77.5 | 81.3 | 81.4 (PatchTST) | 89.5 |
| 48 | 45.9 / 43.2 / 59.7 | 55.8 | 86.3 | 88.3 | 90.9 (GBM) | 107.3 |

- From lead 6 onward, the best zero-shot TSFMs beat every supervised model at every training ratio. At lead 24, the best supervised model (PatchTST, 100 % of the data) has 1.5× ci-proto's full-history error and 2× its matched-history error.
- With a moderate history (matched 0.4–0.8), ci-proto is the best non-flagged model at every lead ≥ 3.
- Giving ci-proto the full history raises its RMSE by 15–38 %, which agrees with E5.
- TimesFM-2.5 and TimesFM-3 are excluded from this ranking. Their errors (TimesFM-2.5: RMSE ≈ 19 at every lead 1–48 h) are inconsistent with forecasting from the past alone, and the audit flags both on this dataset.

## 9. E7 — Ablations and missing data

Source: `results/E7_ablations_robustness/full/summary.csv`, `tests_vs_default.csv`. The table shows SQL.

**Test-time processing (TTA) flags:**
- `postprocess=False` is the only flag that matters everywhere: Beijing 0.407 → 0.496, bike 0.460 → 0.484, synthetic 0.625 → 0.641, Ember 0.898 → 0.903.
- `tta_sign_flip=False` matters as much on Beijing and bike.
- `tta_diff=False` gives a small loss on synthetic data (0.625 → 0.632).
- Disabling calendar covariates *helps* slightly on bike (0.460 → 0.436), synthetic (0.625 → 0.597) and Ember (0.898 → 0.891). This repeats the E4 C1 finding.
- `integer_round`, `clip_nonnegative`, `target_encode_categoricals` and `tta_running_totals` are neutral (|Δ| < 0.1 %). Turning `bfloat16` off (fp32) is neutral on bike and Ember and slightly better on the synthetic suite (0.625 → 0.615).

**Inference mode:**
- Cohort and joint modes improve Ember daily prices: 0.898 → 0.859 (−4.3 %).
- They make no difference on ETTh1, bike or synthetic data.

**Missing values** (SQL at 0 → 50 % missing):

| Dataset | Pattern | ci-proto | Chronos-2 | TiRex-2 | TimesFM-3 | Moirai-2 |
|---|---|---|---|---|---|---|
| Bike sharing | Block | 0.460 → **0.471** | 0.583 → 0.619 | 0.517 → 0.629 | 0.387 → 0.637 | 0.432 → 0.628 |
| Bike sharing | Random | 0.460 → **0.521** | 0.583 → 0.587 | 0.517 → 0.597 | 0.387 → 0.588 | 0.432 → 0.825 |
| Ember daily | Random | 0.898 → 0.913 | 0.866 → 0.912 | 0.873 → 0.912 | 0.875 → 0.944 | 0.878 → 1.044 |
| Synthetic | Random | 0.625 → 0.763 | 0.702 → 1.597 | 0.656 → 0.697 | 0.615 → 0.733 | 0.662 → 1.434 |
| ETTh1 | Block | 0.572 → 0.592 | 0.566 → 0.587 | 0.568 → 0.599 | 0.578 → 0.764 | 0.578 → 0.652 |
| ETTh1 | Random | 0.572 → 0.639 | 0.566 → 0.606 | 0.568 → 0.599 | 0.578 → 0.676 | 0.578 → 0.793 |
| Beijing | Block | 0.407 → 0.425 | 0.302 → 0.319 | 0.307 → 0.367 | 0.218 → 0.314\* | 0.419 → 0.485 |
| Beijing | Random | 0.407 → 0.428 | 0.302 → 0.257 | 0.307 → 0.339 | 0.218 → 0.226\* | 0.419 → 0.570 |

\* TimesFM-3 is contamination-flagged on Beijing (§4a).

- ci-proto is the most robust model to block-missing data on bike sharing.
- On Beijing it degrades least among the non-flagged models under block missingness (+4 %, against +6 % for Chronos-2 and +16–20 % for TiRex-2 and Moirai-2). Under random missingness it degrades +5 %, while Chronos-2 even improves. Its absolute SQL on Beijing (0.407) is, however, worse than that of Chronos-2 and TiRex-2.
- On ETTh1 it is mid-pack: +3.5 % (block) and +12 % (random at 50 %), against +4–7 % for Chronos-2 and +5 % for TiRex-2.
- TiRex-2 and TimesFM-3 are most robust to random missingness on the synthetic suite. Moirai-2 and Chronos-2 degrade most there.

## 10. Probabilistic forecast tests

Sources: `results/tables/full/prob_*`, `analysis/probabilistic.py`. There are 15 evaluation sources (E2 prospective, E3 daily and hourly, E9). The tests are:
- Diebold–Mariano tests on CRPS per series with Holm correction, plus a pairwise matrix.
- MCS (90 %) on CRPS.
- Kupiec and Christoffersen tests of 80 %-interval coverage.

| Model | Sources where the model is in the 90 % CRPS MCS (of 15) | Mean \|COV80 − 0.8\| | Share of sources where Kupiec rejects |
|---|---|---|---|
| **ci-proto** | **14** | 0.083 | 0.64 |
| **Chronos-2** | **14** | 0.055 | 0.59 |
| TimesFM-3 | 13 | 0.060 | 0.59 |
| TiRex-2 | 13 | 0.063 | 0.56 |
| TiRex | 11 | 0.054 | 0.53 |
| Moirai-2 | 9 | 0.037 | 0.38 |
| TimesFM-2.5 | 6 | 0.096 | 0.60 |
| ci-proto[cohort] (E3 daily only) | 3 of 3 | **0.032** | 0.46 |

- Distribution accuracy (CRPS) is ci-proto's strength. It and Chronos-2 are the only models in the confidence set for 24 h and 40 h Spain.
- Calibration is its weakness:
  - Univariate intervals are too narrow on the daily price data. COV80 is 0.62–0.72 in the E2 prospective windows and 0.72–0.73 in E3 daily.
  - In the hourly markets COV80 is 0.83–0.93, too wide.
  - That is, the univariate model's interval width does not adapt to the data's scale.
- These calibration tables evaluate one forecast set per source, the selected or full context (`probabilistic.py`). The E3 hourly coverages therefore differ from the context-selected values in §5b.

## 11. E8 — Inference cost (RTX 4070 Laptop, 8 GB)

Source: `results/E8_efficiency/full/efficiency.csv`. Default inference path for each model, batch 1, horizon 64. Latency is the median of 10 calls after 3 warm-ups.

| Context | ci-proto (fused Mamba-3) | TiRex-2 | Chronos-2 | TimesFM-3 | TimesFM-2.5 | TiRex | Moirai-2 |
|---|---|---|---|---|---|---|---|
| 2 048 | 0.059 s / 0.79 GB | **0.015 s** / 0.35 GB | 0.049 s / 0.47 GB | 0.328 s / 1.27 GB | 0.274 s / 0.91 GB | 1.42 s / 0.14 GB | 0.018 s @ 1680 |
| 16 384 | 0.056 s / 0.82 GB | 0.012 s / 0.35 GB | (native 8 k) | 1.044 s / 1.32 GB | (native 15 k) | (native 2 k) | (native 1.7 k) |
| 262 144 | 0.397 s / 1.96 GB | 0.015 s / 0.35 GB | | | | | |
| 2 097 152 | 3.16 s / 5.72 GB | 0.054 s / 0.36 GB | | | | | |

- **Size:** ci-proto has 167 M parameters, against 82.5 M (TiRex-2), 119 M (Chronos-2), 331 M (TimesFM-3) and 231 M (TimesFM-2.5).
- **FLOPs** per series grow linearly for ci-proto: 27.6 GFLOP at 2 k, 99 at 8 k, 195 at 16 k. TiRex-2 is constant at 5.8. Chronos-2 needs 130 at 8 k and TimesFM-3 144 at 8 k.
- **Batch 32 at 2 k:** 0.164 s for ci-proto, 0.100 s for TiRex-2, 0.22 s for Chronos-2 and 1.56 s for TimesFM-3.
- **Fused Mamba-3 kernels:** they make ci-proto 6× (512) to 290× (262 k) faster than its PyTorch-eager path at batch 1. For example, at 262 k it takes 0.41 s fused against 119 s eager in bf16.
- **CPU:** at 2 k and batch 1, ci-proto takes 3.8 s, against 0.27 s for TiRex-2 and 0.21 s for Chronos-2. CPU deployment is a weakness.
- **Cells not measured:** PyTorch-eager cells whose first call took over 300 s were not repeated (PROTOCOL_CHANGES §12). The eager ci-proto variants at ≥ 1 M context and Chronos-2 at 8 k × 256 hit the CUDA-driver fault.

## 12. E9 — Volatility (negative control)

Data: daily |returns| of 17 FRED series (equity indices, FX, oil, gas), one-step-ahead, for 2023, 2024 and 2025. Source: `results/E9_volatility/full/`.
- **No model dominates.** All TSFMs are within ±2 % of HAR-RV on RMSE.
- **ci-proto** has the second-lowest QLIKE in 2025 (0.526, after TiRex-2 at 0.518) and is in the 90 % MCS for 5 of 6 year × loss cells.
- **Significance vs HAR-RV:** ci-proto beats it with Holm significance in 6 % (squared loss) and 16 % (QLIKE) of series-years. Other TSFMs manage 6–12 % and 16–22 %.
- **Moirai-2** has one extreme QLIKE failure in 2024 (3102).

This is the expected negative-control result. TSFMs, ci-proto included, add little over HAR-RV for daily volatility, which matches Zhong, Fu & Zhu (2026).

---

## 13. Validity, limitations and open items

- **Hardware faults:** the WSL CUDA driver failed deterministically on some large inputs ("device not ready / Failed to create GPU mapping").
  - The affected tasks are listed in `_cuda_fatal_*.txt` and handled as described in PROTOCOL_CHANGES §11–13. They count as Seasonal Naive in fev-bench, and as missing configurations for Chronos-2 in GIFT-Eval.
  - The same fault silently produced error rows in E7 (ci-proto on ETTh1/Beijing in the missing-data study, `indep` mode, six TTA flags) and E5 (ci-proto at contexts ≥ 262 k).
  - These were rerun on 2026-11-26/27 (`scripts/rerun_gaps.sh`, `rerun_e5_long.sh`, `rerun_e7_missing.sh`; old checkpoints in `results/_ckpt_backup_2026112{6,7}/`). All are now complete except ci-proto at ≥ 1 M context in E5 (out of memory on 8 GB), see the update note at the top of this file.
- **Reproduction gap:** our harness reproduces most official fev-bench numbers to within 0.1 %. ci-proto is 2 % worse than its self-reported entry and TimesFM-3 is 6 % worse. On GIFT-Eval, TiRex-2 is 2.3 % worse than official.
  - Differences of ≤ 2 % between models should not be claimed as wins without the paired tests reported here.
- **Contamination:**
  - TimesFM-2.5 and TimesFM-3 on Beijing air quality are flagged; exclude them from E4, E6 and E7 Beijing comparisons.
  - The E2 synthetic suite and the 2026 prospective window are the leakage-free evidence. ci-proto is best on the former and tied for best on the latter.
- **Cost-driven settings:**
  - The large LTSF sets use stride 96, and only ci-proto has a full-history lookback there (PROTOCOL_CHANGES §9).
  - E8 uses one laptop GPU; absolute latencies are hardware-specific, and the ratios are the result to report.
- **Small post-release window:** 13 days and 299 one-day-ahead forecasts. Use it only as a sanity check.
- **Calibration:** ci-proto's 80 % intervals fail the Kupiec test in 64 % of sources. A conformal or quantile-recalibration layer is a likely reviewer request.

## 14. Suggested claims for the paper

| Claim | Supported? | Evidence |
|---|---|---|
| State-of-the-art zero-shot accuracy on leakage-free data | **Yes** (significant) | E2 Friedman–Nemenyi; skill 0.601 |
| Top accuracy on standard benchmarks | **Yes, near the top** | fev-bench: our re-runs rank 1st and 2nd. GIFT-Eval: 2nd, 1.3 % behind TimesFM-3. LTSF: best mean rank |
| Best for electricity-price forecasting | **Yes for intraday horizons and the cohort mode** | E3 hourly: 6 of 9 cells, always in the MCS. E3 daily: cohort mode best |
| Exploits ultra-long (≥ 100 k) context | **No** | E5 and E6: gains stop at ~16 k and very long histories hurt |
| Runs at million-step contexts | **Yes** | E8: 2 M points in 3.2 s, 5.7 GB |
| Uses known covariates well | **Yes (future covariates)**, **no (past-only)** | E4 |
| Well-calibrated intervals | **No** (univariate); **yes** (cohort) | §10 |
| Efficient | **Partly** | GPU latency like Chronos-2; linear in context; slow on CPU; TiRex-2 is cheaper |
