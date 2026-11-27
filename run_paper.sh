#!/usr/bin/env bash
# Publication protocol runner (Linux + CUDA). Profiles: smoke (minutes, every model/code path) | full.
#   ./run_paper.sh smoke                 # validate wrappers end to end
#   ./run_paper.sh full                  # the paper run (days on an 8 GB laptop GPU)
#   ./run_paper.sh full e03_epf e05_context   # a subset, in the given order
# Environment (once):  uv sync --extra chronos --extra stats --extra fm --extra bench --extra neural
#                      (cd envs/moirai && uv sync)       # Moirai-2 lives in its own env (torch 2.4)
set -u
PROFILE="${1:-smoke}"; shift || true
cd "$(dirname "$0")"
export PYTHONUTF8=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
mkdir -p results/logs
RUN="uv run --no-sync python"
EXPS=("$@")
[ ${#EXPS[@]} -eq 0 ] && EXPS=(e02_leakage e03_epf e04_covariates e05_context e06_scarcity e07_ablations e08_efficiency e09_volatility e01_benchmarks)

$RUN scripts/get_data.py ember ember_hourly household beijing ett bike fred > results/logs/01_data.log 2>&1 || echo "[warn] some datasets missing (see results/logs/01_data.log)"
$RUN -m pytest -q > results/logs/02_tests.log 2>&1; echo "tests: $(tail -1 results/logs/02_tests.log)"

for e in "${EXPS[@]}"; do
  start=$(date +%s)
  $RUN "experiments/$e.py" --profile "$PROFILE" > "results/logs/${PROFILE}_$e.log" 2>&1
  code=$?
  echo "$(date '+%F %T') $e ($PROFILE): exit $code, $(( ($(date +%s) - start) / 60 )) min"
done

$RUN analysis/report.py epf --results "results/E3_electricity_prices/$PROFILE" --reference "ci-proto" > "results/logs/${PROFILE}_90_report_epf.log" 2>&1
$RUN analysis/report.py bench --results "results/E2_leakage_control/$PROFILE" --metric SQL --baseline SeasonalNaive --task-cols family series horizon_cfg > "results/logs/${PROFILE}_91_report_bench.log" 2>&1
$RUN analysis/paper.py --profile "$PROFILE" > "results/logs/${PROFILE}_92_paper.log" 2>&1
$RUN analysis/probabilistic.py --profile "$PROFILE" > "results/logs/${PROFILE}_94_probabilistic.log" 2>&1
$RUN analysis/summarize.py --profile "$PROFILE" > "results/logs/${PROFILE}_93_summary.log" 2>&1
echo "Done. results/SUMMARY_$PROFILE.md, figures in results/figures/$PROFILE"
