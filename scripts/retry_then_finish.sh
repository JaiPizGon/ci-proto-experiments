#!/usr/bin/env bash
# Follow-up to after_full_run.sh (2026-11-26). Its fev/E10 retries stopped every time on the same task, which
# crashes the WSL CUDA driver deterministically, so no other failed task was retried. Tasks that crash the driver
# are now recorded in _cuda_fatal_<model>.txt and skipped by the next attempt. This script waits for
# after_full_run.sh, retries until no new crash (exit 75) occurs, then rebuilds the analysis.
#   setsid nohup ./scripts/retry_then_finish.sh <after-pid> > results/logs/full_retry.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
export PYTHONUTF8=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PID="${1:-}"
if [ -n "$PID" ]; then while kill -0 "$PID" 2>/dev/null; do sleep 10; done; fi
rm -f results/logs/FULL_RUN_DONE
RUN="uv run --no-sync python"
retry() {  # retry <name> <script> [args...]: rerun while the run stops on a new CUDA-driver crash (exit 75)
  local name=$1; shift
  for attempt in $(seq 1 60); do
    mv -f "results/logs/full_${name}.log" "results/logs/full_${name}_$(date +%m%d%H%M%S).log" 2>/dev/null
    local s=$(date +%s)
    $RUN "$@" --profile full > "results/logs/full_${name}.log" 2>&1
    local code=$?
    echo "$(date '+%F %T') ${name} (skip-list retry ${attempt}): exit ${code}, $(( ($(date +%s) - s) / 60 )) min" | tee -a results/logs/full_driver.log
    [ "$code" -ne 75 ] && return 0
    sleep 30
  done
}
retry e01_benchmarks experiments/e01_benchmarks.py --only fev
retry e10_gifteval experiments/e10_gifteval.py
$RUN analysis/report.py epf --results results/E3_electricity_prices/full --reference ci-proto > results/logs/full_90_report_epf.log 2>&1
$RUN analysis/report.py bench --results results/E2_leakage_control/full --metric SQL --baseline SeasonalNaive --task-cols family series horizon_cfg > results/logs/full_91_report_bench.log 2>&1
$RUN analysis/paper.py --profile full > results/logs/full_92_paper.log 2>&1
$RUN analysis/probabilistic.py --profile full > results/logs/full_94_probabilistic.log 2>&1
$RUN analysis/summarize.py --profile full > results/logs/full_93_summary.log 2>&1
echo "$(date '+%F %T') analysis rebuilt after retries: exit $?" | tee -a results/logs/full_driver.log
touch results/logs/FULL_RUN_DONE
