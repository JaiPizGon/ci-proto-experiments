#!/usr/bin/env bash
# Runs once the full queue (run_paper.sh full) has finished: refreshes the prospective window with the
# newest Ember days (E2 Part B, after every other experiment so no experiment sees a changed data file),
# then rebuilds every analysis output so tables and figures include the refreshed window.
#   setsid nohup ./scripts/after_full_run.sh <driver-pid> > results/logs/full_after.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PID="${1:-}"
if [ -n "$PID" ]; then
  while kill -0 "$PID" 2>/dev/null; do sleep 300; done
fi
RUN="uv run --no-sync python"
# Retries. Failed fev/GIFT tasks are not saved, so a rerun skips finished work and retries only failures:
# Moirai-2 pandas-2.2 frequency aliases (fixed in envs/moirai/worker.py) and WSL CUDA driver faults
# ("device not ready", then !handles_.at(i) asserts for every later task). fev and E10 now exit 75 on such a
# fault; each retry is a fresh process. E3 and the large LTSF sets are rerun only if their last exit was non-zero.
retry() {  # retry <name> <script> [args...]: up to 4 attempts until exit 0
  local name=$1; shift
  for attempt in 1 2 3 4; do
    mv -f "results/logs/full_${name}.log" "results/logs/full_${name}_$(date +%m%d%H%M%S).log" 2>/dev/null
    local s=$(date +%s)
    $RUN "$@" --profile full > "results/logs/full_${name}.log" 2>&1
    local code=$?
    echo "$(date '+%F %T') ${name} (retry ${attempt}): exit ${code}, $(( ($(date +%s) - s) / 60 )) min" | tee -a results/logs/full_driver.log
    [ "$code" -eq 0 ] && return 0
    sleep 60
  done
}
last_exit() { grep -E "^[0-9-]+ [0-9:]+ $1 \(full" results/logs/full_driver.log | tail -1 | grep -oP "exit \K[0-9]+"; }
retry e01_benchmarks experiments/e01_benchmarks.py --only fev
retry e10_gifteval experiments/e10_gifteval.py
[ "$(last_exit e01_ltsf_large)" != "0" ] && retry e01_ltsf_large experiments/e01_benchmarks.py --only ltsf_large
[ "$(last_exit e03_epf)" != "0" ] && retry e03_epf experiments/e03_epf.py
echo "$(date '+%F %T') queue finished; refreshing prospective window"
./scripts/refresh_prospective.sh full; echo "$(date '+%F %T') refresh: exit $?"
$RUN analysis/report.py epf --results results/E3_electricity_prices/full --reference ci-proto > results/logs/full_90_report_epf.log 2>&1
$RUN analysis/report.py bench --results results/E2_leakage_control/full --metric SQL --baseline SeasonalNaive --task-cols family series horizon_cfg > results/logs/full_91_report_bench.log 2>&1
$RUN analysis/paper.py --profile full > results/logs/full_92_paper.log 2>&1
$RUN analysis/probabilistic.py --profile full > results/logs/full_94_probabilistic.log 2>&1
$RUN analysis/summarize.py --profile full > results/logs/full_93_summary.log 2>&1
echo "$(date '+%F %T') analysis rebuilt: exit $?"
touch results/logs/FULL_RUN_DONE
