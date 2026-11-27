#!/usr/bin/env bash
# Re-run ci-proto's E5 part A with GPU cache release before contexts > 131k; exit 75 = restart in a
# fresh process (an OOM poisons the CUDA context). Then rebuild the analysis.
cd "$(dirname "$0")/.."
log() { echo "$(date '+%F %T') $*" >> results/logs/full_driver.log; }
for i in 1 2 3 4; do
  t0=$(date +%s)
  uv run --no-sync python experiments/e05_context.py --profile full > results/logs/full_e05_context_long_$i.log 2>&1
  rc=$?
  log "e05_context (long-context rerun, try $i): exit $rc, $(( ($(date +%s)-t0)/60 )) min"
  [ $rc -ne 75 ] && break
done
for s in analysis/paper.py analysis/probabilistic.py analysis/summarize.py; do
  uv run --no-sync python $s --profile full >> results/logs/full_e05_context_long_analysis.log 2>&1
done
log "analysis rebuilt after E5 long rerun: exit $?"
touch results/logs/E5LONG_DONE
