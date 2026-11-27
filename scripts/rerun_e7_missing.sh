#!/usr/bin/env bash
# After the E5 long rerun: recompute ci-proto missing-data on ETTh1/Beijing (per-dataset checkpoints),
# restarting in a fresh process after a WSL CUDA fault (exit 75), then rebuild the analysis.
cd "$(dirname "$0")/.."
log() { echo "$(date '+%F %T') $*" >> results/logs/full_driver.log; }
while [ ! -f results/logs/E5LONG_DONE ]; do sleep 30; done
for i in 1 2 3 4 5 6; do
  t0=$(date +%s)
  uv run --no-sync python experiments/e07_ablations.py --profile full > results/logs/full_e07_missing_$i.log 2>&1
  rc=$?
  log "e07_ablations (missing per-dataset, try $i): exit $rc, $(( ($(date +%s)-t0)/60 )) min"
  [ $rc -ne 75 ] && break
done
for s in analysis/paper.py analysis/probabilistic.py analysis/summarize.py; do
  uv run --no-sync python $s --profile full >> results/logs/full_e07_missing_analysis.log 2>&1
done
log "analysis rebuilt after E7 missing rerun: exit $?"
touch results/logs/E7MISS_DONE
