#!/usr/bin/env bash
# E3 was OOM-killed (exit 137) while concatenating the 17 daily checkpoints; it is patched to reduce
# them one model at a time. This waits for the running queue, reruns E3 (daily block resumes from its
# checkpoints, then the hourly block B), and finally runs the post-run refresh and analysis.
#   setsid nohup ./scripts/rerun_e03_then_finish.sh <driver-pid> > results/logs/full_after.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PID="${1:-}"
if [ -n "$PID" ]; then
  while kill -0 "$PID" 2>/dev/null; do sleep 300; done
fi
export PYTHONUTF8=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
mv -f results/logs/full_e03_epf.log results/logs/full_e03_epf_oom.log 2>/dev/null
start=$(date +%s)
uv run --no-sync python experiments/e03_epf.py --profile full > results/logs/full_e03_epf.log 2>&1
echo "$(date '+%F %T') e03_epf (full, rerun): exit $?, $(( ($(date +%s) - start) / 60 )) min" | tee -a results/logs/full_driver.log
exec ./scripts/after_full_run.sh
