#!/usr/bin/env bash
# Remaining full-profile queue after the 2026-11-18 restart (E1 gained per-cell checkpoints; E3 was
# OOM-killed and patched). Order: E1 (ETT + fev-bench), E4-E10, E8, the E3 rerun, the five large LTSF
# datasets (E1 part ltsf_large, days of GPU time), then the prospective refresh and the analysis rebuild.
#   setsid nohup ./scripts/resume_queue.sh > results/logs/full_queue.log 2>&1 < /dev/null & disown
set -u
cd "$(dirname "$0")/.."
export PYTHONUTF8=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # less fragmentation on the 8 GB GPU
RUN="uv run --no-sync python"
step() {  # step <log-name> <script> [args...]
  local name="$1"; shift
  local start; start=$(date +%s)
  [ -f "results/logs/full_${name}.log" ] && mv -f "results/logs/full_${name}.log" "results/logs/full_${name}_$(date +%m%d%H%M).log"
  $RUN "$@" --profile full > "results/logs/full_${name}.log" 2>&1
  local code=$?  # read before any $(...) below resets $?
  echo "$(date '+%F %T') ${name} (full): exit ${code}, $(( ($(date +%s) - start) / 60 )) min" | tee -a results/logs/full_driver.log
}
# Done: E1 ETT + fev, E2-E10 (E3 finished 2026-11-25; fev/E10 failures are retried in after_full_run.sh).
step e01_ltsf_large experiments/e01_benchmarks.py --only ltsf_large
exec ./scripts/after_full_run.sh
