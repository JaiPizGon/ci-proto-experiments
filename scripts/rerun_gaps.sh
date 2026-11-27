#!/usr/bin/env bash
# 2026-11-26: rerun the ci-proto cells that the WSL CUDA driver fault turned into errors after the main run:
# E7 (missing-data study on ETTh1/Beijing, indep mode, six TTA flags) and E5 (ci-proto contexts >= 262k).
# Old checkpoints are in results/_ckpt_backup_20261126/. Then rebuild the analysis.
set -u
cd "$(dirname "$0")/.."
export PYTHONUTF8=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN="uv run --no-sync python"
for e in e07_ablations e05_context; do
  mv -f results/logs/full_$e.log results/logs/full_${e}_$(date +%m%d%H%M).log 2>/dev/null
  s=$(date +%s); $RUN experiments/$e.py --profile full > results/logs/full_$e.log 2>&1; code=$?
  echo "$(date '+%F %T') $e (gap rerun): exit $code, $(( ($(date +%s)-s)/60 )) min" | tee -a results/logs/full_driver.log
done
$RUN analysis/paper.py --profile full > results/logs/full_92_paper.log 2>&1
$RUN analysis/probabilistic.py --profile full > results/logs/full_94_probabilistic.log 2>&1
$RUN analysis/summarize.py --profile full > results/logs/full_93_summary.log 2>&1
echo "$(date '+%F %T') analysis rebuilt after gap rerun: exit $?" | tee -a results/logs/full_driver.log
touch results/logs/GAPS_DONE
