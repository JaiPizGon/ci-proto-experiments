#!/usr/bin/env bash
# Resume the paper run after a shutdown: remaining experiments (checkpoints skip finished models),
# then the post-run chain (prospective refresh + analysis). E2 is complete and not re-run.
cd "$(dirname "$0")/.."
cp results/logs/full_driver.log "results/logs/full_driver_$(date +%F_%H%M).log" 2>/dev/null
setsid nohup ./run_paper.sh full e03_epf e01_benchmarks e04_covariates e05_context e06_scarcity e07_ablations \
  e09_volatility e10_gifteval e08_efficiency > results/logs/full_driver.log 2>&1 < /dev/null &
DRIVER=$!
disown
setsid nohup ./scripts/after_full_run.sh "$DRIVER" > results/logs/full_after.log 2>&1 < /dev/null &
disown
echo "resumed: driver PID $DRIVER; progress in results/logs/full_driver.log"
