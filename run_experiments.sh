#!/usr/bin/env bash
# Runs the ci-proto experiment suite with uv on Linux/macOS/WSL.
#   ./run_experiments.sh            # laptop profile (CPU)
#   ./run_experiments.sh full       # full protocol (GPU machine)
set -u
PROFILE="${1:-laptop}"
cd "$(dirname "$0")"
export PYTHONUTF8=1 PYTHONUNBUFFERED=1
command -v uv >/dev/null || { echo "uv not installed: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
mkdir -p results/logs

echo "=== uv sync ==="
uv sync --extra chronos 2>&1 | tee results/logs/00_sync.log
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "uv sync failed"; exit 1; }
uv run --no-sync python scripts/get_data.py 2>&1 | tee results/logs/01_data.log
uv run --no-sync pytest -q 2>&1 | tee results/logs/02_tests.log

for e in e02_leakage e04_covariates e03_epf e06_scarcity e05_context e07_ablations e08_efficiency e01_benchmarks; do
  echo "=== $e ($PROFILE) ==="
  start=$(date +%s)
  uv run --no-sync python "experiments/$e.py" --profile "$PROFILE" 2>&1 | tee "results/logs/$e.log"
  echo "    $e finished in $(( ($(date +%s) - start) / 60 )) min (exit ${PIPESTATUS[0]})"
done

uv run --no-sync python analysis/report.py epf --results "results/E3_electricity_prices/$PROFILE" --reference "ci-proto[joint]" 2>&1 | tee results/logs/90_report_epf.log
uv run --no-sync python analysis/report.py bench --results "results/E2_leakage_control/$PROFILE" --metric SQL --baseline SeasonalNaive 2>&1 | tee results/logs/91_report_bench.log
uv run --no-sync python analysis/summarize.py --profile "$PROFILE" 2>&1 | tee results/logs/92_summary.log
echo "Done. Open results/SUMMARY_$PROFILE.md"
