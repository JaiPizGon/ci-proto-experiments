#!/usr/bin/env bash
# Extend E2 Part B (temporal hold-out 2026 + post-release window after 2026-11-12) with the newest Ember
# daily prices, then recompute the probabilistic tests. Run it shortly before submission:
#   ./scripts/refresh_prospective.sh            # full profile
# Part B checkpoints are keyed by the data end date, so earlier windows are never overwritten.
set -eu
cd "$(dirname "$0")/.."
PROFILE="${1:-full}"
uv run --no-sync python scripts/get_data.py ember --refresh
uv run --no-sync python experiments/e02_leakage.py --profile "$PROFILE" --only B
uv run --no-sync python analysis/probabilistic.py --profile "$PROFILE"
echo "Prospective window refreshed: results/E2_leakage_control/$PROFILE/prospective_summary.csv"
