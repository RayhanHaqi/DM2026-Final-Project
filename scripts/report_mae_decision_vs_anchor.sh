#!/usr/bin/env bash
# Distribution stats vs anchor (median/quantile will NOT pass strict gate — expected).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

REF="${REF:-output/daily_candidates/prob_blend_recycle8089_ord08.csv}"
PATTERN="${1:-output/daily_candidates/prob_mae_*_8089_ord08_full.csv}"

for f in $PATTERN; do
  [[ -f "$f" ]] || continue
  echo "=== $f ==="
  PYTHONPATH=. python scripts/compare_candidate_distribution.py --candidate "$f" --reference "$REF" || true
  echo
done
