#!/usr/bin/env bash
# Substitution prob pipeline (Codex B3 season + B1 history-only ordinal).
# Usage:
#   bash scripts/run_substitution_prob_pipeline.sh smoke
#   bash scripts/run_substitution_prob_pipeline.sh full
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

MODE="${1:-smoke}"
REF="${REF:-output/daily_candidates/prob_blend_recycle8089_ord08.csv}"

if [[ "$MODE" == "smoke" ]]; then
  PYTHONPATH=. python scripts/generate_substitution_prob_submissions.py --smoke --track both
  PATTERN="output/daily_candidates/prob_subst_*_smoke.csv"
elif [[ "$MODE" == "full" ]]; then
  PYTHONPATH=. python scripts/generate_substitution_prob_submissions.py --track both
  PATTERN="output/daily_candidates/prob_subst_*_full.csv"
else
  echo "Usage: bash scripts/run_substitution_prob_pipeline.sh [smoke|full]"
  exit 1
fi

failed=0
for f in $PATTERN; do
  [[ -f "$f" ]] || continue
  echo "=== GATE $f ==="
  if PYTHONPATH=. python scripts/gate_and_validate_submissions.py --reference "$REF" --candidate "$f"; then
    echo "PASS $f"
  else
    echo "FAIL $f"
    failed=$((failed + 1))
  fi
  echo
done
if [[ "$MODE" == "smoke" ]]; then
  echo "Smoke note: history ordinal uses 2 train windows; gate vs 8088 is diagnostic only."
fi
exit 0
