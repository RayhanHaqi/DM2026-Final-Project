#!/usr/bin/env bash
# MAE decision rule on existing 92/8 prob caches (median / quantile).
# Non-mean decoders are INVALID on soft caches — use run_prob_decoder_diagnostic.sh first.
# Usage:
#   bash scripts/run_mae_decision_pipeline.sh smoke
#   bash scripts/run_mae_decision_pipeline.sh full
#   ALLOW_INVALID_DECODER=1 bash scripts/run_mae_decision_pipeline.sh full  # not for Kaggle
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

MODE="${1:-smoke}"
REF="${REF:-output/daily_candidates/prob_blend_recycle8089_ord08.csv}"

EXTRA=()
if [[ "${ALLOW_INVALID_DECODER:-}" == "1" ]]; then
  EXTRA+=(--allow-invalid-decoder)
else
  echo "Note: median/quantile require ALLOW_INVALID_DECODER=1 (see diagnose_prob_cache_decoders.py)"
  EXTRA+=(--decision mean)
fi

if [[ "$MODE" == "smoke" ]]; then
  PYTHONPATH=. python scripts/generate_mae_decision_submissions.py --smoke "${EXTRA[@]}"
  PATTERN="output/daily_candidates/prob_mae_*_8089_ord08_smoke.csv"
elif [[ "$MODE" == "full" ]]; then
  PYTHONPATH=. python scripts/generate_mae_decision_submissions.py "${EXTRA[@]}"
  PATTERN="output/daily_candidates/prob_mae_*_8089_ord08_full.csv"
else
  echo "Usage: bash scripts/run_mae_decision_pipeline.sh [smoke|full]"
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

echo "Done. failed=$failed"
