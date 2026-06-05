#!/usr/bin/env bash
# OOF temperature-calibrated hybrid ordinal + 92/8 prob blend candidate.
# Usage:
#   bash scripts/run_oof_ordinal_calibration_pipeline.sh smoke
#   bash scripts/run_oof_ordinal_calibration_pipeline.sh full
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

MODE="${1:-smoke}"
REF="${REF:-output/daily_candidates/prob_blend_recycle8089_ord08.csv}"
SOFT_CACHE="${SOFT_CACHE:-output/prob_cache/soft_prob_best8089.npz}"
ORD_CAL="${ORD_CAL:-output/prob_cache/ordinal_hybrid_oofcal_best.npz}"
ORD_W="${ORD_W:-0.08}"
SOFT_W="${SOFT_W:-0.92}"

if [[ "$MODE" == "smoke" ]]; then
  PYTHONPATH=. python scripts/cache_ordinal_probabilities_calibrated.py --smoke
  OUT="output/daily_candidates/prob_blend_8089_oofcal_ord08_smoke.csv"
elif [[ "$MODE" == "full" ]]; then
  PYTHONPATH=. python scripts/cache_ordinal_probabilities_calibrated.py
  OUT="output/daily_candidates/prob_blend_8089_oofcal_ord08_full.csv"
else
  echo "Usage: bash scripts/run_oof_ordinal_calibration_pipeline.sh [smoke|full]"
  exit 1
fi

PYTHONPATH=. python scripts/blend_prob_submissions.py \
  --cache "${SOFT_CACHE}:${SOFT_W}" \
  --cache "${ORD_CAL}:${ORD_W}" \
  --output-path "$OUT"

echo "=== GATE $OUT vs $REF ==="
if [[ "$MODE" == "smoke" ]]; then
  PYTHONPATH=. python scripts/gate_and_validate_submissions.py \
    --reference "$REF" \
    --candidate "$OUT" || echo "smoke: gate often fails (2 windows); check OOF MAE above"
else
  PYTHONPATH=. python scripts/gate_and_validate_submissions.py \
    --reference "$REF" \
    --candidate "$OUT"
fi

echo "=== decoder diagnostic (mean only) ==="
PYTHONPATH=. python scripts/diagnose_prob_cache_decoders.py \
  --cache "${SOFT_CACHE}:${SOFT_W}" \
  --cache "${ORD_CAL}:${ORD_W}"

echo "Saved: $OUT"
