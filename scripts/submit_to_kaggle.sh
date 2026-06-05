#!/usr/bin/env bash
# Submit with duplicate guard + ledger record.
# Usage:
#   bash scripts/submit_to_kaggle.sh output/daily_candidates/foo.csv "description"
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

FILE="${1:?usage: submit_to_kaggle.sh <csv> [description]}"
DESC="${2:-}"
COMP="${KAGGLE_COMP:-data-mining-2026-final-project}"

if [[ ! -f "$FILE" ]]; then
  echo "Missing file: $FILE" >&2
  exit 1
fi

PYTHONPATH=. python scripts/kaggle_submission_ledger.py check "$FILE"

REF="${REF:-output/daily_candidates/prob_blend_recycle8089_ord08.csv}"
if [[ -f "$REF" ]]; then
  echo "=== distribution gate vs $REF ==="
  PYTHONPATH=. python scripts/gate_and_validate_submissions.py \
    --reference "$REF" \
    --candidate "$FILE" || {
    echo "Gate FAILED — not submitting. Fix candidate or override gate manually." >&2
    exit 1
  }
fi

kaggle competitions submit -c "$COMP" -f "$FILE" -m "${DESC:-submission}"
PYTHONPATH=. python scripts/kaggle_submission_ledger.py record "$FILE" --description "$DESC"
echo "Submitted and recorded: $FILE"
