#!/usr/bin/env bash
# Read-only: show why non-mean decoders fail on 92/8 caches.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

PYTHONPATH=. python scripts/diagnose_prob_cache_decoders.py "$@"
