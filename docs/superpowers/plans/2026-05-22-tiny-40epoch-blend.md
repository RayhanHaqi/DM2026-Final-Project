# Tiny 40-Epoch CNN Blend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate and evaluate 5% and 10% blends of the 40-epoch small CNN candidate into the current `w25` best, then recommend at most one manual submission.

**Architecture:** Reuse `scripts/blend_submissions.py` with `w25` as the reference and the 40-epoch CNN as the candidate. No model training and no Kaggle submission should be run by the assistant.

**Tech Stack:** Python, pandas, NumPy, `scripts/blend_submissions.py`, `scripts/compare_candidate_distribution.py`, `AGENTS.md`.

---

## File Structure

- No source-code changes are expected.
- Modify `AGENTS.md` after evaluation.
- Generated CSVs live under ignored `output/daily_candidates/`.

## Task 1: Confirm References

**Files:**
- No source changes expected.

- [ ] **Step 1: Verify paths exist**

Run:

```bash
python - <<'PY'
from pathlib import Path
paths = [
    Path('output/daily_candidates/cnn_gru_blend_w25_20260522_163741.csv'),
    Path('output/daily_candidates/cnn_1d_small_20260519_044615.csv'),
]
for path in paths:
    print(path, path.exists())
PY
```

Expected: both paths print `True`. Stop if either path is missing.

## Task 2: Generate Tiny Blends

**Files:**
- Generated: `output/daily_candidates/w25_40epoch_blend_w05_YYYYMMDD_HHMMSS.csv`
- Generated: `output/daily_candidates/w25_40epoch_blend_w10_YYYYMMDD_HHMMSS.csv`

- [ ] **Step 1: Generate 5% blend**

Run:

```bash
python scripts/blend_submissions.py --candidate output/daily_candidates/cnn_1d_small_20260519_044615.csv --reference output/daily_candidates/cnn_gru_blend_w25_20260522_163741.csv --candidate-weight 0.05 --name w25_40epoch_blend
```

Expected: terminal prints a saved `w25_40epoch_blend_w05_YYYYMMDD_HHMMSS.csv` path.

- [ ] **Step 2: Generate 10% blend**

Run:

```bash
python scripts/blend_submissions.py --candidate output/daily_candidates/cnn_1d_small_20260519_044615.csv --reference output/daily_candidates/cnn_gru_blend_w25_20260522_163741.csv --candidate-weight 0.10 --name w25_40epoch_blend
```

Expected: terminal prints a saved `w25_40epoch_blend_w10_YYYYMMDD_HHMMSS.csv` path.

## Task 3: Distribution Checks

**Files:**
- No source changes expected.

- [ ] **Step 1: Check 5% blend vs w25**

Run with the generated 5% path:

```bash
python scripts/compare_candidate_distribution.py --candidate output/daily_candidates/w25_40epoch_blend_w05_YYYYMMDD_HHMMSS.csv --reference output/daily_candidates/cnn_gru_blend_w25_20260522_163741.csv --min-corr 0.995 --max-mean-abs-diff 0.035 --max-week-mean-shift 0.02
```

Expected: record safe result.

- [ ] **Step 2: Check 10% blend vs w25**

Run with the generated 10% path:

```bash
python scripts/compare_candidate_distribution.py --candidate output/daily_candidates/w25_40epoch_blend_w10_YYYYMMDD_HHMMSS.csv --reference output/daily_candidates/cnn_gru_blend_w25_20260522_163741.csv --min-corr 0.995 --max-mean-abs-diff 0.035 --max-week-mean-shift 0.02
```

Expected: record safe result.

## Task 4: Selection And User-Run Submission

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: Select candidate**

If 10% passes, recommend 10%. If 10% fails but 5% passes, recommend 5%. If both fail, recommend no submission.

- [ ] **Step 2: Log results in AGENTS.md**

Record generated paths, metrics, and final recommendation.

- [ ] **Step 3: Commit AGENTS.md**

Run:

```bash
git add AGENTS.md
git commit -m "docs: log tiny 40epoch blend search"
```

- [ ] **Step 4: Stop for manual submission**

Do not run Kaggle submission commands. Tell the user the recommended CSV path, or state that no submission is recommended.

## Self-Review Notes

- Spec coverage: references, candidate weights, distribution gates, manual submission constraint, and logging are covered.
- Placeholder scan: timestamp examples use `YYYYMMDD_HHMMSS`; copy exact generated paths from terminal output during execution.
- Type consistency: `--candidate-weight` uses the 40-epoch CNN weight.
