# Fine-Grained GRU Blend Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate and evaluate `w25.5` and `w26.5` GRU blends, then recommend at most one manual Kaggle submission.

**Architecture:** Reuse the existing CPU-only blend and distribution scripts. No model training and no Kaggle submission should be run by the assistant.

**Tech Stack:** Python, pandas, NumPy, `scripts/blend_submissions.py`, `scripts/compare_candidate_distribution.py`, `AGENTS.md`.

---

## File Structure

- No source-code changes are expected.
- Modify `AGENTS.md` after evaluation.
- Generated CSVs live under ignored `output/daily_candidates/`.

## Task 1: Generate Fractional Blends

**Files:**
- Generated: `output/daily_candidates/cnn_gru_blend_w26_YYYYMMDD_HHMMSS.csv` for 25.5 due script rounding.
- Generated: `output/daily_candidates/cnn_gru_blend_w26_YYYYMMDD_HHMMSS.csv` for 26.5 due script rounding.

- [ ] **Step 1: Generate w25.5 with explicit name**

Run:

```bash
python scripts/blend_submissions.py --candidate output/daily_candidates/cnn_1d_cnn_gru_20260522_154103.csv --reference output/daily_candidates/cnn_1d_20260518_180837.csv --candidate-weight 0.255 --name cnn_gru_blend_w25p5
```

Expected: terminal prints a saved path beginning with `output/daily_candidates/cnn_gru_blend_w25p5_`.

- [ ] **Step 2: Generate w26.5 with explicit name**

Run:

```bash
python scripts/blend_submissions.py --candidate output/daily_candidates/cnn_1d_cnn_gru_20260522_154103.csv --reference output/daily_candidates/cnn_1d_20260518_180837.csv --candidate-weight 0.265 --name cnn_gru_blend_w26p5
```

Expected: terminal prints a saved path beginning with `output/daily_candidates/cnn_gru_blend_w26p5_`.

## Task 2: Compare Against Old Best

**Files:**
- No source changes expected.

- [ ] **Step 1: Check w25.5 vs old best**

Run with the generated w25.5 path:

```bash
python scripts/compare_candidate_distribution.py --candidate output/daily_candidates/cnn_gru_blend_w25p5_w26_YYYYMMDD_HHMMSS.csv --reference output/daily_candidates/cnn_1d_20260518_180837.csv
```

Expected: record correlation, mean_abs_diff, max_week_mean_shift, prediction_range.

- [ ] **Step 2: Check w26.5 vs old best**

Run with the generated w26.5 path:

```bash
python scripts/compare_candidate_distribution.py --candidate output/daily_candidates/cnn_gru_blend_w26p5_w26_YYYYMMDD_HHMMSS.csv --reference output/daily_candidates/cnn_1d_20260518_180837.csv
```

Expected: record correlation, mean_abs_diff, max_week_mean_shift, prediction_range.

## Task 3: Compare Against Current Best Plateau

**Files:**
- No source changes expected.

- [ ] **Step 1: Check w25.5 vs w25**

Run with the generated w25.5 path:

```bash
python scripts/compare_candidate_distribution.py --candidate output/daily_candidates/cnn_gru_blend_w25p5_w26_YYYYMMDD_HHMMSS.csv --reference output/daily_candidates/cnn_gru_blend_w25_20260522_163741.csv --min-corr 0.9998 --max-mean-abs-diff 0.010 --max-week-mean-shift 0.005
```

Expected: record whether incremental drift from w25 is safe.

- [ ] **Step 2: Check w26.5 vs w25**

Run with the generated w26.5 path:

```bash
python scripts/compare_candidate_distribution.py --candidate output/daily_candidates/cnn_gru_blend_w26p5_w26_YYYYMMDD_HHMMSS.csv --reference output/daily_candidates/cnn_gru_blend_w25_20260522_163741.csv --min-corr 0.9998 --max-mean-abs-diff 0.010 --max-week-mean-shift 0.005
```

Expected: record whether incremental drift from w25 is safe.

## Task 4: Selection And User-Run Submission

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: Apply decision rule**

Select `w26.5` if it passes all gates. Otherwise select `w25.5` if it passes all gates. If neither passes, select no candidate.

- [ ] **Step 2: Log results in AGENTS.md**

Record each generated path, metrics vs old best, metrics vs w25, and final recommendation.

- [ ] **Step 3: Commit AGENTS.md**

Run:

```bash
git add AGENTS.md
git commit -m "docs: log fine-grained gru blend search"
```

- [ ] **Step 4: Stop for manual submission**

Do not run Kaggle submission commands. Tell the user the single recommended CSV path, or state that no submission is recommended.

## Task 5: Verification

**Files:**
- No source changes expected.

- [ ] **Step 1: Check working tree**

Run:

```bash
git status --short
```

Expected: clean after committing `AGENTS.md`, except ignored `output/` files.

## Self-Review Notes

- Spec coverage: fractional candidates, references, thresholds, manual submission constraint, and logging are covered.
- Placeholder scan: timestamp examples use `YYYYMMDD_HHMMSS`; copy exact generated paths from terminal output during execution.
- Type consistency: explicit `--name` avoids ambiguous integer rounding in output names.
