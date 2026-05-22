# Higher GRU Blend Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate and evaluate higher CNN-GRU blend weights beyond the public-winning `w20`, then choose at most one final submission candidate.

**Architecture:** Reuse `scripts/blend_submissions.py` and `scripts/compare_candidate_distribution.py`. This plan is CPU-only: it blends existing CSV files and compares distributions against both the old CNN best and the new `w20` public best.

**Tech Stack:** Python, pandas, NumPy, existing blend and distribution scripts, `AGENTS.md` experiment log.

---

## File Structure

- No source-code changes are expected.
- Modify `AGENTS.md` after generating and evaluating candidates.
- Generated CSVs live under ignored `output/daily_candidates/`.

## Task 1: Confirm References

**Files:**
- Read: `AGENTS.md`

- [ ] **Step 1: Verify fixed reference paths exist**

Run:

```bash
python - <<'PY'
from pathlib import Path
paths = [
    Path('output/daily_candidates/cnn_1d_20260518_180837.csv'),
    Path('output/daily_candidates/cnn_1d_cnn_gru_20260522_154103.csv'),
    Path('output/daily_candidates/cnn_gru_blend_w20_20260522_160258.csv'),
]
for path in paths:
    print(path, path.exists())
PY
```

Expected output:

```text
output/daily_candidates/cnn_1d_20260518_180837.csv True
output/daily_candidates/cnn_1d_cnn_gru_20260522_154103.csv True
output/daily_candidates/cnn_gru_blend_w20_20260522_160258.csv True
```

Stop if any path prints `False`.

## Task 2: Generate Initial Higher Blends

**Files:**
- Generated: `output/daily_candidates/cnn_gru_blend_w22_YYYYMMDD_HHMMSS.csv`
- Generated: `output/daily_candidates/cnn_gru_blend_w24_YYYYMMDD_HHMMSS.csv`
- Generated: `output/daily_candidates/cnn_gru_blend_w25_YYYYMMDD_HHMMSS.csv`

- [ ] **Step 1: Generate w22**

Run:

```bash
python scripts/blend_submissions.py --candidate output/daily_candidates/cnn_1d_cnn_gru_20260522_154103.csv --reference output/daily_candidates/cnn_1d_20260518_180837.csv --candidate-weight 0.22
```

Expected: terminal prints a saved `cnn_gru_blend_w22_YYYYMMDD_HHMMSS.csv` path.

- [ ] **Step 2: Generate w24**

Run:

```bash
python scripts/blend_submissions.py --candidate output/daily_candidates/cnn_1d_cnn_gru_20260522_154103.csv --reference output/daily_candidates/cnn_1d_20260518_180837.csv --candidate-weight 0.24
```

Expected: terminal prints a saved `cnn_gru_blend_w24_YYYYMMDD_HHMMSS.csv` path.

- [ ] **Step 3: Generate w25**

Run:

```bash
python scripts/blend_submissions.py --candidate output/daily_candidates/cnn_1d_cnn_gru_20260522_154103.csv --reference output/daily_candidates/cnn_1d_20260518_180837.csv --candidate-weight 0.25
```

Expected: terminal prints a saved `cnn_gru_blend_w25_YYYYMMDD_HHMMSS.csv` path.

## Task 3: Compare Initial Blends Against Old Best

**Files:**
- No source changes expected.

- [ ] **Step 1: Check w22 vs old best**

Run with the generated w22 path:

```bash
python scripts/compare_candidate_distribution.py --candidate output/daily_candidates/cnn_gru_blend_w22_YYYYMMDD_HHMMSS.csv --reference output/daily_candidates/cnn_1d_20260518_180837.csv
```

Expected: record correlation, mean_abs_diff, max_week_mean_shift, prediction_range, and safe.

- [ ] **Step 2: Check w24 vs old best**

Run with the generated w24 path:

```bash
python scripts/compare_candidate_distribution.py --candidate output/daily_candidates/cnn_gru_blend_w24_YYYYMMDD_HHMMSS.csv --reference output/daily_candidates/cnn_1d_20260518_180837.csv
```

Expected: record correlation, mean_abs_diff, max_week_mean_shift, prediction_range, and safe.

- [ ] **Step 3: Check w25 vs old best**

Run with the generated w25 path:

```bash
python scripts/compare_candidate_distribution.py --candidate output/daily_candidates/cnn_gru_blend_w25_YYYYMMDD_HHMMSS.csv --reference output/daily_candidates/cnn_1d_20260518_180837.csv
```

Expected: record correlation, mean_abs_diff, max_week_mean_shift, prediction_range, and safe.

## Task 4: Compare Initial Blends Against w20

**Files:**
- No source changes expected.

- [ ] **Step 1: Check w22 vs w20**

Run with the generated w22 path:

```bash
python scripts/compare_candidate_distribution.py --candidate output/daily_candidates/cnn_gru_blend_w22_YYYYMMDD_HHMMSS.csv --reference output/daily_candidates/cnn_gru_blend_w20_20260522_160258.csv --min-corr 0.995 --max-mean-abs-diff 0.035 --max-week-mean-shift 0.02
```

Expected: record safe result using incremental thresholds.

- [ ] **Step 2: Check w24 vs w20**

Run with the generated w24 path:

```bash
python scripts/compare_candidate_distribution.py --candidate output/daily_candidates/cnn_gru_blend_w24_YYYYMMDD_HHMMSS.csv --reference output/daily_candidates/cnn_gru_blend_w20_20260522_160258.csv --min-corr 0.995 --max-mean-abs-diff 0.035 --max-week-mean-shift 0.02
```

Expected: record safe result using incremental thresholds.

- [ ] **Step 3: Check w25 vs w20**

Run with the generated w25 path:

```bash
python scripts/compare_candidate_distribution.py --candidate output/daily_candidates/cnn_gru_blend_w25_YYYYMMDD_HHMMSS.csv --reference output/daily_candidates/cnn_gru_blend_w20_20260522_160258.csv --min-corr 0.995 --max-mean-abs-diff 0.035 --max-week-mean-shift 0.02
```

Expected: record safe result using incremental thresholds.

## Task 5: Optional Boundary Blends

**Files:**
- Generated only if w25 remains plausible.

- [ ] **Step 1: Decide whether w25 is plausible**

Generate `w27` and `w30` only if w25 satisfies all of these:

```text
correlation vs old best >= 0.975
mean_abs_diff vs old best <= 0.13
max_week_mean_shift vs old best <= 0.055
correlation vs w20 >= 0.995
mean_abs_diff vs w20 <= 0.035
```

Expected: skip Task 5 steps 2-5 if w25 fails these checks.

- [ ] **Step 2: Generate w27**

Run only if Step 1 passes:

```bash
python scripts/blend_submissions.py --candidate output/daily_candidates/cnn_1d_cnn_gru_20260522_154103.csv --reference output/daily_candidates/cnn_1d_20260518_180837.csv --candidate-weight 0.27
```

Expected: terminal prints a saved `cnn_gru_blend_w27_YYYYMMDD_HHMMSS.csv` path.

- [ ] **Step 3: Generate w30**

Run only if Step 1 passes:

```bash
python scripts/blend_submissions.py --candidate output/daily_candidates/cnn_1d_cnn_gru_20260522_154103.csv --reference output/daily_candidates/cnn_1d_20260518_180837.csv --candidate-weight 0.30
```

Expected: terminal prints a saved `cnn_gru_blend_w30_YYYYMMDD_HHMMSS.csv` path.

- [ ] **Step 4: Check w27 distributions**

Run both comparisons with the generated w27 path:

```bash
python scripts/compare_candidate_distribution.py --candidate output/daily_candidates/cnn_gru_blend_w27_YYYYMMDD_HHMMSS.csv --reference output/daily_candidates/cnn_1d_20260518_180837.csv
python scripts/compare_candidate_distribution.py --candidate output/daily_candidates/cnn_gru_blend_w27_YYYYMMDD_HHMMSS.csv --reference output/daily_candidates/cnn_gru_blend_w20_20260522_160258.csv --min-corr 0.995 --max-mean-abs-diff 0.035 --max-week-mean-shift 0.02
```

Expected: record old-best and w20-relative metrics.

- [ ] **Step 5: Check w30 distributions**

Run both comparisons with the generated w30 path:

```bash
python scripts/compare_candidate_distribution.py --candidate output/daily_candidates/cnn_gru_blend_w30_YYYYMMDD_HHMMSS.csv --reference output/daily_candidates/cnn_1d_20260518_180837.csv
python scripts/compare_candidate_distribution.py --candidate output/daily_candidates/cnn_gru_blend_w30_YYYYMMDD_HHMMSS.csv --reference output/daily_candidates/cnn_gru_blend_w20_20260522_160258.csv --min-corr 0.995 --max-mean-abs-diff 0.035 --max-week-mean-shift 0.02
```

Expected: record old-best and w20-relative metrics.

## Task 6: Select One Submission Candidate

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: Apply decision rule**

Select the highest generated weight satisfying all of these:

```text
correlation vs old best >= 0.975
mean_abs_diff vs old best <= 0.13
max_week_mean_shift vs old best <= 0.055
correlation vs w20 >= 0.995
mean_abs_diff vs w20 <= 0.035
prediction range within [0, 5]
```

Expected: if no higher blend passes, submit nothing and keep w20 as best.

- [ ] **Step 2: Log results in AGENTS.md**

Add a section with:

```text
generated blend path
weight
metrics vs old best
metrics vs w20
selected/rejected decision
submission recommendation
```

- [ ] **Step 3: Commit AGENTS.md**

Run:

```bash
git add AGENTS.md
git commit -m "docs: log higher gru blend search"
```

## Task 7: Verification

**Files:**
- No source changes expected.

- [ ] **Step 1: Check working tree**

Run:

```bash
git status --short
```

Expected: clean after committing `AGENTS.md`, except ignored `output/` files.

## Self-Review Notes

- Spec coverage: candidate weights, dual-reference comparisons, optional boundary candidates, selection rule, and logging are covered.
- Placeholder scan: timestamp examples use `YYYYMMDD_HHMMSS` because filenames are created at runtime; exact generated paths must be copied from terminal output.
- Type consistency: blend script uses `--candidate-weight`; distribution script uses existing threshold flags.
