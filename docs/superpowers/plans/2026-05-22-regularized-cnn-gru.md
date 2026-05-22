# Regularized CNN-GRU Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run and evaluate three regularized CNN-GRU training variants to see whether standalone GRU can become distribution-safe.

**Architecture:** Reuse the existing CNN-GRU model and CNN submission runner. This is an experiment-only plan with no source-code changes unless a run exposes a bug.

**Tech Stack:** Python, PyTorch, existing `scripts/generate_cnn_submission.py`, existing `scripts/run_temporal_backtest.py`, existing `scripts/compare_candidate_distribution.py`.

---

## File Structure

- Modify `AGENTS.md`: record each variant command, output path, validation MAE, distribution metrics, optional backtest MAE, and decision.
- No source files should be modified for this plan.

## Task 1: Confirm Baseline Context

**Files:**
- Read: `AGENTS.md`

- [ ] **Step 1: Confirm current-best and GRU candidate references**

Use these fixed reference paths:

```text
current best: output/daily_candidates/cnn_1d_20260518_180837.csv
unsafe GRU: output/daily_candidates/cnn_1d_cnn_gru_20260522_154103.csv
```

Expected context:

```text
current-best public MAE: 0.8222
unsafe GRU backtest MAE: 0.184923
unsafe GRU distribution: corr=0.643475, diff=0.495751, safe=False
```

## Task 2: Run Regularized GRU Variant A

**Files:**
- Generated output: `output/daily_candidates/`

- [ ] **Step 1: Generate candidate**

Run:

```bash
python scripts/generate_cnn_submission.py --model cnn_gru --no-backtest --max-windows-per-region 52 --epochs 25 --seed 42 --dropout 0.25 --weight-decay 0.001 --batch-size 128
```

Expected: terminal prints epoch validation MAE and a `Saved submission:` path.

- [ ] **Step 2: Distribution check**

Run with the generated path from Step 1:

```bash
python scripts/compare_candidate_distribution.py --candidate output/daily_candidates/cnn_1d_cnn_gru_YYYYMMDD_HHMMSS.csv --reference output/daily_candidates/cnn_1d_20260518_180837.csv
```

Expected: record correlation, mean_abs_diff, max_week_mean_shift, candidate week means, and `safe`.

## Task 3: Run Regularized GRU Variant B

**Files:**
- Generated output: `output/daily_candidates/`

- [ ] **Step 1: Generate candidate**

Run:

```bash
python scripts/generate_cnn_submission.py --model cnn_gru --no-backtest --max-windows-per-region 52 --epochs 25 --seed 42 --dropout 0.30 --weight-decay 0.002 --batch-size 128
```

Expected: terminal prints epoch validation MAE and a `Saved submission:` path.

- [ ] **Step 2: Distribution check**

Run with the generated path from Step 1:

```bash
python scripts/compare_candidate_distribution.py --candidate output/daily_candidates/cnn_1d_cnn_gru_YYYYMMDD_HHMMSS.csv --reference output/daily_candidates/cnn_1d_20260518_180837.csv
```

Expected: record correlation, mean_abs_diff, max_week_mean_shift, candidate week means, and `safe`.

## Task 4: Run Regularized GRU Variant C

**Files:**
- Generated output: `output/daily_candidates/`

- [ ] **Step 1: Generate candidate**

Run:

```bash
python scripts/generate_cnn_submission.py --model cnn_gru --no-backtest --max-windows-per-region 52 --epochs 25 --seed 42 --dropout 0.25 --weight-decay 0.002 --lr 0.0005 --scheduler --batch-size 128
```

Expected: terminal prints epoch validation MAE and a `Saved submission:` path.

- [ ] **Step 2: Distribution check**

Run with the generated path from Step 1:

```bash
python scripts/compare_candidate_distribution.py --candidate output/daily_candidates/cnn_1d_cnn_gru_YYYYMMDD_HHMMSS.csv --reference output/daily_candidates/cnn_1d_20260518_180837.csv
```

Expected: record correlation, mean_abs_diff, max_week_mean_shift, candidate week means, and `safe`.

## Task 5: Lightweight Backtest For Any Plausible Variant

**Files:**
- Generated output: `output/backtests/`

- [ ] **Step 1: Decide whether to backtest**

Only backtest a variant if its distribution has:

```text
correlation >= 0.95
mean_abs_diff <= 0.15
max_week_mean_shift <= 0.10
```

Expected: skip backtest if no variant clears these looser pre-gates.

- [ ] **Step 2: Run lightweight backtest for the best plausible variant**

Use the same hyperparameters as the selected generated candidate. For Variant C, run:

```bash
python scripts/run_temporal_backtest.py --mode cnn --model cnn_gru --recent-cutoffs 2 --max-windows-per-region 52 --epochs 3 --batch-size 128 --dropout 0.25 --weight-decay 0.002 --lr 0.0005 --scheduler
```

Expected: terminal prints `cnn overall_mae` and saves a CSV in `output/backtests/`.

## Task 6: Log Results And Decision

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: Add regularized GRU result section**

Record:

```text
variant name
command summary
generated file path
validation MAE
distribution metrics
backtest MAE if run
submit/reject decision
```

- [ ] **Step 2: Commit AGENTS.md**

Run:

```bash
git add AGENTS.md
git commit -m "docs: log regularized cnn gru variants"
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

- Spec coverage: all three regularized GRU variants, distribution-first gating, optional backtest, and logging are covered.
- Placeholder scan: timestamp examples use `YYYYMMDD_HHMMSS` because filenames are created at runtime; copy exact generated paths from terminal output.
- Type consistency: CLI flags match existing `generate_cnn_submission.py` and `run_temporal_backtest.py` flags.
