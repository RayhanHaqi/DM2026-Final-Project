# Calendar-Small-CNN Scheduler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate one scheduler-enabled Calendar-Small-CNN candidate under memory-safe execution settings.

**Architecture:** Reuse the existing small CNN, calendar feature flag, scheduler flag, temporal backtest runner, and distribution comparison script. This is an experiment-only plan with no source-code changes unless a run exposes a bug.

**Tech Stack:** Python, PyTorch, existing `scripts/run_temporal_backtest.py`, existing `scripts/generate_cnn_submission.py`, existing `scripts/compare_candidate_distribution.py`.

---

## File Structure

- Modify `AGENTS.md`: record backtest, generated output path, validation MAE, distribution metrics, and decision.
- No source files should be modified for this plan.

## Task 1: Lightweight Calendar Scheduler Backtest

**Files:**
- Generated output: `output/backtests/`

- [ ] **Step 1: Run standalone lightweight backtest**

Run:

```bash
python scripts/run_temporal_backtest.py --mode cnn --model small --calendar --recent-cutoffs 2 --max-windows-per-region 52 --epochs 3 --batch-size 128 --scheduler
```

Expected: terminal prints `cnn overall_mae` and saves `output/backtests/temporal_backtest_YYYYMMDD_HHMMSS.csv`.

- [ ] **Step 2: Decide whether to generate**

Generate only if the command completes and the backtest is not clearly worse than the anchor `0.189607`. If it is killed or much worse, stop this plan and log rejection.

## Task 2: Generate Scheduler Calendar Candidate

**Files:**
- Generated output: `output/daily_candidates/`

- [ ] **Step 1: Run full generation without integrated backtest**

Run:

```bash
python scripts/generate_cnn_submission.py --model small --calendar --scheduler --no-backtest --max-windows-per-region 52 --epochs 25 --seed 42 --dropout 0.15 --weight-decay 0.001 --batch-size 128
```

Expected: terminal prints epoch validation MAE and saves `output/daily_candidates/cnn_1d_small_YYYYMMDD_HHMMSS.csv`.

## Task 3: Distribution Gate

**Files:**
- No source changes expected.

- [ ] **Step 1: Compare against current best**

Run with the generated path from Task 2:

```bash
python scripts/compare_candidate_distribution.py --candidate output/daily_candidates/cnn_1d_small_YYYYMMDD_HHMMSS.csv --reference output/daily_candidates/cnn_1d_20260518_180837.csv
```

Expected: submit only if `safe True`.

- [ ] **Step 2: Reject if distribution remains unsafe**

Reject immediately if metrics resemble the prior unsafe calendar run:

```text
correlation around 0.879
mean_abs_diff around 0.239
max_week_mean_shift around 0.114
```

## Task 4: Log Results And Decision

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: Add calendar scheduler result section**

Record:

```text
backtest command and MAE
generated file path
validation MAE
distribution metrics
submit/reject decision
```

- [ ] **Step 2: Commit AGENTS.md**

Run:

```bash
git add AGENTS.md
git commit -m "docs: log calendar scheduler candidate"
```

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

- Spec coverage: lightweight backtest, generation, distribution gate, and logging are covered.
- Placeholder scan: timestamp examples use `YYYYMMDD_HHMMSS` because filenames are created at runtime; copy exact generated paths from terminal output.
- Type consistency: all flags match existing CNN runner and temporal backtest runner.
