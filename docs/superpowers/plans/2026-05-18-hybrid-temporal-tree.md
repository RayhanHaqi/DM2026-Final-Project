# Hybrid Temporal Tree Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the weak temporal-block tree candidate with a safer hybrid feature mode that keeps the proven 168 baseline features and adds only targeted temporal deltas.

**Architecture:** `model/temporal_features.py` will expose a `feature_set` option: `blocks` for the old wide feature set and `hybrid` for the new baseline-plus-targeted set. `scripts/generate_temporal_tree_submission.py` will default to `hybrid`, support `--gpu`, and write the selected feature mode to the summary CSV.

**Tech Stack:** Python, NumPy, pandas, XGBoost, unittest.

---

### Task 1: Hybrid Features

**Files:**
- Modify: `tests/test_temporal_candidates.py`
- Modify: `model/temporal_features.py`

- [ ] Add tests proving hybrid features include baseline `trend/skew/kurt` fields, targeted temporal delta fields, and fewer columns than block mode.
- [ ] Implement `build_hybrid_temporal_features_from_window()` using `utils._aggregate_array()` plus targeted temporal means/deltas.
- [ ] Add `feature_set` parameters to temporal train/test builders and cache names.

### Task 2: Script Flags

**Files:**
- Modify: `tests/test_scripts.py`
- Modify: `scripts/generate_temporal_tree_submission.py`

- [ ] Add tests for `--feature-set {hybrid,blocks}` and `--gpu` help output.
- [ ] Default temporal script to `--feature-set hybrid`.
- [ ] Add `--gpu` to set XGBoost `tree_method=hist` and `device=cuda`.
- [ ] Use stronger baseline-like params for hybrid mode.

### Task 3: Logs

**Files:**
- Modify: `AGENTS.md`
- Modify: `output/SUBMISSIONS.md`

- [ ] Record CNN submission `cnn_1d_20260518_180837.csv` with public MAE `0.8222` as current best.
- [ ] Record temporal-block tree CV `0.46336` as rejected/not submitted.

### Task 4: Verification

**Files:**
- Run: `tests/test_temporal_candidates.py`
- Run: `tests/test_scripts.py`

- [ ] Run `python -m unittest tests/test_temporal_candidates.py tests/test_scripts.py -v`.
- [ ] Run `python -m compileall model scripts tests`.
- [ ] Do not run heavy training during implementation verification.
