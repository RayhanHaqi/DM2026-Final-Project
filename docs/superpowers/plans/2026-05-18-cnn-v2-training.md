# CNN V2 Training Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare tomorrow's CNN candidates by adding a stronger CNN V2 architecture and configurable training controls.

**Architecture:** `model/cnn_candidate.py` will expose model construction for `small` and `v2`, plus training parameters for seed, patience, dropout, weight decay, and scheduler. `scripts/generate_cnn_submission.py` will pass those options through and record them in the summary CSV.

**Tech Stack:** Python, NumPy, pandas, PyTorch, scikit-learn, unittest.

---

### Task 1: CNN Model Options

**Files:**
- Modify: `tests/test_temporal_candidates.py`
- Modify: `model/cnn_candidate.py`

- [ ] Add a failing test that `build_torch_model("v2", n_features, dropout)` outputs 5 values and has more parameters than `small`.
- [ ] Implement `build_torch_model()` with `small` and `v2` architectures.
- [ ] Update `train_torch_cnn()` to use the selected model.

### Task 2: Training Controls

**Files:**
- Modify: `tests/test_scripts.py`
- Modify: `scripts/generate_cnn_submission.py`
- Modify: `model/cnn_candidate.py`

- [ ] Add script help tests for `--model`, `--seed`, `--patience`, `--dropout`, `--weight-decay`, and `--scheduler`.
- [ ] Pass those options from the script to `train_torch_cnn()`.
- [ ] Record those options in the summary CSV.

### Task 3: Documentation

**Files:**
- Modify: `AGENTS.md`

- [ ] Add next-candidate commands for CNN V2 and improved small CNN training.

### Task 4: Verification

**Files:**
- Run tests only, no heavy training.

- [ ] Run `python -m unittest tests/test_temporal_candidates.py tests/test_scripts.py -v`.
- [ ] Run `python -m compileall model scripts tests`.
- [ ] Run `python scripts/generate_cnn_submission.py --help`.
