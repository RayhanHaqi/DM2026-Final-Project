# Daily Three Experiment Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a small, safe experiment runner that trains several offline candidates, validates submissions, and produces the top three files for each Kaggle day.

**Architecture:** Add a focused `model/experiments.py` module for candidate definitions, prediction clipping/blending, schema validation, and CSV writing. Keep model training in `model/train.py` and data loading in `model/utils.py`; the notebook can call the runner after data is loaded.

**Tech Stack:** Python, NumPy, pandas, scikit-learn MAE, existing XGBoost training utilities, `unittest`.

---

### Task 1: Submission Utilities

**Files:**
- Create: `model/experiments.py`
- Create: `tests/test_experiments.py`

- [ ] **Step 1: Write failing tests for clipping, blending, and sample-order validation**

```python
def test_clip_predictions_limits_values_to_target_range(self):
    preds = np.array([[-1.0, 2.0, 6.0]])
    result = experiments.clip_predictions(preds)
    np.testing.assert_allclose(result, [[0.0, 2.0, 5.0]])

def test_blend_predictions_combines_two_arrays(self):
    current = np.array([[1.0, 3.0]])
    candidate = np.array([[3.0, 5.0]])
    result = experiments.blend_predictions(candidate, current, candidate_weight=0.75)
    np.testing.assert_allclose(result, [[2.5, 4.5]])

def test_validate_submission_requires_sample_columns_and_order(self):
    sample = pd.DataFrame({"region_id": ["R1", "R2"], "pred_week1": [0, 0]})
    sub = pd.DataFrame({"region_id": ["R2", "R1"], "pred_week1": [0.2, 0.1]})
    ok, messages = experiments.validate_submission(sub, sample)
    self.assertFalse(ok)
    self.assertIn("ID order does not match sample submission", messages)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest tests/test_experiments.py -v`
Expected: FAIL because `model.experiments` does not exist.

- [ ] **Step 3: Implement minimal utilities**

```python
def clip_predictions(preds):
    return np.clip(preds, 0.0, 5.0)

def blend_predictions(candidate, current_best, candidate_weight):
    return candidate_weight * candidate + (1.0 - candidate_weight) * current_best

def validate_submission(sub, sample):
    messages = []
    if list(sub.columns) != list(sample.columns):
        messages.append("Columns do not match sample submission")
    if sub.shape != sample.shape:
        messages.append("Shape does not match sample submission")
    if sub.iloc[:, 0].tolist() != sample.iloc[:, 0].tolist():
        messages.append("ID order does not match sample submission")
    if sub.isna().any().any():
        messages.append("Submission contains null values")
    return len(messages) == 0, messages
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest tests/test_experiments.py -v`
Expected: PASS.

### Task 2: Candidate Runner

**Files:**
- Modify: `model/experiments.py`
- Modify: `tests/test_experiments.py`

- [ ] **Step 1: Write failing tests for candidate selection**

```python
def test_select_top_candidates_sorts_by_cv_mae(self):
    results = [
        {"name": "b", "cv_mae": 0.4},
        {"name": "a", "cv_mae": 0.3},
        {"name": "c", "cv_mae": 0.5},
    ]
    selected = experiments.select_top_candidates(results, limit=2)
    self.assertEqual([r["name"] for r in selected], ["a", "b"])
```

- [ ] **Step 2: Implement candidate selection and daily params**

Define three default XGBoost candidates:
- `xgb_a_depth4_300`: 300 estimators, lr 0.04, max_depth 4
- `xgb_b_depth5_300`: 300 estimators, lr 0.03, max_depth 5
- `xgb_c_reg_depth4_200`: 200 estimators, lr 0.05, max_depth 4, stronger regularization

- [ ] **Step 3: Verify tests pass**

Run: `python -m unittest tests/test_experiments.py -v`
Expected: PASS.

### Task 3: Notebook Hook And Verification

**Files:**
- Modify: `Project_Analysis.ipynb`
- Modify: `README.md` or `AGENTS.md` only if needed for run instructions

- [ ] **Step 1: Add notebook cell showing daily runner usage**

Add a cell after the baseline submission section with commented usage for `experiments.run_daily_candidates(...)`, so users can intentionally run it without changing the default baseline path.

- [ ] **Step 2: Run full lightweight verification**

Run: `python -m unittest tests/test_progress.py tests/test_submission.py tests/test_experiments.py -v`
Expected: PASS.

Run: `python -m compileall model tests`
Expected: no compile errors.

Run: `python -m json.tool Project_Analysis.ipynb >/dev/null`
Expected: valid JSON.
