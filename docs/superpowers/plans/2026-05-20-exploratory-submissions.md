# Exploratory Submissions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate, validate, and select up to three exploratory Kaggle submissions focused on small CNN variants and a safe blend against the current best CNN.

**Architecture:** Use the existing `scripts/generate_cnn_submission.py` runner for raw small-CNN candidates, then use one-off Python validation commands to compare generated CSVs against `data/sample_submission.csv` and the current best submission. No new model architecture is added; only generated output files and experiment logs should change during execution.

**Tech Stack:** Python, NumPy, pandas, PyTorch through the existing CNN runner, git, Kaggle web UI or user-side upload.

---

## File Structure

- Read only: `data/sample_submission.csv` for required columns and row order.
- Read only: `output/daily_candidates/cnn_1d_20260518_180837.csv` as the current best baseline.
- Execute: `scripts/generate_cnn_submission.py` for both new raw CNN candidates.
- Modify after Kaggle results: `AGENTS.md` with generated candidates and public MAE results.
- Modify after Kaggle results: `output/SUBMISSIONS.md` with submitted file names, MD5 prefixes, public scores, and notes.
- Generated but not committed: `output/daily_candidates/*.csv` and `output/daily_candidates/*_summary.csv`.

### Task 1: Preflight Verification

**Files:**
- Read: `data/sample_submission.csv`
- Read: `output/daily_candidates/cnn_1d_20260518_180837.csv`
- Execute: `tests/test_progress.py`, `tests/test_submission.py`, `tests/test_experiments.py`, `tests/test_temporal_candidates.py`, `tests/test_scripts.py`

- [ ] **Step 1: Confirm clean source state**

Run: `git status --short`

Expected: no tracked source changes before candidate generation. Ignored/generated CSV files may exist and should not be staged.

- [ ] **Step 2: Run lightweight test suite**

Run: `python -m unittest tests/test_progress.py tests/test_submission.py tests/test_experiments.py tests/test_temporal_candidates.py tests/test_scripts.py -v`

Expected: `Ran 21 tests` and `OK`.

- [ ] **Step 3: Run script help checks**

Run: `python scripts/generate_cnn_submission.py --help`

Expected: exits successfully and includes `--model {small,v2}`.

Run: `python scripts/generate_temporal_tree_submission.py --help`

Expected: exits successfully and includes `--feature-set {hybrid,blocks}`.

- [ ] **Step 4: Validate current best baseline file exists and matches sample schema**

Run:

```bash
python - <<'PY'
import pandas as pd

sample = pd.read_csv('data/sample_submission.csv')
best = pd.read_csv('output/daily_candidates/cnn_1d_20260518_180837.csv')

assert list(best.columns) == list(sample.columns), 'columns differ from sample_submission.csv'
id_col = sample.columns[0]
assert best[id_col].tolist() == sample[id_col].tolist(), 'ID order differs from sample_submission.csv'
assert best.iloc[:, 1:].notna().all().all(), 'current best contains NaN predictions'
assert ((best.iloc[:, 1:] >= 0) & (best.iloc[:, 1:] <= 5)).all().all(), 'current best predictions outside [0, 5]'

print('baseline rows:', len(best))
print('baseline week means:', best.iloc[:, 1:].mean().round(4).to_dict())
PY
```

Expected: prints `baseline rows: 2248` and week means without assertion errors.

### Task 2: Generate Candidate A, Small CNN Seed Variant

**Files:**
- Execute: `scripts/generate_cnn_submission.py`
- Generated: `output/daily_candidates/cnn_1d_small_<timestamp>.csv`
- Generated: `output/daily_candidates/cnn_1d_small_<timestamp>_summary.csv`

- [ ] **Step 1: Generate seed-7 small CNN candidate**

Run:

```bash
python scripts/generate_cnn_submission.py --model small --max-windows-per-region 52 --epochs 25 --seed 7 --dropout 0.15 --weight-decay 0.001 --scheduler
```

Expected: prints epoch validation MAE, then `Saved submission: output/daily_candidates/cnn_1d_small_<timestamp>.csv` and `Saved summary: output/daily_candidates/cnn_1d_small_<timestamp>_summary.csv`.

- [ ] **Step 2: Record generated paths**

Run:

```bash
python - <<'PY'
from pathlib import Path

for path in sorted(Path('output/daily_candidates').glob('cnn_1d_small_20260520_*.csv')):
    if not path.name.endswith('_summary.csv'):
        print(path)
PY
```

Expected: the newest printed non-summary CSV is Candidate A. Do not use a `_summary.csv` path as a submission.

### Task 3: Generate Candidate B, Small CNN Training Variant

**Files:**
- Execute: `scripts/generate_cnn_submission.py`
- Generated: `output/daily_candidates/cnn_1d_small_<timestamp>.csv`
- Generated: `output/daily_candidates/cnn_1d_small_<timestamp>_summary.csv`

- [ ] **Step 1: Generate seed-123 regularization variant**

Run:

```bash
python scripts/generate_cnn_submission.py --model small --max-windows-per-region 52 --epochs 30 --seed 123 --dropout 0.20 --weight-decay 0.0005 --scheduler
```

Expected: prints epoch validation MAE, then `Saved submission: output/daily_candidates/cnn_1d_small_<timestamp>.csv` and `Saved summary: output/daily_candidates/cnn_1d_small_<timestamp>_summary.csv`.

- [ ] **Step 2: Record generated paths**

Run:

```bash
python - <<'PY'
from pathlib import Path

for path in sorted(Path('output/daily_candidates').glob('cnn_1d_small_20260520_*.csv')):
    if not path.name.endswith('_summary.csv'):
        print(path)
PY
```

Expected: the two generated non-summary CSVs for May 20 are printed. The later timestamp is Candidate B, and the earlier May 20 timestamp is Candidate A.

### Task 4: Validate And Compare Raw Candidates

**Files:**
- Read: `data/sample_submission.csv`
- Read: `output/daily_candidates/cnn_1d_20260518_180837.csv`
- Read: May 20 Candidate A CSV from Task 2
- Read: May 20 Candidate B CSV from Task 3

- [ ] **Step 1: Run schema and distribution comparison**

Run:

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
import pandas as pd

sample_path = Path('data/sample_submission.csv')
best_path = Path('output/daily_candidates/cnn_1d_20260518_180837.csv')
summary_paths = sorted(Path('output/daily_candidates').glob('cnn_1d_small_20260520_*_summary.csv'))

sample = pd.read_csv(sample_path)
best = pd.read_csv(best_path)
best_values = best.iloc[:, 1:].to_numpy(float)
candidate_paths = []

for summary_path in summary_paths:
    summary = pd.read_csv(summary_path)
    row = summary.iloc[0]
    if row['model'] == 'small' and row['seed'] in (7, 123):
        candidate_paths.append(Path(row['path']))

assert len(candidate_paths) == 2, f'expected two May 20 small CNN candidates, found {candidate_paths}'

for path in candidate_paths:
    df = pd.read_csv(path)
    assert list(df.columns) == list(sample.columns), f'{path}: columns differ from sample'
    id_col = sample.columns[0]
    assert df[id_col].tolist() == sample[id_col].tolist(), f'{path}: ID order differs from sample'
    assert df.iloc[:, 1:].notna().all().all(), f'{path}: contains NaN predictions'
    assert ((df.iloc[:, 1:] >= 0) & (df.iloc[:, 1:] <= 5)).all().all(), f'{path}: predictions outside [0, 5]'

    values = df.iloc[:, 1:].to_numpy(float)
    diff = np.abs(values - best_values).mean()
    corr = np.corrcoef(values.ravel(), best_values.ravel())[0, 1]
    week_means = df.iloc[:, 1:].mean().round(4).to_dict()
    print(path)
    print('  mean_abs_diff_vs_best:', round(float(diff), 6))
    print('  corr_vs_best:', round(float(corr), 6))
    print('  week_means:', week_means)
PY
```

Expected: both candidates pass assertions. Favor candidates with high correlation to current best, moderate mean absolute difference, and no strong week-by-week downward drift.

- [ ] **Step 2: Read candidate summaries**

Run:

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path

for path in sorted(Path('output/daily_candidates').glob('cnn_1d_small_20260520_*_summary.csv')):
    print(path)
    print(pd.read_csv(path).to_string(index=False))
PY
```

Expected: summaries show Candidate A as `seed=7`, `epochs=25`, `dropout=0.15`, `weight_decay=0.001`, `scheduler=True`; Candidate B as `seed=123`, `epochs=30`, `dropout=0.20`, `weight_decay=0.0005`, `scheduler=True`.

### Task 5: Build Blend Candidate

**Files:**
- Read: `data/sample_submission.csv`
- Read: `output/daily_candidates/cnn_1d_20260518_180837.csv`
- Read: safest May 20 raw candidate CSV from Task 4
- Generated: `output/daily_candidates/cnn_1d_small_blend_<timestamp>.csv`

- [ ] **Step 1: Create a conservative 70/30 blend from the safest raw candidate**

Run:

```bash
python - <<'PY'
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

sample = pd.read_csv('data/sample_submission.csv')
best = pd.read_csv('output/daily_candidates/cnn_1d_20260518_180837.csv')
best_values = best.iloc[:, 1:].to_numpy(float)

summary_paths = sorted(Path('output/daily_candidates').glob('cnn_1d_small_20260520_*_summary.csv'))
candidates = []
for summary_path in summary_paths:
    summary = pd.read_csv(summary_path)
    row = summary.iloc[0]
    if row['model'] == 'small' and row['seed'] in (7, 123):
        path = Path(row['path'])
        df = pd.read_csv(path)
        values = df.iloc[:, 1:].to_numpy(float)
        corr = np.corrcoef(values.ravel(), best_values.ravel())[0, 1]
        diff = np.abs(values - best_values).mean()
        candidates.append((corr, -diff, path, df))

assert len(candidates) == 2, f'expected two May 20 raw candidates, found {len(candidates)}'
candidates.sort(reverse=True, key=lambda item: (item[0], item[1]))
corr, neg_diff, candidate_path, candidate = candidates[0]
diff = -neg_diff
assert corr >= 0.90, f'safest candidate correlation too low for blending: {corr:.6f}'
assert diff <= 0.25, f'safest candidate mean difference too high for blending: {diff:.6f}'

assert list(candidate.columns) == list(sample.columns), 'candidate columns differ from sample'
id_col = sample.columns[0]
assert candidate[id_col].tolist() == sample[id_col].tolist(), 'candidate ID order differs from sample'

blend = sample.copy()
blend.iloc[:, 1:] = (0.70 * best.iloc[:, 1:].to_numpy(float)) + (0.30 * candidate.iloc[:, 1:].to_numpy(float))
blend.iloc[:, 1:] = blend.iloc[:, 1:].clip(0.0, 5.0)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
out_path = Path('output/daily_candidates') / f'cnn_1d_small_blend_{timestamp}.csv'
blend.to_csv(out_path, index=False)
print(out_path)
print('source_candidate:', candidate_path)
print('source_corr_vs_best:', round(float(corr), 6))
print('source_mean_abs_diff_vs_best:', round(float(diff), 6))
print('week_means:', blend.iloc[:, 1:].mean().round(4).to_dict())
PY
```

Expected: prints a new `output/daily_candidates/cnn_1d_small_blend_<timestamp>.csv` path, the source candidate, and week means. If the assertion fails, skip the blend and do not spend the third submission on it.

- [ ] **Step 2: Validate blend against current best**

Run:

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
import pandas as pd

sample = pd.read_csv('data/sample_submission.csv')
best = pd.read_csv('output/daily_candidates/cnn_1d_20260518_180837.csv')
blend_paths = sorted(Path('output/daily_candidates').glob('cnn_1d_small_blend_20260520_*.csv'))
assert blend_paths, 'no May 20 blend files found'
blend_path = blend_paths[-1]
blend = pd.read_csv(blend_path)

assert list(blend.columns) == list(sample.columns), 'blend columns differ from sample'
id_col = sample.columns[0]
assert blend[id_col].tolist() == sample[id_col].tolist(), 'blend ID order differs from sample'
assert blend.iloc[:, 1:].notna().all().all(), 'blend contains NaN predictions'
assert ((blend.iloc[:, 1:] >= 0) & (blend.iloc[:, 1:] <= 5)).all().all(), 'blend predictions outside [0, 5]'

blend_values = blend.iloc[:, 1:].to_numpy(float)
best_values = best.iloc[:, 1:].to_numpy(float)
print('blend_path:', blend_path)
print('mean_abs_diff_vs_best:', round(float(np.abs(blend_values - best_values).mean()), 6))
print('corr_vs_best:', round(float(np.corrcoef(blend_values.ravel(), best_values.ravel())[0, 1]), 6))
print('week_means:', blend.iloc[:, 1:].mean().round(4).to_dict())
PY
```

Expected: assertions pass, correlation is very high, and mean absolute difference is smaller than the selected raw candidate's difference.

### Task 6: Decide Submission Order

**Files:**
- Read: Candidate A CSV and summary
- Read: Candidate B CSV and summary
- Read: blend CSV if created

- [ ] **Step 1: Rank candidates before upload**

Run:

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
import pandas as pd

sample = pd.read_csv('data/sample_submission.csv')
best = pd.read_csv('output/daily_candidates/cnn_1d_20260518_180837.csv')
best_values = best.iloc[:, 1:].to_numpy(float)
candidate_paths = []

for summary_path in sorted(Path('output/daily_candidates').glob('cnn_1d_small_20260520_*_summary.csv')):
    row = pd.read_csv(summary_path).iloc[0]
    if row['model'] == 'small' and row['seed'] in (7, 123):
        candidate_paths.append(Path(row['path']))

candidate_paths.extend(sorted(Path('output/daily_candidates').glob('cnn_1d_small_blend_20260520_*.csv'))[-1:])

ranked = []
for path in candidate_paths:
    df = pd.read_csv(path)
    assert list(df.columns) == list(sample.columns), f'{path}: columns differ from sample'
    id_col = sample.columns[0]
    assert df[id_col].tolist() == sample[id_col].tolist(), f'{path}: ID order differs from sample'
    values = df.iloc[:, 1:].to_numpy(float)
    corr = float(np.corrcoef(values.ravel(), best_values.ravel())[0, 1])
    diff = float(np.abs(values - best_values).mean())
    week_means = df.iloc[:, 1:].mean().to_numpy(float)
    downward_drift = week_means[0] - week_means[-1]
    uploadable = corr >= 0.90 and diff <= 0.25 and downward_drift <= 0.12
    ranked.append((uploadable, corr, -diff, -abs(downward_drift), path, week_means))

ranked.sort(reverse=True, key=lambda item: (item[0], item[1], item[2], item[3]))
for uploadable, corr, neg_diff, neg_drift, path, week_means in ranked:
    print(path)
    print('  uploadable:', uploadable)
    print('  corr_vs_best:', round(corr, 6))
    print('  mean_abs_diff_vs_best:', round(-neg_diff, 6))
    print('  week_means:', dict(zip(sample.columns[1:], np.round(week_means, 4))))
PY
```

Rank by this order of evidence:

1. Passes schema and `[0, 5]` range checks.
2. No obvious V2-style week drift.
3. Best balance of local validation MAE and similarity to current best.
4. Blend safety if raw candidates are more divergent than expected.

Expected: a short ordered list of 1-3 uploadable CSV paths.

- [ ] **Step 2: Stop if no candidate passes the gate**

If all raw candidates are divergent and the blend does not look safe, do not submit. Record the generated candidates and rejection reason in `AGENTS.md` without using Kaggle submissions.

### Task 7: Submit And Log Results

**Files:**
- Modify: `AGENTS.md`
- Modify: `output/SUBMISSIONS.md`
- Generated, not committed: submitted CSVs under `output/daily_candidates/`

- [ ] **Step 1: Submit the selected CSVs manually**

Use the Kaggle web UI or the user's available Kaggle submission method. The local environment previously had no working `kaggle` CLI, so do not assume CLI upload works.

Expected: Kaggle returns a public MAE score or a rejection reason for each uploaded CSV.

- [ ] **Step 2: Compute MD5 prefixes for submitted files**

Run:

```bash
md5sum output/daily_candidates/cnn_1d_small_20260520_*.csv output/daily_candidates/cnn_1d_small_blend_20260520_*.csv
```

Expected: one MD5 hash per generated May 20 candidate file. Use the first 8 characters for files that were submitted or intentionally skipped in `output/SUBMISSIONS.md`.

- [ ] **Step 3: Update `output/SUBMISSIONS.md`**

Append one row per submitted or intentionally not-submitted May 20 candidate. Each row must include the actual CSV basename, `May 20`, the first 8 MD5 characters, the Kaggle public score or skipped/rejected status, model type (`Small 1D CNN` or `Small 1D CNN blend`), `Raw 91-day sequences`, and the exact training settings plus validation notes from Tasks 4-6.

Expected: every upload or skipped generated candidate has a clear status and note.

- [ ] **Step 4: Update `AGENTS.md`**

Add the May 20 results to the submission history and current ranking. If a new score beats `0.8222`, update the current best section. If not, keep `cnn_1d_20260518_180837.csv` as current best.

Expected: `AGENTS.md` clearly states which May 20 candidates were submitted, their public MAE, and the next recommended direction.

### Task 8: Final Verification And Commit Logs

**Files:**
- Modify: `AGENTS.md`
- Modify: `output/SUBMISSIONS.md`

- [ ] **Step 1: Run docs/source verification**

Run: `python -m json.tool Project_Analysis.ipynb >/dev/null`

Expected: exits successfully.

Run: `python -m unittest tests/test_progress.py tests/test_submission.py tests/test_experiments.py tests/test_temporal_candidates.py tests/test_scripts.py -v`

Expected: `Ran 21 tests` and `OK`.

- [ ] **Step 2: Stage only tracking docs**

Run: `git add AGENTS.md output/SUBMISSIONS.md docs/superpowers/plans/2026-05-20-exploratory-submissions.md`

Expected: generated submission CSVs remain unstaged.

- [ ] **Step 3: Inspect staged files**

Run: `git diff --cached --stat`

Expected: only `AGENTS.md`, `output/SUBMISSIONS.md`, and this plan file appear unless implementation changed code unexpectedly.

- [ ] **Step 4: Commit tracking updates**

Run: `git commit -m "docs: log exploratory submission results"`

Expected: commit succeeds.

- [ ] **Step 5: Verify final status**

Run: `git status --short`

Expected: no tracked source changes remain. Ignored generated CSVs may remain in `output/daily_candidates/`.
