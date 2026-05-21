# Tomorrow Todo - 2026-05-22

## Tonight Progress Review

- Removed the daily 3-shot XGBoost runner from `model/experiments.py`.
- Kept `model/experiments.py` as shared submission utilities only: clipping, blending, submission building, and submission validation.
- Removed the daily-runner tests from `tests/test_experiments.py`.
- Updated `AGENTS.md` so temporal backtest is documented as the validation workflow instead of daily 3-shot selection.
- Committed that cleanup as `167a257 feat: remove daily 3-shot runner, keep submission utilities`.
- Added temporal backtest execution into both active submission scripts:
  - `scripts/generate_cnn_submission.py` now runs CNN temporal backtest before final submission training unless `--no-backtest` is passed.
  - `scripts/generate_temporal_tree_submission.py` now runs tree temporal backtest before final submission training unless `--no-backtest` is passed.
  - Both scripts support `--backtest-cutoffs`, defaulting to `2`.

## Current Pipeline Map

| Pipeline | Entry Point | Role | Notes |
|---|---|---|---|
| Aggregate XGBoost | `model/train.py` + `model/utils.py` | Baseline model | Useful reference, not the main candidate path. |
| Temporal Tree | `scripts/generate_temporal_tree_submission.py` | Tree candidate generator | Now runs temporal backtest before final training. |
| Small 1D CNN | `scripts/generate_cnn_submission.py --model small` | Main candidate generator | Best public score so far: `0.8222`. Now runs temporal backtest before final training. |
| V2 CNN | `scripts/generate_cnn_submission.py --model v2` | Larger CNN experiment | Historically overfit; use only if temporal backtest supports it. |
| Temporal Backtest | `scripts/run_temporal_backtest.py` | Main validation pipeline | Use for relative ranking before spending Kaggle submissions. |

## Tomorrow Priority List

### 1. Commit Backtest Integration Into Submission Scripts

- Review the current diff in `scripts/generate_cnn_submission.py` and `scripts/generate_temporal_tree_submission.py`.
- Run verification:

```bash
python -m unittest tests/test_progress.py tests/test_submission.py tests/test_experiments.py tests/test_temporal_candidates.py tests/test_scripts.py tests.test_backtest -v
python -m compileall model scripts tests
```

- If verification passes, commit:

```bash
git add scripts/generate_cnn_submission.py scripts/generate_temporal_tree_submission.py docs/TOMORROW_TODO_2026-05-22.md
git commit -m "feat: run temporal backtest before submission generation"
```

### 2. Run One Low-Cost Integrated Smoke Test

- Use `--no-backtest` only if the PC is under load. Otherwise test the new default path.
- CNN low-cost smoke:

```bash
python scripts/generate_cnn_submission.py --model small --max-windows-per-region 52 --epochs 2 --batch-size 256 --backtest-cutoffs 1
```

- Tree low-cost smoke:

```bash
python scripts/generate_temporal_tree_submission.py --feature-set hybrid --max-windows-per-region 52 --n-splits 2 --backtest-cutoffs 1
```

- Confirm each script prints `backtest_mae` and writes a timestamped submission plus summary CSV.

### 3. Decide Tomorrow's Submission Candidate

- Compare integrated backtest MAE, historical public score, and prediction distribution.
- Default candidate should be conservative: small CNN only, unless the temporal tree or a blend has clearly better backtest evidence.
- Do not submit V2 CNN unless temporal backtest ranks it competitively against the small CNN.

### 4. Update Tracking Before Any Kaggle Upload

- Update `AGENTS.md` with the integrated backtest smoke results.
- Update `output/SUBMISSIONS.md` after any Kaggle upload and public score.
- Keep filenames timestamp-only; do not use `_vN` suffixes.

### 5. Optional Improvement If Time Allows

- Add a lighter `--max-regions` or `--sample-regions` option to `scripts/run_temporal_backtest.py` for faster smoke tests.
- This should be validation-only and should not affect real candidate generation.

## Guardrails

- No full all-window local training.
- Safe local default remains `--max-windows-per-region 52`.
- Use temporal backtest for ranking; do not trust old region-only validation alone.
- Use Kaggle submissions sparingly; current best is still `cnn_1d_20260518_180837.csv` with public MAE `0.8222`.
