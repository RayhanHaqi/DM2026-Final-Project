# Tomorrow Notes - 2026-05-19

## Current State
- Working directory: `/home/tilakoid/datamining/DM2026-Final-Project`
- Current best accepted Kaggle submission: `output/submission_xgb_v2_fixed.csv`
- Current best public MAE: `0.8434`
- Safe local feature setup uses `max_windows_per_region=52` and `n_jobs=2`.
- Do not run full all-window feature generation locally yet; it froze the PC.
- All generated submission CSVs now use timestamp-only names: `name_YYYYMMDD_HHMMSS.csv`.
- Do not use `_v1`, `_v2`, etc. for new generated CSVs.

## What Was Done Today
- Added progress bars for train feature generation and cross-validation.
- Added safer XGBoost defaults: `n_estimators=200`, `n_jobs=2`.
- Fixed submission generation to match `data/sample_submission.csv` row order and columns.
- Added daily experiment runner in `model/experiments.py`.
- Added tests for progress bars, submission formatting, and daily experiment output.
- Added timestamped CSV output for `utils.generate_submission()` and `experiments.run_daily_candidates()`.
- Updated `AGENTS.md` with the daily 3-shot workflow and timestamp naming rule.

## Verified Before Stopping
Run from `/home/tilakoid/datamining/DM2026-Final-Project`:

```bash
python -m unittest tests/test_progress.py tests/test_submission.py tests/test_experiments.py -v
python -W error::FutureWarning -m unittest tests/test_experiments.py -v
python -m compileall model tests
python -m json.tool Project_Analysis.ipynb >/dev/null
```

Status today: all passed.

## Tomorrow Plan
1. Open `Project_Analysis.ipynb` and run the safe cached/52-window workflow first.
2. Run the daily experiment section using `run_daily_candidates(...)`.
3. Inspect generated files in `output/daily_candidates/`.
4. Submit at most 3 files to Kaggle:
   - Best local CV candidate.
   - Second-best local CV candidate.
   - 70/30 blend against current best `submission_xgb_v2_fixed.csv`.
5. Record every submitted file and Kaggle public MAE in `AGENTS.md`.
6. If any file is rejected, record the rejection reason in `AGENTS.md` before trying another file.

## Commands To Re-run Before Submitting
```bash
python -m unittest tests/test_progress.py tests/test_submission.py tests/test_experiments.py -v
python -m json.tool Project_Analysis.ipynb >/dev/null
```

## Important Reminders
- `sample_submission.csv` order is not the same as raw test group order. Always use the submission utilities.
- Accepted best file was `submission_xgb_v2_fixed.csv`, not the earlier rejected files.
- Keep every experiment/result tracked in `AGENTS.md`.
- Use only 3 Kaggle submissions per day.
