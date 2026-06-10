# NYCU Data Mining (Spring 2026) — Final Project

**Course:** 535703 資料探勘 Data Mining  
**Student:** Muhammad Rayhan Athaillah (313540001)  
**Competition:** Natural disaster severity prediction (0–5) for 2,248 regions, 5-week horizon from 91 days of weather  
**Metric:** Mean Absolute Error (MAE) on Kaggle — lower is better

## Assignment deliverables (Moodle + Kaggle)

Both **Assignment 3** and **Final Project** use the same Kaggle competition. Check [E3 Moodle](https://e3p.nycu.edu.tw/course/view.php?id=23520) for the latest notices.

| Deliverable | Where | Deadline (Spring 2026) |
|---|---|---|
| Kaggle CSV submission | [Kaggle competition](https://www.kaggle.com/t/7177902eb8b34b25a75e932d4e235b32) (NYCU email required) | **Jun 10, 11:55 PM** |
| Assignment 3 upload | Moodle → Assignment 3 | **Jun 10, 11:59 PM** |
| Final Project upload | Moodle → Final Project | **Jun 10, 11:59 PM** |
| Source code | This GitHub repo | With report / as required by slides |
| Written report (PDF) | `report/report.pdf` (build from `report/report.tex`) | Upload to Moodle Final Project |

**Official rules and format:** [Course slides (Google Drive)](https://docs.google.com/presentation/d/1Mxmsw4LFCm1yV7WJn1QinvFUQCZ2M6ja/edit)  
**Intro video:** [Google Drive](https://drive.google.com/file/d/1mPXbpRrI1EacI3SxEMRHSBt1R9mjYDj1/view)

### Submission checklist

1. **Kaggle** — submit a CSV that matches `data/sample_submission.csv` (row order and columns).
2. **Moodle Assignment 3** — complete the assignment slot (competition submission per course instructions).
3. **Moodle Final Project** — upload the report PDF and any other files required by the slides.
4. **GitHub** — ensure this repo is accessible and matches what you describe in the report.

**Current best candidate:** `output/daily_candidates/prob_blend_recycle8089_ord08.csv` (public MAE **0.8088**).  
Reproduce from caches (see [Reproduce best submission](#reproduce-best-submission)).

## Problem summary

- **Train:** 12.3M rows, 5,480 days × 2,248 regions, 14 meteorological features, weekly severity labels 0–5.
- **Test:** Last 91 days of weather per region; predict the next 5 weekly scores.
- **Output:** One CSV row per region, columns `region_id`, `week_1` … `week_5`.

## Repository layout

```text
.
├── Project_Analysis.ipynb    # Main notebook: EDA, baseline, candidate generation
├── README.md
├── requirements.txt
├── setup.py                  # pip install + optional Kaggle data download
├── model/                    # Training, features, blending, validation helpers
├── scripts/                  # Submission generators, caches, gates, backtests
├── tests/                    # Unit tests (run with PYTHONPATH=.)
├── report/
│   ├── report.tex            # Final project report (LaTeX source)
│   └── report.pdf            # Built PDF for Moodle upload
├── data/                     # gitignored — train.csv, test.csv, sample_submission.csv
└── output/                   # gitignored — submissions, prob caches, backtests
```

Key `model/` modules: `utils.py`, `train.py`, `temporal_features.py`, `ordinal_tree.py`, `probability_blend.py`, `severity_history.py`, `experiments.py`.

Key `scripts/`: `gate_and_validate_submissions.py`, `blend_prob_submissions.py`, `cache_ordinal_probabilities.py`, `cache_submission_soft_probs.py`, and family-specific `generate_*_submission.py` runners.

## Setup

```bash
pip install -e .
```

Kaggle auth: `~/.kaggle/kaggle.json` or `~/.kaggle/access_token` (see `setup.py`).

### Data

1. Join the competition with your **NYCU email**: https://www.kaggle.com/t/7177902eb8b34b25a75e932d4e235b32  
2. Place `train.csv`, `test.csv`, and `sample_submission.csv` in `data/`, or run `pip install -e .` to trigger the bundled download helper.

## Running

Execute the main notebook (first run builds window feature caches; can take a while):

```bash
jupyter nbconvert --to notebook --execute --inplace Project_Analysis.ipynb
```

Or open `Project_Analysis.ipynb` in Jupyter and run all cells.

## Validate a submission CSV

Format check and distribution gate vs current best:

```bash
PYTHONPATH=. python scripts/gate_and_validate_submissions.py --reference output/daily_candidates/prob_blend_recycle8089_ord08.csv --candidate output/daily_candidates/prob_blend_recycle8089_ord08.csv
```

## Reproduce best submission

Rebuild the 92% soft recycled anchor + 8% hybrid ordinal blend:

```bash
PYTHONPATH=. python scripts/cache_submission_soft_probs.py --submission output/daily_candidates/prob_blend_recycle8092_ord10.csv --output-path output/prob_cache/soft_prob_best8089.npz
```

```bash
PYTHONPATH=. python scripts/blend_prob_submissions.py --cache output/prob_cache/soft_prob_best8089.npz:0.92 --cache output/prob_cache/ordinal_hybrid_best.npz:0.08 --output-path output/daily_candidates/prob_blend_recycle8089_ord08.csv
```

Ordinal cache (slow — trains classifiers): `PYTHONPATH=. python scripts/cache_ordinal_probabilities.py`

## Submit to Kaggle

```bash
kaggle competitions submit -c data-mining-2026-final-project -f output/daily_candidates/prob_blend_recycle8089_ord08.csv -m "prob blend recycle8089 ord08"
```

## Build report PDF

```bash
cd report && pdflatex report.tex && pdflatex report.tex
```

Upload `report/report.pdf` to Moodle **Final Project**.

## Tests

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
```

## Results (public leaderboard)

| Stage | Representative method | Public MAE |
|---|---|---:|
| XGBoost baseline | 168 aggregate features | 0.8434 |
| Small 1D CNN | Raw 91-day sequence | 0.8222 |
| CNN-GRU + tree + ordinal blends | Low-weight corrections | 0.8144 |
| LightGBM + season-tree | Grid search blends | 0.8112 |
| **Current best** | 92% prob anchor + 8% hybrid ordinal | **0.8088** |

Full experiment log: `output/SUBMISSIONS.md` and `report/report.pdf`.

## Links

- **GitHub:** https://github.com/RayhanHaqi/DM2026-Final-Project  
- **Kaggle:** https://www.kaggle.com/competitions/data-mining-2026-final-project  
- **Moodle:** https://e3p.nycu.edu.tw/course/view.php?id=23520  
- **Instructor:** Jyun-Yu Jiang — wu80623@gmail.com (subject: `DM final`)
