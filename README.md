# NYCU Data Mining (Spring 2026) Final Project: Natural Disaster Severity Prediction

Kaggle competition: predict drought severity (0-5) for 5 future weeks per region from 91 days of meteorological data.

## File Structure

```text
.
├── Project_Analysis.ipynb       # Main notebook: EDA, baseline, submission
├── README.md
├── .gitignore
├── requirements.txt
├── setup.py
├── model/
│   ├── __init__.py
│   ├── utils.py                 # Data loading, feature engineering, submission
│   └── train.py                 # XGBoost multi-output regression
├── data/                        # gitignored — download manually
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
└── output/                      # gitignored
    └── submission_xgb.csv
```

## Setup

```bash
pip install -e .
```

## Data

1. Log in to Kaggle with your NYCU email: https://www.kaggle.com/t/7177902eb8b34b25a75e932d4e235b32
2. Download `train.csv`, `test.csv`, `sample_submission.csv` to `data/`

## Running

```bash
jupyter nbconvert --to notebook --execute Project_Analysis.ipynb
```
