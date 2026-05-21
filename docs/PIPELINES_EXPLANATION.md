---
title: "NYCU Data Mining 2026 — Final Project Pipelines Explained"
geometry: margin=0.85in
fontsize: 10pt
---

# Goal

Predict natural disaster severity over the next five weeks.

Input per region: the last 91 daily meteorological rows.

Target: five future weekly severity scores.

Kaggle metric: mean absolute error (MAE), lower is better.

# Pipeline Map

| Pipeline | Main File | Role |
|---|---|---|
| Aggregate XGBoost | `model/train.py`, `model/utils.py` | Stable baseline |
| Temporal Tree | `scripts/generate_temporal_tree_submission.py` | Better tree model |
| Small 1D CNN | `scripts/generate_cnn_submission.py --model small` | Best current model |
| V2 CNN | `scripts/generate_cnn_submission.py --model v2` | Larger CNN, overfit before |
| Temporal Backtest | `scripts/run_temporal_backtest.py` | Main validation method |

# Data Windowing Pipeline

Raw data is daily meteorological records per region. The competition
gives 91 daily test rows for each region and asks for five weekly
predictions.

For training, we create many supervised examples from history:

```
region R1:
  days 1-91      → next 5 weekly scores
  days 8-98      → next 5 weekly scores
  days 15-105    → next 5 weekly scores
  ...
```

Every example is shaped exactly like the test task:

```
X  = 91 days × 14 meteorological features
y  = 5 future weekly severity scores
```

That is why **91 days** is used everywhere.

# Aggregate XGBoost Baseline

## How It Works

XGBoost is a tree-boosting model. Instead of training one large decision
tree, it trains many small trees sequentially. Each new tree tries to
correct the remaining errors from the previous ones.

```
pred_0  = initial guess
tree_1  = corrects pred_0 errors
tree_2  = corrects remaining errors after tree_1
tree_3  = corrects remaining errors after tree_2
...
final   = sum of all tree outputs
```

## Feature Engineering

The baseline XGBoost does **not** use the raw 91-day sequence. It
summarises each window with handcrafted statistics:

```text
mean, std, min, max, q25, q50, q75
last7_mean, last30_mean
trend, skew, kurtosis
```

Compression:

```
91 days × 14 features → 168 aggregate features
```

The XGBoost model then predicts five weekly scores from those 168
numbers. Because XGBoost predicts one target at a time, multi-output
wrapping is used so the same architecture handles all five weeks in one
training call.

## Strengths & Weaknesses

- **Fast, stable, easy to debug.**
- **Loses detailed time order inside the 91 days.**  “Rain was high
  recently” and “rain was high 80 days ago” may look similar to the
  model because only global statistics are used.

## Performance

Best XGBoost submission: `submission_xgb_v2_fixed.csv`, public MAE
0.8434.

# Temporal Tree

## How It Differs from the Baseline

Temporal Tree is still XGBoost, but the features try to preserve more
**time information** from the 91-day window.

The baseline asks:

> What are the overall statistics over 91 days?

The temporal tree also asks:

> What changed recently?  Is the last 7 days different from the full
> period?  Is the last 30 days different from earlier days?

## Feature Style

The best version is the **hybrid temporal tree**.  It combines:

- The original 168 aggregate features (same as baseline).
- Targeted temporal deltas: recent-vs-full-window differences, recent
  trend, and recent-vs-older signals.

Example features that the temporal tree can use:

```text
last7_mean  - full_mean
last30_mean - full_mean
recent trend
recent-vs-old delta
```

## Why It Can Help

A region may have normal 91-day average rainfall but very high rainfall
in the last week.  A pure aggregate model may under-react because the
global average looks normal.  The temporal tree can notice the recent
spike.

## What Did Not Work

A wide temporal block feature set (splitting the 91 days into many
blocks, creating ~938 features) was also tested.  It produced CV MAE
0.46336 and was **not submitted**.

## Performance

Best temporal tree submission: `temporal_tree_hybrid_20260518_181718.csv`,
public MAE 0.8403.  Better than the XGBoost baseline, but still not as
strong as the small CNN.

# Small 1D CNN

## Why 1D

The input is a **time sequence**, not an image:

```text
(91 days) × (14 weather features)
```

The only ordered axis is time.  Convolution filters slide along the
**day axis**:

```text
[day 1–5]   → pattern
[day 2–6]   → pattern
[day 3–7]   → pattern
...
```

A 2D CNN is used for image-like grids (height × width × channels).  Our
data is not a picture — it is one region at a time over time.  So 1D is
the correct fit.

## How It Works

The small CNN uses the raw 91-day meteorological sequence directly.
It does **not** rely on hand-engineered aggregate statistics.

The model consists of:

- 1D convolution layers that slide across time.
- Global average pooling to compress the time dimension.
- Dense layers to produce five weekly predictions.

It can learn temporal patterns automatically:

- Rainfall spike followed by humidity increase.
- Temperature trend over several days.
- Pressure-drop pattern.
- Wind/rain combinations.

## Why It Is the Best So Far

- It preserves time order.
- It learns useful temporal patterns without manual feature engineering.
- It avoids depending only on aggregates that lose time structure.

## Performance

Best current submission: `cnn_1d_20260518_180837.csv`, public MAE 0.8222
(25 epochs, small 1D CNN).

# V2 CNN (Larger Architecture)

## What Changed

V2 CNN adds:

- More convolution channels.
- Batch normalisation.
- A deeper dense head.

## The Validation Problem

Local (region-based) validation results were:

| Model | Local val MAE | Public MAE |
|---|---|---|
| Small CNN (best) | 0.3614 | **0.8222** |
| V2 CNN | 0.2417 | 0.8901 |

Local validation was **much better** for V2, but Kaggle public MAE was
**much worse**.  This means the old region-based split was not reliable
for architecture selection.  The larger model overfit that split.

## Current Policy

Do **not** trust V2 CNN unless the temporal backtest (next section)
shows it is genuinely better than the small CNN.

# Temporal Backtest — Main Validation Pipeline

## Why the Old Validation Was Misleading

Previous validation used **region-based splits**:

```text
train on some regions → validate on other regions
```

But the Kaggle task is **not** an unseen-region problem.  Kaggle gives
the **same regions** and asks to predict the **next future horizon**.

## How Temporal Backtest Works

For each region, historical terminal-like tasks are created:

```text
Historical cutoff 1:
  train on older windows
  validate on a recent future horizon

Historical cutoff 2:
  train on older windows
  validate on another recent future horizon
```

The validation question matches Kaggle more closely:

> Given the last 91 days of weather, can you predict the next five
> weekly scores?

This answers the same “future horizon” question for historical cutoffs
where labels are known.

## What the Backtest Detects

- Overfitting patterns like the V2 CNN case.
- Terminal label drift (labels become sparser or shift toward the end
  of history).
- Whether better local validation really means better future
  predictions.

## Smoke Results (May 21)

| Mode | Overall MAE | Config |
|---|---|---|
| Tree | 0.300622 | 2 recent cutoffs |
| CNN small | 0.249133 | 1 recent cutoff, 2 epochs |

The absolute MAE is **not** directly comparable to Kaggle public MAE
because the public set has additional distribution shift and unknown
labels.  The backtest scores are for **relative ranking** of candidates.

## Usage

Before spending one of the limited daily Kaggle submissions, run:

```bash
python scripts/run_temporal_backtest.py --mode both --recent-cutoffs 2
```

If backtest MAE looks suspicious (too high relative to other runs), do
not submit that candidate.

# Recommended Workflow

```
1. Choose a candidate model and parameter set.
2. Run the temporal backtest.
3. If backtest MAE is competitive, train on all available training data.
4. Generate a submission CSV.
5. Validate the CSV format and column order against the Kaggle sample.
6. Submit to Kaggle only if the evidence is strong.
7. Log the result in AGENTS.md and output/SUBMISSIONS.md.
```

# Key Takeaway

| Pipeline | Role | Approach |
|---|---|---|
| Aggregate XGBoost | Stable baseline | Handcrafted statistics, no time structure |
| Temporal Tree | Better tree | Adds targeted recent-change signals |
| Small 1D CNN | **Best model** | Learns temporal patterns from raw sequence |
| Temporal Backtest | **Decision gate** | Validates on historical future-horizon tasks |
