# Temporal Backtest Design - 2026-05-20

## Goal

Build a validation-only temporal backtest that better predicts Kaggle public MAE for this project than the current region-based validation setup.

## Problem

Current model selection is driven by validation schemes that do not match the Kaggle task closely enough.

- Tree models are ranked by `GroupKFold` over regions.
- CNN models are ranked by a single `GroupShuffleSplit` holdout over regions.
- Kaggle test data is not an unseen-region problem. It is a future-horizon problem on the same region universe, using the final 91 daily rows per region to predict the next 5 weekly scores.

This mismatch has already produced bad selection decisions:

- V2 CNN had much better local validation than the small CNN, but much worse public MAE.
- Later small-CNN seed and regularization variants also looked acceptable offline but degraded publicly.
- Public results suggest the validation signal is not ranking candidates in the same order as Kaggle.

## Scope

This design only covers validation infrastructure.

It will:

- build temporal backtest splits that mimic Kaggle's terminal prediction shape
- support both tree-style and CNN-style backtest evaluation
- report metrics that help rank candidates before using submission slots

It will not:

- add new model architectures
- add new feature families beyond what current pipelines already use
- generate new Kaggle submissions as part of the validation tool itself

## Design Principles

- Match the public task shape as closely as possible.
- Reuse existing feature and training code where practical.
- Keep the first version lightweight enough for local execution.
- Make CNN runs optional and parameterized because they are slower than tree backtests.
- Prefer explicit, inspectable reports over opaque ranking logic.

## Backtest Shape

Each validation example should match a Kaggle-like prediction event.

- Unit of evaluation: one prediction row per region per cutoff.
- Input window: the 91 daily meteorological rows immediately before the validation horizon.
- Target horizon: the next 5 known weekly severity scores.
- Region universe: same regions as training, because Kaggle test also uses the same region universe.

The backtest should use several recent cutoffs rather than a random or region-only split. A cutoff is a terminal-like historical point where enough prior daily rows and enough future weekly labels exist.

## Split Strategy

For each region, the historical labeled sequence should be indexed in weekly-score order.

For a chosen cutoff offset from the end:

- validation uses exactly one horizon per region
- training uses only horizons strictly earlier than that validation horizon
- no validation horizon or future target should leak into training examples for that cutoff

The first implementation should support a small number of recent cutoffs, such as the last 1, 2, 3, or 5 valid horizons per region, so the user can trade off runtime against validation stability.

## Components

### `model/backtest.py`

This module should hold the shared temporal validation infrastructure.

Responsibilities:

- build per-region cutoff definitions
- build train and validation index selections for backtest runs
- provide generic MAE reporting helpers
- provide tree and CNN evaluator entry points that use the same cutoff logic

Suggested logical units inside the module:

- cutoff builder
- shared metric/report helpers
- tree backtest evaluator
- CNN backtest evaluator

These can remain in one file initially if the implementation stays readable.

### Tree Backtest Evaluator

The tree evaluator should reuse current feature builders and tree training code as much as possible.

- aggregate-feature backtest should work with the existing `utils`-style feature representation
- the first version must support the existing aggregate baseline feature path; temporal-hybrid tree support is out of scope for this first implementation
- training and scoring should return overall MAE, per-week MAE, and per-cutoff MAE

Tree backtest is the fast baseline path and should be the easiest validation mode to run.

### CNN Backtest Evaluator

The CNN evaluator should reuse the current raw 91-day sequence formulation.

- training windows must be built with the same 91-day input and 5-week target shape used by submission generation
- normalization must be fit on the backtest training portion only for each cutoff
- validation windows must be transformed with train-only normalization statistics
- the evaluator should support safe defaults and lower-epoch runs to reduce local cost

The first version only needs to support the existing `small` and `v2` model options that already exist in the codebase.

## Reporting

Backtest output should make selection failures visible instead of hiding them in one scalar.

Required metrics:

- overall MAE across all regions, cutoffs, and weeks
- per-week MAE for weeks 1 through 5
- per-cutoff MAE
- prediction means by week

Useful optional metrics:

- target means by week
- prediction minus target bias by week
- comparison against an existing reference submission or model if predictions are available

The main consumer is the user deciding whether a candidate is worth tomorrow's submission budget.

## CLI Entry Point

Add a script such as `scripts/run_temporal_backtest.py`.

It should:

- expose validation mode selection for tree and CNN paths
- expose number of recent cutoffs to evaluate
- expose safe runtime controls such as epochs, batch size, and `max_windows_per_region`
- write a concise summary artifact under `output/backtests/`
- print a compact text summary suitable for terminal use

The default configuration should be safe for local use and should not require all-window retraining.

## Data Handling Rules

- Respect existing submission-order assumptions, but the backtest itself should score against historical labels rather than submission CSV shape.
- Do not rely on random region holdouts as the primary ranking metric.
- Do not use validation data to fit normalization statistics for CNN evaluation.
- Do not use cached artifacts that silently ignore `max_windows_per_region` if that would invalidate backtest comparisons.

## Error Handling

The first version only needs lightweight, explicit guards.

- fail loudly if there are not enough labeled horizons for the requested cutoff count
- fail loudly if a region cannot provide the required 91-day window before a validation horizon
- fail loudly if requested CNN backtest settings require unavailable dependencies
- fail loudly if cached data shape does not match the requested validation mode

## Testing Strategy

Tests should prove the validation logic, not just that functions run.

Required coverage:

- cutoff builder produces terminal-style validation rows
- training horizons for a cutoff are strictly earlier than validation horizons
- tree evaluator returns correctly shaped summaries
- CNN evaluator fits normalization on train-only data for each split
- recent-cutoff selection uses the intended end-of-history horizons

The tests should be small synthetic fixtures, not large dataset runs.

## Success Criteria

The new backtest is useful if it does all of the following:

- explains why V2 CNN was falsely attractive under the old validation scheme
- does not rank the failed May 20 CNN variants as clearly superior to the May 18 best candidate
- surfaces the heavy terminal label drift already observed in recent historical horizons
- gives the user a more trustworthy offline ranking before spending Kaggle slots

## Rollout Order

1. Implement shared cutoff and metric infrastructure.
2. Implement tree backtest path and verify reports on small synthetic fixtures.
3. Implement CNN backtest path with train-only normalization per cutoff.
4. Add CLI entry point and output summaries.
5. Use the tool to re-evaluate known historical candidates before trying new submissions.

## Risks And Tradeoffs

- CNN backtest will cost more runtime than tree backtest, especially with multiple cutoffs.
- A better offline metric may still be noisy, but it should be less misleading than the current setup.
- Reusing current feature/training code reduces implementation scope, but it also carries forward current modeling assumptions.
- The first version should optimize for trustworthiness of ranking, not absolute speed.
