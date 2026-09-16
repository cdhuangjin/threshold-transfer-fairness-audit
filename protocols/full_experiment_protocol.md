# BAF Threshold-Transfer Full Experiment Protocol v0.1

Date frozen: 2026-09-15
Precondition: threshold-transfer Gate `GO` via the pre-registered ranking-turnover path.
Data: existing BAF v2 files under `D:\data数据集\BAF`; no new download.

## Main question

Does the ranking instability observed in the low-cost Gate persist across seeds, BAF's six controlled variants, operating-point targets, and controlled feature corruption?

## Main experiment

- Six variants: Base, Variant I, Variant II, Variant III, Variant IV, Variant V.
- Three fixed LightGBM capacities: `small`, `reference`, `large`; no hyperparameter tuning.
- Five seeds: `42, 314, 2718, 1618, 8675309`.
- Fit period: months 0–4; historical calibration: month 5; evaluation: months 6–7.
- Total model fits: `6 × 3 × 5 = 90`.
- Policies: `test_oracle`, `fixed_m5`, `lag1`, using the same definitions as the Gate.
- Operating-point targets: `0.01, 0.02, 0.05, 0.10`. Threshold selector always chooses the largest empirical FPR not exceeding the target, then the smallest tied threshold.

The 5% results are the confirmatory extension of the Gate. The 1%, 2%, and 10% results are predeclared sensitivity analyses from the same fitted score arrays and are not used to redefine the primary Gate decision.

## Pressure test

For the `reference` capacity only, keep the model and clean month-5 thresholds fixed and add zero-mean Gaussian noise to numeric test features at standard-deviation multipliers `0, 0.25, 0.50, 1.00` relative to the fit-period column standard deviation. Do not corrupt `month` or `customer_age`; categorical columns remain unchanged. The noise seed is a deterministic function of model seed and severity. The test-oracle threshold is recalculated on the corrupted test labels only as a non-deployable reference; fixed and lag-1 policies use clean historical thresholds.

## Endpoints

Confirmatory endpoints at target 5%:

1. Capacity-ranking turnover between `test_oracle` and `fixed_m5` by variant and seed.
2. Per-policy predictive equality ratio, group-FPR difference, global FPR, recall, alert rate, Brier score, and log-loss.

Sensitivity endpoints: changes in the same metrics over target FPR levels. Pressure-test endpoints: recall, group-FPR difference, PE ratio, and alert rate by noise severity.

Score-level Brier/log-loss/calibration are reported as model-score diagnostics; they are not interpreted as threshold-policy effects. Policy effects are restricted to binary decision metrics and explicit operating-point cost.

## Statistical reporting

The confirmatory unit is variant × capacity × seed. Report mean, median, standard deviation, paired bootstrap 95% intervals, and per-variant direction counts. Ranking turnover is a descriptive paired proportion; report numerator and denominator. For target/severity curves, do not claim a single pooled causal effect. Use paired differences and confidence intervals; no post-hoc selection of the most favorable target or severity.

## Fairness and leakage controls

- `test_oracle` is explicitly labeled future-label reference and never called deployable.
- All preprocessing is fit on months 0–4 and rejects unseen future categorical values.
- `fixed_m5` uses month-5 labels only; `lag1` uses month 5 for month 6 and month 6 for month 7.
- No raw BAF file is modified, reweighted, filtered, or copied into the repository.
- BAF v1 reproduction results and BAF v2 full-experiment results remain separate.

## Stop and failure rules

Stop at 8 CPU-hours, 4 wall-hours, 12 GB RSS, 5 GB new artifacts, or any corrupted/incomplete data hash. A crashed fit is recorded and the run is incomplete; no partial run is used for claims. If all 90 fits finish, the raw outputs and manifest are immutable inputs to the reviewer-proof phase.
