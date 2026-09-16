# BAF Threshold-Transfer Gate Protocol v0.1

Date frozen: 2026-09-15
Status: frozen before model fitting
Scope: BAF v2 files already present under `D:\data数据集\BAF`.

## Research question

When a fraud benchmark contains temporal and group-conditional shift, how much do a test-label-selected 5% FPR operating point, a historical fixed calibration threshold, and a one-period-lag rolling threshold disagree on fairness and risk conclusions?

This is an operating-policy audit, not a new classifier. The source BAF v1 reproduction remains unchanged and its v1/v2 boundary is preserved.

## Hypotheses

- H1: selecting the threshold from future evaluation labels changes the estimated predictive-equality ratio relative to a leakage-safe historical threshold by at least 0.05 in at least two of six BAF variants.
- H2: a one-period-lag rolling threshold reduces historical-to-future operating-point error relative to the fixed month-5 threshold in at least four of six variants, without increasing majority-class log loss in more than four variants.
- H3: policy gaps are associated with month-to-month score-distribution shift and group-specific negative-score quantile movement, rather than only overall fraud prevalence. H3 is descriptive and does not claim causality.

## Data and version boundary

Use only these existing files:

| Variant | Path | Source | SHA-256 |
|---|---|---|---|
| Base | `D:\data数据集\BAF\Base.csv` | Kaggle `sgpjesus/bank-account-fraud-dataset-neurips-2022` | `7bf10a37ce07e72e14c1b09e5efee3d27261baff4facc7da767b0474dcf9b809` |
| Variant I | `D:\data数据集\BAF\Variant I.csv` | same | `48c637a255d1fa4515da4286ccb99251412a6d905937ec7c108838ddfc1674bd` |
| Variant II | `D:\data数据集\BAF\Variant II.csv` | same | `60bcd971cf28779abb183c3286990cff9a87275bd8b83d9f6054a396c95cc736` |
| Variant III | `D:\data数据集\BAF\Variant III.csv` | same | `64693e549cff7ef802cf043fc7c937839e5929966e33a0852953b3f4c4321339` |
| Variant IV | `D:\data数据集\BAF\Variant IV.csv` | same | `c6cddebbad34fa262a278deeb7985e7b669a64308d26e480d8eb85bfd08dac75` |
| Variant V | `D:\data数据集\BAF\Variant V.csv` | same | `470899bbef97c32a100528d63e863e1b1036df9501df6ef8140e7ec13b7365b4` |

Each file is read independently. The public Base semantic mismatch and unreconstructible source recipe remain limitations; no reweighting or filtering is allowed.

## Temporal and group contract

- Fit rows: months 0–4.
- Historical calibration: month 5.
- Final evaluation: months 6–7, reported both separately and pooled.
- Protected group: `older = customer_age > 50`; `customer_age == 50` is younger.
- Categorical encoding and every learned preprocessing object are fitted on months 0–4 only. Future categorical values cause a hard failure.

## Fixed models and policies

Train one LightGBM model for each of six variants, three capacities (`small`, `reference`, `large`), and three seeds `(42, 314, 2718)`: 54 fits total. No hyperparameter tuning, early stopping, resampling, or test-driven model selection is allowed.

Threshold selector: among finite ROC operating points, first select the largest empirical negative-class FPR that is `<= 0.05`, then select the smallest score threshold among tied points. This avoids the trivial infinite threshold with zero alerts and matches the source notebook's strict-below-target operating-point rule. A zero-negative or zero-positive calibration slice is invalid.

1. `test_oracle`: select on pooled months 6–7 labels. This is a leakage reference and is not a deployable policy.
2. `fixed_m5`: select on month 5 labels and apply the same threshold to months 6 and 7.
3. `lag1`: use the month-5 threshold for month 6; select a new threshold from month 6 labels and apply it to month 7. Month 6 is historical when month 7 is scored; the runner records this dependency explicitly.

## Endpoints and analysis

Primary effect unit: variant × model capacity × seed. Primary endpoint is `abs(PE_test_oracle - PE_fixed_m5)` on pooled months 6–7, where `PE=min(FPR_old,FPR_young)/max(FPR_old,FPR_young)` matches the BAF paper's predictive-equality ratio. A zero denominator is invalid, never smoothed.

Secondary endpoints are absolute error in group-FPR difference, absolute error in pooled recall, per-month recall and Brier score, majority/minority log loss, score-shift diagnostics, and capacity-ranking turnover between policies.

Bootstrap 2,000 times by resampling paired effect units within variant. Report mean, median, 95% percentile CI, per-variant direction counts, and raw paired units. No unregistered significance search is permitted.

## Decision rule

- `GO`: H1 reaches effect `>=0.05`, its paired bootstrap lower bound is `>0.02`, and at least two variants have the same direction; **or** capacity-ranking turnover between `test_oracle` and `fixed_m5` is `>=20%`.
- `BORDERLINE`: effect is between 0.02 and 0.05, CI crosses zero, or only one variant shows the direction. Permit one repair only: five seeds with the same models, data, policies, and endpoints.
- `NO-GO`: primary median `<0.02`, CI upper bound `<0.05`, and ranking turnover `<5%` across all variants.

The decision is recorded exactly once from the frozen endpoints. No post-hoc threshold, dataset, model, seed, or metric changes are allowed.

## Resource limits and evidence

Hard limits: 54 fits, 8 CPU-hours, 4 wall-hours, 12 GB peak RSS, 5 GB new artifacts, 0 GPU-hours. First limit reached stops the run and yields cost-Gate failure/uncertainty.

Required artifacts: raw JSON/CSV rows, summary CSV, data/config hashes, environment record, command, Gate report, and innovation-ledger entry. The result is a BAF v2 policy audit and must not be written as an exact reproduction of the BAF v1 paper.
