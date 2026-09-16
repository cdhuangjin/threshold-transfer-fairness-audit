# WhyShift external threshold-transfer replication protocol v0.1

## Purpose

This is an external replication of the BAF threshold-transfer audit on a natural tabular domain-shift benchmark. It is confirmatory for transportability of the *audit question*—whether a threshold calibrated under one distribution changes group-sensitive model comparison on another distribution—not a claim that BAF's synthetic mechanisms are reproduced.

## Data provenance

- Shared root: `D:\data数据集\WhyShift\acs\2018\1-Year`.
- Public source named by WhyShift/Folktables: U.S. Census ACS 2018 1-Year PUMS.
- Local manifest: `D:\data数据集\WhyShift\MANIFEST.md`.
- Files used: `psam_p06.csv` (CA), `psam_p72.csv` (PR), `psam_p28.csv` (MS), `psam_p15.csv` (HI), and `psam_p25.csv` (MA).
- The exact SHA-256 values are taken from the shared-data manifest and are copied into the run artifact.

## Task and preprocessing

Use the official `folktables.ACSIncome` feature definition: `AGEP, COW, SCHL, MAR, OCCP, POBP, RELP, WKHP, SEX, RAC1P`; target `PINCP > 50,000`; official filter `AGEP > 16`, `PINCP > 100`, `WKHP > 0`, `PWGTP >= 1`. The runner loads only these columns plus no hidden target-derived fields. Rows are unweighted for model fitting and endpoint computation, matching the BAF audit's unweighted convention. The protected audit group is `AGEP > 50` (older) versus `AGEP <= 50` (younger); age remains an input feature, as in the BAF audit.

## Domain pairs and splits

The frozen directed pairs are `CA->PR`, `CA->HI`, `MS->HI`, and `MA->PR`. Source rows are split into an 80% training partition and a disjoint 20% calibration partition using a label/group-stratified deterministic split. The entire target state is held out for evaluation. The split seed is one of five fixed seeds: `42, 314, 2718, 1618, 8675309`.

## Models and policies

Fit the same three fixed LightGBM capacities used in the BAF full experiment (`small`, `reference`, `large`) on source-training rows only. The threshold target FPRs are `.01, .02, .05, .10`.

- `target_oracle`: threshold chosen from target negative scores; future target labels make this a reference-only policy.
- `source_calibrated`: threshold chosen from source-calibration negative scores and transferred to the target; this is the deployable analogue.

For each pair × capacity × seed × target-FPR unit, report target global FPR, recall, group FPRs, predictive-equity ratio (PE ratio = lower group FPR / higher group FPR), Brier score, log loss, and alert rate. Primary analysis uses target-FPR `.05` and pooled target rows.

## Confirmatory endpoints

1. Absolute PE gap between `target_oracle` and `source_calibrated`, summarized by the median across 60 pair × capacity × seed units with a pair-stratified bootstrap 95% interval.
2. Capacity-ranking turnover: whether the recall ordering of the three capacities changes between the two policies, summarized across 20 pair × seed units with a pair-stratified bootstrap 95% interval.

Target-FPR sensitivity (`.01/.02/.05/.10`) is secondary. No endpoint uses target labels to train a model. The oracle threshold is not a deployable result and is reported only as a reference.

The four directed pairs are prespecified but are not fully independent clusters: CA is reused as a source and PR/HI are reused as targets. Pair-stratified intervals are therefore descriptive uncertainty summaries for this fixed collection of shifts, not population-level confidence intervals.

## Acceptance and interpretation

This external replication is successful if it completes all 60 fits, produces finite outputs for all 4 pairs × 4 target-FPRs, and the pair-stratified uncertainty is reported. It is not required to reproduce the BAF effect size or sign. Any positive or null result is retained; no post-hoc pair, group, threshold, or capacity is added after inspecting the outcome.

## Reproducibility limits

WhyShift ACS is a natural geographic shift, not a temporal stream and not the BAF synthetic benchmark. The source calibration partition is an operational proxy for historical labeled calibration. Survey weights are not used in this transfer audit. These boundaries prevent the external result from being presented as universal evidence or as a literal reproduction of the mother paper.
