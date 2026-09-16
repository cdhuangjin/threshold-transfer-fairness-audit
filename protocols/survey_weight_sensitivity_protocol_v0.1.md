# WhyShift ACS Survey-Weight Sensitivity Protocol v0.1

## Question

Does the WhyShift ACS threshold-transfer result change materially when the
official person-level ACS weight `PWGTP` is used for operating-point selection
and held-out target evaluation?

This is a secondary estimand sensitivity. It does not replace or pool with the
primary unweighted ACS replication, and it is not a new method or an additional
independent dataset.

## Data and provenance

- Shared local data root: `D:\data数据集\WhyShift\acs\2018\1-Year`
- Local manifest: `D:\data数据集\WhyShift\MANIFEST.md`
- Source: WhyShift ACS 2018 1-Year files prepared from the public ACS release;
  the exact source URL and file hashes are recorded in the local manifest.
- State files and inherited SHA-256 hashes are the frozen constants in
  `threshold_gate.external_transfer`: CA `psam_p06.csv`, PR `psam_p72.csv`, MS
  `psam_p28.csv`, HI `psam_p15.csv`, and MA `psam_p25.csv`.
- Official filter: `AGEP > 16`, `PINCP > 100`, `WKHP > 0`, and `PWGTP >= 1`.
- Target label: `PINCP > 50000`.
- Protected field: `AGEP > 50` versus `AGEP <= 50`.
- Directed pairs: CA->PR, CA->HI, MS->HI, and MA->PR.

The runner verifies the five state-file hashes before loading them. No dataset
is downloaded by this sensitivity experiment.

## Experimental matrix

- Model capacities: the three frozen `MODEL_PARAMETERS` configurations from
  the primary threshold-gate experiment.
- Seeds: `42`, `314`, `2718`, `1618`, and `8675309`.
- Target FPRs: `0.01`, `0.02`, `0.05`, and `0.10`.
- Expected fits: `4 pairs x 3 capacities x 5 seeds = 60`.
- Expected metric rows: `60 fits x 4 FPRs x 2 policies = 480`.

## Weighting boundary

The score model is fit without sample weights, exactly as in the primary
external replication. The source train/calibration split remains the same
unweighted stratified split. `PWGTP` is used only for:

1. selecting the source-calibrated threshold from the source calibration
   negatives;
2. selecting the target-oracle reference threshold from target negatives; and
3. evaluating held-out target FPR, recall, alert rate, group FPRs, PE ratio,
   Brier score, and log loss.

The source-calibrated policy uses no target labels and is the deployable
analogue. The target-oracle policy uses target labels by design and is a
reference-only upper-bound diagnostic. Survey weighting changes the target
population estimand; it does not eliminate the geographic-pair overlap or make
the target-oracle policy deployable.

## Reproducibility and outputs

The independent runner writes a checkpoint after every fit and stores the raw
rows, code/protocol/data hashes, environment, split counts, weighting mode, and
completion status under a new sensitivity result directory. Resume is allowed
only when the expected matrix and inherited state-file hashes agree. The
primary `external_transfer_20260916_full` directory is never modified.

## Analysis boundary

The report gives weighted median absolute PE gap, pair-stratified bootstrap
intervals, capacity-ranking turnover, per-pair direction, and target-FPR
sensitivity. Differences from the unweighted estimates are reported as
sensitivity diagnostics. They are not a new Gate and are not interpreted as a
claim of universal fairness or exact BAF reproduction.
