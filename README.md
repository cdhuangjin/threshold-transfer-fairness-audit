# Threshold-transfer fairness audit toolkit

Reproducible code, frozen protocols, and data-provenance records for auditing threshold transfer under tabular distribution shift.

## Scope

The repository audits whether the source of a decision threshold changes fairness-sensitive model comparison under distribution shift while the fitted score model is held fixed. It contains a controlled temporal audit on the six-variant Bank Account Fraud (BAF) snapshot and an external geographic-shift replication on 2018 American Community Survey (ACS) data.

The repository does not contain the raw BAF or ACS files. See [DATA_PROVENANCE.md](DATA_PROVENANCE.md) for public sources, version notes, exact hashes, and data-use boundaries.

## Repository layout

- `threshold_gate/src/threshold_gate/`: reusable threshold-transfer data, metric, model, and analysis modules.
- `threshold_gate/`: entry points for the BAF audit, ACS replication, analysis, and figure generation.
- `protocols/`: frozen BAF, ACS, and survey-weight sensitivity protocols.

## Reproduction

Use Python 3.12 or later. Install the package and test dependencies with:

```text
python -m pip install -e threshold_gate
python -m pytest
```

Run the BAF audit with the local directory containing `Base.csv` and `Variant I.csv` through `Variant V.csv`:

```text
python threshold_gate/run_full_experiment.py --data-root <BAF_DATA_DIRECTORY> --output-dir results/baf_full --resume
python threshold_gate/analyze_full_experiment.py --input results/baf_full/full_raw_results.json --report results/baf_full/full_report.md --summary results/baf_full/full_summary.json
```

Run the ACS replication with the local 2018 1-Year PUMS directory:

```text
python threshold_gate/run_external_transfer.py --data-root <ACS_DATA_DIRECTORY> --output-dir results/acs_external --resume
python threshold_gate/analyze_external_transfer.py --input results/acs_external/external_raw_results.json --output-dir results/acs_external/analysis
```

The angle-bracketed values in these examples are local data directories and are not part of the repository. The protocols define the required files, filters, state pairs, splits, threshold policies, and endpoints. Run outputs should be kept in a local output directory or a separate results archive.
