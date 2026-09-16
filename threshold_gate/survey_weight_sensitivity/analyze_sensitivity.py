from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PRIMARY_KEYS = ["pair", "model_config", "seed"]
POLICIES = ("target_oracle", "source_calibrated")
METRICS = ("global_fpr", "recall", "alert_rate", "fpr_older", "fpr_younger", "fpr_difference", "pe_ratio", "brier", "logloss")


def _primary_effects(rows: pd.DataFrame, target_fpr: float = 0.05) -> pd.DataFrame:
    selected = rows.loc[rows["target_fpr"].eq(target_fpr) & rows["policy"].isin(POLICIES), PRIMARY_KEYS + ["policy", "pe_ratio"]]
    wide = selected.pivot_table(index=PRIMARY_KEYS, columns="policy", values="pe_ratio", aggfunc="first").dropna().reset_index()
    wide["abs_pe_gap"] = (wide["target_oracle"] - wide["source_calibrated"]).abs()
    wide["signed_pe_gap"] = wide["target_oracle"] - wide["source_calibrated"]
    return wide


def _bootstrap_statistic(groups: list[np.ndarray], statistic, replicates: int, seed: int) -> tuple[float | None, float | None]:
    if not groups:
        return None, None
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sampled = [rng.choice(group, size=len(group), replace=True) for group in groups]
        values[index] = statistic(np.concatenate(sampled))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _ranking_units(rows: pd.DataFrame, target_fpr: float = 0.05) -> pd.DataFrame:
    selected = rows.loc[rows["target_fpr"].eq(target_fpr) & rows["policy"].isin(POLICIES), ["pair", "seed", "model_config", "policy", "recall"]]
    records = []
    for (pair, seed), group in selected.groupby(["pair", "seed"], sort=True):
        pivot = group.pivot(index="model_config", columns="policy", values="recall").dropna()
        if pivot.empty or not all(policy in pivot.columns for policy in POLICIES):
            continue
        oracle_order = tuple(pivot.sort_values("target_oracle", ascending=False, kind="stable").index)
        source_order = tuple(pivot.sort_values("source_calibrated", ascending=False, kind="stable").index)
        records.append(
            {
                "pair": pair,
                "seed": int(seed),
                "changed": int(oracle_order != source_order),
                "oracle_order": "|".join(oracle_order),
                "source_order": "|".join(source_order),
            }
        )
    return pd.DataFrame(records)


def _ranking_summary(rows: pd.DataFrame, replicates: int = 2_000, seed: int = 20260916) -> dict:
    units = _ranking_units(rows)
    changed = int(units["changed"].sum()) if not units.empty else 0
    groups = [group["changed"].to_numpy(dtype=float) for _, group in units.groupby("pair", sort=True)]
    ci_low, ci_high = _bootstrap_statistic(groups, np.mean, replicates, seed)
    return {
        "units": int(len(units)),
        "changed_units": changed,
        "turnover": changed / len(units) if len(units) else None,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "bootstrap_replicates": int(replicates),
        "bootstrap_seed": int(seed),
    }


def _target_fpr_sensitivity(rows: pd.DataFrame) -> list[dict]:
    records = []
    for target_fpr, group in rows.loc[rows["policy"].isin(POLICIES)].groupby("target_fpr", sort=True):
        effects = _primary_effects(group, target_fpr=float(target_fpr))
        records.append(
            {
                "target_fpr": float(target_fpr),
                "n": int(len(effects)),
                "median_abs_pe_gap": float(effects["abs_pe_gap"].median()) if not effects.empty else None,
                "mean_abs_pe_gap": float(effects["abs_pe_gap"].mean()) if not effects.empty else None,
            }
        )
    return records


def _pair_summary(effects: pd.DataFrame) -> list[dict]:
    records = []
    for pair, group in effects.groupby("pair", sort=True):
        records.append(
            {
                "pair": pair,
                "n": int(len(group)),
                "median_abs_pe_gap": float(group["abs_pe_gap"].median()),
                "mean_abs_pe_gap": float(group["abs_pe_gap"].mean()),
                "oracle_higher_units": int((group["signed_pe_gap"] > 0).sum()),
                "oracle_lower_units": int((group["signed_pe_gap"] < 0).sum()),
            }
        )
    return records


def _compare_weighting(weighted_rows: pd.DataFrame, unweighted_rows: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    keys = ["pair", "source", "target", "model_config", "seed", "policy", "target_fpr"]
    left = weighted_rows.loc[:, keys + list(METRICS)].copy()
    right = unweighted_rows.loc[:, keys + list(METRICS)].copy()
    merged = left.merge(right, on=keys, how="outer", suffixes=("_weighted", "_unweighted"), validate="one_to_one")
    for metric in METRICS:
        merged[f"delta_{metric}"] = merged[f"{metric}_weighted"] - merged[f"{metric}_unweighted"]
    summary = []
    for (target_fpr, policy), group in merged.groupby(["target_fpr", "policy"], sort=True):
        item = {"target_fpr": float(target_fpr), "policy": policy, "n": int(len(group))}
        for metric in ("global_fpr", "recall", "pe_ratio", "brier", "logloss"):
            values = group[f"delta_{metric}"].dropna()
            item[f"median_delta_{metric}"] = float(values.median()) if not values.empty else None
            item[f"mean_delta_{metric}"] = float(values.mean()) if not values.empty else None
        summary.append(item)
    return merged, summary


def summarize(rows: pd.DataFrame, metadata: dict, unweighted_rows: pd.DataFrame, replicates: int = 2_000, seed: int = 20260916) -> tuple[dict, pd.DataFrame, list[dict]]:
    effects = _primary_effects(rows)
    groups = [group["abs_pe_gap"].to_numpy(dtype=float) for _, group in effects.groupby("pair", sort=True)]
    ci_low, ci_high = _bootstrap_statistic(groups, np.median, replicates, seed)
    comparison, comparison_summary = _compare_weighting(rows, unweighted_rows)
    summary = {
        "protocol": "WhyShift ACS survey-weight sensitivity protocol v0.1",
        "fit_count": metadata.get("fit_count"),
        "expected_fit_count": metadata.get("expected_fit_count"),
        "row_count": int(len(rows)),
        "complete": bool(metadata.get("stop_reason") is None and metadata.get("fit_count") == metadata.get("expected_fit_count")),
        "n_primary_units": int(len(effects)),
        "primary_median_abs_pe_gap": float(effects["abs_pe_gap"].median()),
        "primary_mean_abs_pe_gap": float(effects["abs_pe_gap"].mean()),
        "primary_bootstrap_ci_95": [ci_low, ci_high],
        "ranking_turnover": _ranking_summary(rows, replicates, seed),
        "pair_summary": _pair_summary(effects),
        "target_fpr_sensitivity": _target_fpr_sensitivity(rows),
        "weighted_vs_unweighted_summary": comparison_summary,
        "unweighted_primary_median_abs_pe_gap": float(_primary_effects(unweighted_rows)["abs_pe_gap"].median()),
        "interpretation": "secondary survey-weight sensitivity; weighted estimands do not replace the unweighted confirmatory ACS result and state pairs remain overlapping",
    }
    return summary, comparison, comparison_summary


def report(summary: dict, metadata: dict) -> str:
    ranking = summary["ranking_turnover"]
    ci = summary["primary_bootstrap_ci_95"]
    lines = [
        "# WhyShift ACS survey-weight sensitivity report",
        "",
        "Status: **%s**" % ("COMPLETE" if summary["complete"] else "INCOMPLETE"),
        "",
        "This is a secondary sensitivity analysis of the external ACS threshold-transfer replication. The score models and source/calibration split are unchanged; `PWGTP` is used for threshold selection and held-out target evaluation.",
        "",
        "## Execution",
        "",
        f"- Fits: `{summary['fit_count']}/{summary['expected_fit_count']}`; rows: `{summary['row_count']}`",
        f"- Data root: `{metadata.get('data_root')}`",
        f"- Weighting boundary: `{metadata.get('weighting_boundary')}`",
        "",
        "## Weighted endpoint at target FPR = 0.05",
        "",
        f"- Median absolute PE gap: `{summary['primary_median_abs_pe_gap']:.6f}`",
        f"- Mean absolute PE gap: `{summary['primary_mean_abs_pe_gap']:.6f}`",
        f"- Pair-stratified bootstrap 95% CI: `[{ci[0]:.6f}, {ci[1]:.6f}]`",
        f"- Unweighted primary median absolute PE gap: `{summary['unweighted_primary_median_abs_pe_gap']:.6f}`",
        "",
        "| Pair | n | Median abs PE gap | Mean abs PE gap | Oracle higher | Oracle lower |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in summary["pair_summary"]:
        lines.append(f"| {item['pair']} | {item['n']} | {item['median_abs_pe_gap']:.6f} | {item['mean_abs_pe_gap']:.6f} | {item['oracle_higher_units']} | {item['oracle_lower_units']} |")
    lines += [
        "",
        "## Capacity-ranking turnover",
        "",
        f"- Units: `{ranking['units']}`; changed: `{ranking['changed_units']}`; turnover: `{ranking['turnover']:.6f}`",
        f"- Pair-stratified bootstrap 95% CI: `[{ranking['ci_low']:.6f}, {ranking['ci_high']:.6f}]`",
        "",
        "## Target-FPR sensitivity",
        "",
        "| Target FPR | n | Median abs PE gap | Mean abs PE gap |",
        "|---:|---:|---:|---:|",
    ]
    for item in summary["target_fpr_sensitivity"]:
        lines.append(f"| {item['target_fpr']:.2f} | {item['n']} | {item['median_abs_pe_gap']:.6f} | {item['mean_abs_pe_gap']:.6f} |")
    lines += [
        "",
        "## Weighting comparison",
        "",
        "The weighted-minus-unweighted table is a diagnostic of the population estimand, not a new confirmatory endpoint.",
        "",
        "| Target FPR | Policy | Median delta PE ratio | Mean delta PE ratio | Median delta logloss |",
        "|---:|---|---:|---:|---:|",
    ]
    for item in summary["weighted_vs_unweighted_summary"]:
        lines.append(f"| {item['target_fpr']:.2f} | {item['policy']} | {item['median_delta_pe_ratio']:.6f} | {item['mean_delta_pe_ratio']:.6f} | {item['median_delta_logloss']:.6f} |")
    lines += [
        "",
        "## Evidence boundary",
        "",
        "Survey weighting changes the target-population estimand but does not create a second independent dataset or remove state-pair overlap. The target-oracle policy still uses target labels and is reference-only; the source-calibrated policy remains the deployable analogue. These results are not pooled with the BAF experiment and do not support a universal fairness claim.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze WhyShift ACS survey-weight sensitivity results.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--unweighted-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    weighted_payload = json.loads(args.input.read_text(encoding="utf-8"))
    unweighted_payload = json.loads(args.unweighted_input.read_text(encoding="utf-8"))
    weighted_rows = pd.DataFrame(weighted_payload["rows"])
    unweighted_rows = pd.DataFrame(unweighted_payload["rows"])
    summary, comparison, comparison_summary = summarize(weighted_rows, weighted_payload["metadata"], unweighted_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "weighted_external_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    (args.output_dir / "sensitivity_report.md").write_text(report(summary, weighted_payload["metadata"]), encoding="utf-8")
    pd.DataFrame(summary["pair_summary"]).to_csv(args.output_dir / "weighted_pair_summary.csv", index=False)
    pd.DataFrame(summary["target_fpr_sensitivity"]).to_csv(args.output_dir / "weighted_target_fpr_sensitivity.csv", index=False)
    comparison.to_csv(args.output_dir / "weighted_vs_unweighted.csv", index=False, float_format="%.15g")
    pd.DataFrame(comparison_summary).to_csv(args.output_dir / "weighted_vs_unweighted_summary.csv", index=False, float_format="%.15g")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
