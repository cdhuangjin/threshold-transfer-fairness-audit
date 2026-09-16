from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from threshold_gate.analysis import capacity_ranking_turnover, primary_effects, summarize_gate


EXPECTED_FITS = 90
POLICIES = ("test_oracle", "fixed_m5", "lag1")


def load_full_results(path: Path, expected_fit_count: int = EXPECTED_FITS) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    rows = payload.get("rows", [])
    fit_count = int(metadata.get("fit_count", -1))
    if fit_count != expected_fit_count or int(metadata.get("expected_fit_count", -1)) != expected_fit_count:
        raise ValueError(f"full experiment is incomplete: fit_count={fit_count}, expected={expected_fit_count}")
    if metadata.get("stop_reason") is not None:
        raise ValueError(f"full experiment stopped early: {metadata['stop_reason']}")
    if len(rows) == 0:
        raise ValueError("full experiment contains no rows")
    return payload


def confirmatory_rows(rows: pd.DataFrame, target_fpr: float = 0.05) -> pd.DataFrame:
    """Return clean, pooled-test rows at the pre-registered primary target."""
    mask = (
        rows["scenario"].eq("clean")
        & rows["period"].eq("pooled_test")
        & rows["target_fpr"].eq(float(target_fpr))
    )
    return rows.loc[mask].copy()


def target_sensitivity(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    clean = rows.loc[rows["scenario"].eq("clean") & rows["period"].eq("pooled_test")]
    for target, group in clean.groupby("target_fpr", sort=True):
        effects = primary_effects(group)
        ranking = capacity_ranking_turnover(group)
        records.append({
            "target_fpr": float(target),
            "n_primary_units": int(len(effects)),
            "median_abs_pe_gap": float(effects["abs_pe_gap"].median()) if not effects.empty else None,
            "mean_abs_pe_gap": float(effects["abs_pe_gap"].mean()) if not effects.empty else None,
            "ranking_units": int(ranking["units"]),
            "ranking_changed_units": int(ranking["changed_units"]),
            "ranking_turnover": ranking["turnover"],
        })
    return pd.DataFrame(records).sort_values("target_fpr").reset_index(drop=True)


def robustness_summary(rows: pd.DataFrame, target_fpr: float = 0.05) -> pd.DataFrame:
    selected = rows.loc[
        rows["model_config"].eq("reference")
        & rows["period"].eq("pooled_test")
        & rows["target_fpr"].eq(float(target_fpr))
        & rows["policy"].isin(POLICIES)
    ]
    if selected.empty:
        return pd.DataFrame()
    columns = ["global_fpr", "recall", "pe_ratio", "fpr_difference", "alert_rate"]
    grouped = selected.groupby(["scenario", "policy"], sort=True)[columns].agg(["median", "mean"]).reset_index()
    grouped.columns = [
        "_".join(str(part) for part in column if str(part) != "") if isinstance(column, tuple) else str(column)
        for column in grouped.columns
    ]
    return grouped.rename(columns={"scenario_": "scenario", "policy_": "policy"})


def mechanism_summary(rows: pd.DataFrame, target_fpr: float = 0.05) -> pd.DataFrame:
    selected = rows.loc[
        rows["model_config"].eq("reference")
        & rows["scenario"].eq("clean")
        & rows["target_fpr"].eq(float(target_fpr))
        & rows["policy"].eq("fixed_m5")
        & rows["period"].isin(("month6", "month7", "pooled_test"))
    ]
    if selected.empty:
        return pd.DataFrame()
    columns = ["threshold", "negative_score_q95", "negative_score_q95_older", "negative_score_q95_younger", "global_fpr", "recall", "pe_ratio", "alert_rate"]
    return selected.groupby("period", sort=False)[columns].median().reset_index()


def runtime_summary(rows: pd.DataFrame) -> pd.DataFrame:
    fit_rows = rows.drop_duplicates(["variant", "model_config", "seed"])
    return (
        fit_rows.groupby("model_config", sort=True)["elapsed_seconds"]
        .agg(["count", "median", "mean", "max"])
        .reset_index()
    )


def build_summary(payload: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = pd.DataFrame(payload["rows"])
    main_rows = confirmatory_rows(rows)
    main = summarize_gate(main_rows)
    sensitivity = target_sensitivity(rows)
    robustness = robustness_summary(rows)
    mechanism = mechanism_summary(rows)
    runtime = runtime_summary(rows)
    summary = {
        "analysis": "BAF threshold-transfer full experiment",
        "decision": main["decision"],
        "primary": main,
        "target_sensitivity": sensitivity.to_dict(orient="records"),
        "reference_noise_robustness": robustness.to_dict(orient="records"),
        "mechanism": mechanism.to_dict(orient="records"),
        "runtime_by_model": runtime.to_dict(orient="records"),
        "metadata": payload["metadata"],
    }
    return summary, sensitivity, robustness, mechanism, runtime


def markdown_report(summary: dict) -> str:
    primary = summary["primary"]
    metadata = summary["metadata"]
    lines = [
        "# BAF Threshold-Transfer Full Experiment Report",
        "",
        f"Decision: **{summary['decision']}**",
        "",
        "This report evaluates the frozen threshold-transfer protocol. The v1 BAF reproduction boundary remains separate: this is not an exact reproduction of the source paper.",
        "",
        "## Execution",
        "",
        f"- Fits: `{metadata.get('fit_count')}/{metadata.get('expected_fit_count')}`",
        f"- Rows: `{metadata.get('row_count')}`",
        f"- Wall seconds: `{metadata.get('wall_seconds')}`",
        f"- CPU seconds: `{metadata.get('cpu_seconds')}`",
        f"- Peak RSS GiB: `{metadata.get('peak_rss_gib')}`",
        f"- Data root: `{metadata.get('data_root')}`",
        "",
        "## Confirmatory result: clean, pooled test, target FPR 0.05",
        "",
        f"- Primary units: `{primary['n_primary_units']}`",
        f"- Median absolute PE gap: `{primary['primary_median_abs_pe_gap']:.6f}`",
        f"- Stratified bootstrap 95% CI: `[{primary['bootstrap_ci_95'][0]:.6f}, {primary['bootstrap_ci_95'][1]:.6f}]`",
        f"- H1 effect path: `{primary['h1_pass']}`; capacity-ranking path: `{primary['h1_ranking_path_pass']}`",
        f"- Capacity-ranking turnover: `{json.dumps(primary['ranking_turnover'], ensure_ascii=False)}`",
        f"- Stratified bootstrap 95% CI for turnover: `[{primary['ranking_turnover_ci']['ci_low']:.3f}, {primary['ranking_turnover_ci']['ci_high']:.3f}]`",
        "",
        "## Per-variant direction and dispersion",
        "",
        "| Variant | n | Median abs PE gap | Mean abs PE gap | SD abs PE gap | Oracle higher | Oracle lower | Tie |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in primary["variant_summary"]:
        lines.append(
            f"| {item['variant']} | {item['n']} | {item['median_abs_pe_gap']:.6f} | {item['mean_abs_pe_gap']:.6f} | {item['sd_abs_pe_gap']:.6f} | {item['oracle_higher_units']} | {item['oracle_lower_units']} | {item['tie_units']} |"
        )
    lines += [
        "",
        "## Target-FPR sensitivity",
        "",
        "| Target FPR | Primary units | Median abs PE gap | Changed ranking units | Ranking turnover |",
        "|---:|---:|---:|---:|---:|",
    ]
    for item in summary["target_sensitivity"]:
        lines.append(
            f"| {item['target_fpr']:.2f} | {item['n_primary_units']} | {item['median_abs_pe_gap']:.6f} | {item['ranking_changed_units']}/{item['ranking_units']} | {item['ranking_turnover']:.3f} |"
        )
    lines += ["", "## Reference-model test-only noise robustness", "", "| Scenario | Policy | Median FPR | Median recall | Median PE ratio | Median FPR difference | Median alert rate |", "|---|---|---:|---:|---:|---:|---:|"]
    for item in summary["reference_noise_robustness"]:
        lines.append(
            f"| {item['scenario']} | {item['policy']} | {item['global_fpr_median']:.6f} | {item['recall_median']:.6f} | {item['pe_ratio_median'] if item['pe_ratio_median'] is not None else 'NA'} | {item['fpr_difference_median']:.6f} | {item['alert_rate_median']:.6f} |"
        )
    lines += ["", "## Mechanism diagnostics", "", "These are descriptive score-tail and operating-point diagnostics; Brier/log-loss are not treated as threshold-policy effects.", "", "| Period | Median threshold | Negative-score q95 | Older q95 | Younger q95 | Median FPR | Median recall | Median alert rate |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for item in summary["mechanism"]:
        lines.append(
            f"| {item['period']} | {item['threshold']:.6f} | {item['negative_score_q95']:.6f} | {item['negative_score_q95_older']:.6f} | {item['negative_score_q95_younger']:.6f} | {item['global_fpr']:.6f} | {item['recall']:.6f} | {item['alert_rate']:.6f} |"
        )
    lines += [
        "",
        "## Evidence boundary",
        "",
        "The test-oracle threshold uses future labels by design and is a reference benchmark, not a deployable policy. Fixed-m5 and lag-1 are the historically available policies. A positive ranking-turnover result supports a policy-sensitivity audit contribution; it does not establish superiority of a new classifier or causal fairness improvement.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the complete BAF threshold-transfer full experiment.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--sensitivity", type=Path, required=True)
    parser.add_argument("--robustness", type=Path, required=True)
    parser.add_argument("--mechanism", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args()
    payload = load_full_results(args.input)
    summary, sensitivity, robustness, mechanism, runtime = build_summary(payload)
    args.report.write_text(markdown_report(summary), encoding="utf-8")
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    sensitivity.to_csv(args.sensitivity, index=False)
    robustness.to_csv(args.robustness, index=False)
    mechanism.to_csv(args.mechanism, index=False)
    runtime.to_csv(args.runtime, index=False)
    print(json.dumps({"decision": summary["decision"], "fit_count": payload["metadata"]["fit_count"], "rows": len(payload["rows"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
