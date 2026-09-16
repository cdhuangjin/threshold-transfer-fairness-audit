from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _primary_effects(rows: pd.DataFrame, target_fpr: float = 0.05) -> pd.DataFrame:
    keys = ["pair", "model_config", "seed"]
    pooled = rows.loc[rows["target_fpr"].eq(target_fpr) & rows["policy"].isin(("target_oracle", "source_calibrated")), keys + ["policy", "pe_ratio"]]
    wide = pooled.pivot_table(index=keys, columns="policy", values="pe_ratio", aggfunc="first").dropna().reset_index()
    wide["abs_pe_gap"] = (wide["target_oracle"] - wide["source_calibrated"]).abs()
    wide["signed_pe_gap"] = wide["target_oracle"] - wide["source_calibrated"]
    return wide


def _bootstrap_statistic(groups: list[np.ndarray], statistic, replicates: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sampled = [rng.choice(group, size=len(group), replace=True) for group in groups]
        values[index] = statistic(np.concatenate(sampled))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _ranking_units(rows: pd.DataFrame, target_fpr: float = 0.05) -> pd.DataFrame:
    filtered = rows.loc[rows["target_fpr"].eq(target_fpr) & rows["policy"].isin(("target_oracle", "source_calibrated")), ["pair", "seed", "model_config", "policy", "recall"]]
    records = []
    for (pair, seed), group in filtered.groupby(["pair", "seed"], sort=True):
        pivot = group.pivot(index="model_config", columns="policy", values="recall").dropna()
        oracle_order = tuple(pivot.sort_values("target_oracle", ascending=False, kind="stable").index)
        source_order = tuple(pivot.sort_values("source_calibrated", ascending=False, kind="stable").index)
        records.append({"pair": pair, "seed": int(seed), "changed": int(oracle_order != source_order), "oracle_order": "|".join(oracle_order), "source_order": "|".join(source_order)})
    return pd.DataFrame(records)


def _ranking_summary(rows: pd.DataFrame, replicates: int = 2_000, seed: int = 20260916) -> dict:
    units = _ranking_units(rows)
    changed = int(units["changed"].sum()) if not units.empty else 0
    groups = [group["changed"].to_numpy(dtype=float) for _, group in units.groupby("pair", sort=True)]
    ci_low, ci_high = _bootstrap_statistic(groups, np.mean, replicates, seed) if groups else (float("nan"), float("nan"))
    return {"units": int(len(units)), "changed_units": changed, "turnover": changed / len(units) if len(units) else None, "ci_low": ci_low, "ci_high": ci_high, "bootstrap_replicates": int(replicates), "bootstrap_seed": int(seed)}


def summarize(rows: pd.DataFrame, metadata: dict, replicates: int = 2_000, seed: int = 20260916) -> dict:
    effects = _primary_effects(rows)
    groups = [group["abs_pe_gap"].to_numpy(dtype=float) for _, group in effects.groupby("pair", sort=True)]
    ci_low, ci_high = _bootstrap_statistic(groups, np.median, replicates, seed) if groups else (float("nan"), float("nan"))
    sensitivity = []
    for target_fpr, group in rows.loc[rows["policy"].isin(("target_oracle", "source_calibrated"))].groupby("target_fpr", sort=True):
        f_effects = _primary_effects(group, target_fpr=float(target_fpr))
        sensitivity.append({"target_fpr": float(target_fpr), "n": int(len(f_effects)), "median_abs_pe_gap": float(f_effects["abs_pe_gap"].median()), "mean_abs_pe_gap": float(f_effects["abs_pe_gap"].mean())})
    pair_summary = []
    for pair, group in effects.groupby("pair", sort=True):
        pair_summary.append({"pair": pair, "n": int(len(group)), "median_abs_pe_gap": float(group["abs_pe_gap"].median()), "mean_abs_pe_gap": float(group["abs_pe_gap"].mean()), "oracle_higher_units": int((group["signed_pe_gap"] > 0).sum()), "oracle_lower_units": int((group["signed_pe_gap"] < 0).sum())})
    return {
        "protocol": "WhyShift external threshold-transfer replication protocol v0.1",
        "fit_count": metadata.get("fit_count"),
        "expected_fit_count": metadata.get("expected_fit_count"),
        "row_count": int(len(rows)),
        "complete": bool(metadata.get("stop_reason") is None and metadata.get("fit_count") == metadata.get("expected_fit_count")),
        "n_primary_units": int(len(effects)),
        "primary_median_abs_pe_gap": float(effects["abs_pe_gap"].median()),
        "primary_mean_abs_pe_gap": float(effects["abs_pe_gap"].mean()),
        "primary_bootstrap_ci_95": [ci_low, ci_high],
        "ranking_turnover": _ranking_summary(rows, replicates, seed),
        "pair_summary": pair_summary,
        "target_fpr_sensitivity": sensitivity,
        "interpretation": "external replication complete; effect size and sign are descriptive and need not match BAF; pair intervals are not population-level because state pairs overlap",
    }


def report(summary: dict, metadata: dict) -> str:
    ranking = summary["ranking_turnover"]
    lines = [
        "# WhyShift external threshold-transfer replication report",
        "",
        "Status: **%s**" % ("COMPLETE" if summary["complete"] else "INCOMPLETE"),
        "",
        "This is an external replication of the audit question on natural geographic ACS domain shifts. It is not a literal reproduction of BAF's synthetic data-generating mechanisms.",
        "",
        "## Execution",
        "",
        f"- Fits: `{summary['fit_count']}/{summary['expected_fit_count']}`; rows: `{summary['row_count']}`",
        f"- Runtime seconds: `{metadata.get('wall_seconds')}`",
        f"- Data root: `{metadata.get('data_root')}`",
        f"- Pair-stratified primary units: `{summary['n_primary_units']}`",
        "",
        "## Primary endpoint at target FPR = 0.05",
        "",
        f"- Median absolute PE gap: `{summary['primary_median_abs_pe_gap']:.6f}`",
        f"- Mean absolute PE gap: `{summary['primary_mean_abs_pe_gap']:.6f}`",
        f"- Pair-stratified bootstrap 95% CI: `[{summary['primary_bootstrap_ci_95'][0]:.6f}, {summary['primary_bootstrap_ci_95'][1]:.6f}]`",
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
        "## Evidence boundary",
        "",
        "The source-calibrated policy uses labels from a source-only calibration partition. The target-oracle policy uses target labels by design and is a reference, not a deployable procedure. ACS is geographic rather than temporal shift; survey weights are not used; state pairs reuse CA/PR/HI and are not fully independent; and this result is not pooled with the BAF main experiment.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    rows = pd.DataFrame(payload["rows"])
    summary = summarize(rows, payload["metadata"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "external_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    (args.output_dir / "external_report.md").write_text(report(summary, payload["metadata"]), encoding="utf-8")
    pd.DataFrame(summary["pair_summary"]).to_csv(args.output_dir / "external_pair_summary.csv", index=False)
    pd.DataFrame(summary["target_fpr_sensitivity"]).to_csv(args.output_dir / "external_target_fpr_sensitivity.csv", index=False)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
