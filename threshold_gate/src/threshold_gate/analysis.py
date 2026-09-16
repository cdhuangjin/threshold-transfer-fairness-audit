from __future__ import annotations

import json
from typing import Iterable

import numpy as np
import pandas as pd


KEYS = ["variant", "model_config", "seed"]


def primary_effects(rows: pd.DataFrame) -> pd.DataFrame:
    pooled = rows.loc[
        rows["period"].eq("pooled_test") & rows["policy"].isin(("test_oracle", "fixed_m5")),
        KEYS + ["policy", "pe_ratio"],
    ]
    wide = pooled.pivot_table(index=KEYS, columns="policy", values="pe_ratio", aggfunc="first").reset_index()
    wide = wide.dropna(subset=["test_oracle", "fixed_m5"]).copy()
    wide["abs_pe_gap"] = (wide["test_oracle"] - wide["fixed_m5"]).abs()
    wide["signed_pe_gap"] = wide["test_oracle"] - wide["fixed_m5"]
    output = wide[KEYS + ["abs_pe_gap", "signed_pe_gap"]].copy()
    output["model_config"] = pd.Categorical(output["model_config"], categories=["small", "reference", "large"], ordered=True)
    return output.sort_values(KEYS).reset_index(drop=True).assign(model_config=lambda x: x["model_config"].astype(str))


def capacity_ranking_units(rows: pd.DataFrame) -> pd.DataFrame:
    pooled = rows.loc[
        rows["period"].eq("pooled_test") & rows["policy"].isin(("test_oracle", "fixed_m5")),
        ["variant", "seed", "model_config", "policy", "recall"],
    ]
    records = []
    for (variant, seed), group in pooled.groupby(["variant", "seed"], sort=True):
        pivot = group.pivot(index="model_config", columns="policy", values="recall").dropna()
        if not {"test_oracle", "fixed_m5"}.issubset(pivot.columns) or len(pivot) < 2:
            continue
        oracle_order = tuple(pivot.sort_values("test_oracle", ascending=False, kind="stable").index)
        fixed_order = tuple(pivot.sort_values("fixed_m5", ascending=False, kind="stable").index)
        records.append({
            "variant": variant,
            "seed": int(seed),
            "changed": int(oracle_order != fixed_order),
            "oracle_order": "|".join(oracle_order),
            "fixed_order": "|".join(fixed_order),
        })
    return pd.DataFrame(records)


def capacity_ranking_turnover(rows: pd.DataFrame) -> dict[str, float | int]:
    units = capacity_ranking_units(rows)
    changed = int(units["changed"].sum()) if not units.empty else 0
    count = int(len(units))
    return {"units": count, "changed_units": changed, "turnover": changed / count if count else None}


def ranking_turnover_ci(rows: pd.DataFrame, replicates: int = 2_000, seed: int = 20260916) -> dict[str, float | int]:
    units = capacity_ranking_units(rows)
    count = int(len(units))
    changed = int(units["changed"].sum()) if count else 0
    turnover = changed / count if count else None
    if not count:
        return {"units": 0, "changed_units": 0, "turnover": None, "ci_low": None, "ci_high": None, "bootstrap_replicates": int(replicates), "bootstrap_seed": int(seed)}
    rng = np.random.default_rng(seed)
    groups = [group["changed"].to_numpy(dtype=float) for _, group in units.groupby("variant", sort=True)]
    statistics = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sampled = [rng.choice(values, size=len(values), replace=True) for values in groups]
        statistics[index] = np.concatenate(sampled).mean()
    return {
        "units": count,
        "changed_units": changed,
        "turnover": float(turnover),
        "ci_low": float(np.quantile(statistics, 0.025)),
        "ci_high": float(np.quantile(statistics, 0.975)),
        "bootstrap_replicates": int(replicates),
        "bootstrap_seed": int(seed),
    }


def _bootstrap_median_by_variant(effects: pd.DataFrame, replicates: int, seed: int) -> tuple[float, float]:
    if effects.empty:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    groups = [group["abs_pe_gap"].to_numpy(dtype=float) for _, group in effects.groupby("variant", sort=True)]
    statistics = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sampled = [rng.choice(values, size=len(values), replace=True) for values in groups]
        statistics[index] = np.median(np.concatenate(sampled))
    return float(np.quantile(statistics, 0.025)), float(np.quantile(statistics, 0.975))


def _policy_error_table(rows: pd.DataFrame) -> pd.DataFrame:
    pooled = rows.loc[rows["period"].eq("pooled_test"), KEYS + ["policy", "pe_ratio", "logloss_older", "logloss_younger"]]
    wide = pooled.pivot_table(index=KEYS, columns="policy", values=["pe_ratio", "logloss_older", "logloss_younger"], aggfunc="first")
    wide = wide.dropna(subset=[("pe_ratio", "test_oracle")]).copy()
    result = []
    for policy in ("fixed_m5", "lag1"):
        if ("pe_ratio", policy) not in wide:
            continue
        part = wide.reset_index()[KEYS].copy()
        part["policy"] = policy
        part["pe_error_vs_oracle"] = (
            wide[("pe_ratio", policy)] - wide[("pe_ratio", "test_oracle")]
        ).abs().to_numpy()
        part["older_logloss_delta_vs_oracle"] = (
            wide[("logloss_older", policy)] - wide[("logloss_older", "test_oracle")]
        ).to_numpy()
        part["younger_logloss_delta_vs_oracle"] = (
            wide[("logloss_younger", policy)] - wide[("logloss_younger", "test_oracle")]
        ).to_numpy()
        result.append(part)
    return pd.concat(result, ignore_index=True) if result else pd.DataFrame()


def summarize_gate(rows: pd.DataFrame, bootstrap_replicates: int = 2_000, bootstrap_seed: int = 20260915) -> dict:
    effects = primary_effects(rows)
    ci_low, ci_high = _bootstrap_median_by_variant(effects, bootstrap_replicates, bootstrap_seed)
    variant_summary = []
    for variant, group in effects.groupby("variant", sort=True):
        variant_summary.append({
            "variant": variant,
            "n": int(len(group)),
            "median_abs_pe_gap": float(group["abs_pe_gap"].median()),
            "mean_abs_pe_gap": float(group["abs_pe_gap"].mean()),
            "sd_abs_pe_gap": float(group["abs_pe_gap"].std(ddof=1)) if len(group) > 1 else 0.0,
            "median_signed_pe_gap": float(group["signed_pe_gap"].median()),
            "oracle_higher_units": int((group["signed_pe_gap"] > 0).sum()),
            "oracle_lower_units": int((group["signed_pe_gap"] < 0).sum()),
            "tie_units": int((group["signed_pe_gap"] == 0).sum()),
            "direction": "oracle_higher" if group["signed_pe_gap"].median() > 0 else ("oracle_lower" if group["signed_pe_gap"].median() < 0 else "tie"),
        })
    ranking = capacity_ranking_turnover(rows)
    ranking_ci = ranking_turnover_ci(rows)
    qualifying = [x["variant"] for x in variant_summary if x["median_abs_pe_gap"] >= 0.05]
    h1 = len(qualifying) >= 2 and ci_low > 0.02
    h2_table = _policy_error_table(rows)
    h2_summary = []
    for policy, group in h2_table.groupby("policy", sort=True) if not h2_table.empty else []:
        h2_summary.append({
            "policy": policy,
            "n": int(len(group)),
            "median_pe_error_vs_oracle": float(group["pe_error_vs_oracle"].median()),
            "mean_pe_error_vs_oracle": float(group["pe_error_vs_oracle"].mean()),
            "median_older_logloss_delta_vs_oracle": float(group["older_logloss_delta_vs_oracle"].median()),
            "median_younger_logloss_delta_vs_oracle": float(group["younger_logloss_delta_vs_oracle"].median()),
        })
    primary_median = float(effects["abs_pe_gap"].median()) if not effects.empty else float("nan")
    h1_ranking = ranking["turnover"] is not None and ranking["turnover"] >= 0.20
    no_go = bool(primary_median < 0.02 and ci_high < 0.05 and (ranking["turnover"] or 0.0) < 0.05)
    decision = "GO" if bool(h1 or h1_ranking) else ("NO-GO" if no_go else "BORDERLINE")
    return {
        "n_primary_units": int(len(effects)),
        "primary_median_abs_pe_gap": primary_median,
        "primary_mean_abs_pe_gap": float(effects["abs_pe_gap"].mean()) if not effects.empty else float("nan"),
        "bootstrap_ci_95": [ci_low, ci_high],
        "h1_qualifying_variants": qualifying,
        "h1_pass": bool(h1),
        "h1_ranking_path_pass": bool(h1_ranking),
        "ranking_turnover": ranking,
        "ranking_turnover_ci": ranking_ci,
        "h2": h2_summary,
        "variant_summary": variant_summary,
        "decision": decision,
        "bootstrap_replicates": int(bootstrap_replicates),
        "bootstrap_seed": int(bootstrap_seed),
    }


def markdown_report(summary: dict, run_metadata: dict) -> str:
    lines = [
        "# BAF Threshold-Transfer Gate Report",
        "",
        "Decision: **%s**" % summary["decision"],
        "",
        "This is a preregistered BAF v2 policy audit; it is not an exact reproduction of the BAF v1 paper.",
        "",
        "## Execution",
        "",
        f"- Primary units: {summary['n_primary_units']}",
        f"- Runtime seconds: {run_metadata.get('wall_seconds')}",
        f"- Peak RSS GiB: {run_metadata.get('peak_rss_gib')}",
        f"- Python: {run_metadata.get('python')}",
        f"- Data root: `{run_metadata.get('data_root')}`",
        "",
        "## Pre-registered primary result",
        "",
        f"- Median absolute PE gap: `{summary['primary_median_abs_pe_gap']:.6f}`",
        f"- Mean absolute PE gap: `{summary['primary_mean_abs_pe_gap']:.6f}`",
        f"- Stratified bootstrap 95% CI: `[{summary['bootstrap_ci_95'][0]:.6f}, {summary['bootstrap_ci_95'][1]:.6f}]`",
        f"- H1 qualifying variants: `{', '.join(summary['h1_qualifying_variants']) or 'none'}`",
        f"- H1 effect path: `{summary['h1_pass']}`; ranking path: `{summary['h1_ranking_path_pass']}`",
        "",
        "## Variant summary",
        "",
        "| Variant | n | Median abs PE gap | Mean abs PE gap | Median signed gap | Direction |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for item in summary["variant_summary"]:
        lines.append(f"| {item['variant']} | {item['n']} | {item['median_abs_pe_gap']:.6f} | {item['mean_abs_pe_gap']:.6f} | {item['median_signed_pe_gap']:.6f} | {item['direction']} |")
    lines += [
        "",
        "## Capacity-ranking turnover",
        "",
        f"`{json.dumps(summary['ranking_turnover'], ensure_ascii=False)}`",
        "",
        "## H2 descriptive summary",
        "",
        "| Policy | n | Median PE error vs oracle | Median older log-loss delta | Median younger log-loss delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in summary["h2"]:
        lines.append(f"| {item['policy']} | {item['n']} | {item['median_pe_error_vs_oracle']:.6f} | {item['median_older_logloss_delta_vs_oracle']:.6f} | {item['median_younger_logloss_delta_vs_oracle']:.6f} |")
    lines += [
        "",
        "## Decision rule",
        "",
        "GO requires the pre-registered H1 path or capacity-ranking turnover ≥20%. BORDERLINE permits only the frozen five-seed repair. NO-GO stops this direction; no post-hoc endpoint, data, or threshold change is allowed.",
        "",
        "## Evidence boundary",
        "",
        "The test-oracle policy uses future labels by design and is a reference, not a deployable threshold. All fixed and lag-1 policy thresholds are selected from historically available labels. The public Base semantic mismatch and unreconstructible CTGAN source recipe remain separate limitations.",
        "",
    ]
    return "\n".join(lines)
