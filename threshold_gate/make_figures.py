from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from threshold_gate.analysis import capacity_ranking_units, primary_effects


OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
VARIANT_ORDER = ["Base", "Variant I", "Variant II", "Variant III", "Variant IV", "Variant V"]
POLICY_ORDER = ["test_oracle", "fixed_m5", "lag1"]
POLICY_LABELS = {"test_oracle": "Test oracle", "fixed_m5": "Fixed month 5", "lag1": "One-period lag"}


def style() -> None:
    sns.set_theme(style="ticks", context="paper", font_scale=1.0)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def load_rows(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frame = pd.DataFrame(payload["rows"])
    frame = frame.loc[frame["scenario"].eq("clean") | frame["scenario"].str.startswith("noise_")].copy()
    return frame


def figure1_effect_and_turnover(rows: pd.DataFrame, output_dir: Path) -> None:
    main = rows.loc[
        rows["scenario"].eq("clean")
        & rows["period"].eq("pooled_test")
        & rows["target_fpr"].eq(0.05)
    ]
    effects = primary_effects(main)
    units = capacity_ranking_units(main)
    effect_long = effects.assign(variant=pd.Categorical(effects["variant"], VARIANT_ORDER, ordered=True)).sort_values("variant")
    turnover = units.groupby("variant", sort=False)["changed"].agg(["mean", "count"]).reindex(VARIANT_ORDER).reset_index()
    low, high = wilson_interval(turnover["mean"].to_numpy() * turnover["count"].to_numpy(), turnover["count"].to_numpy())

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"wspace": 0.34})
    palette = dict(zip(VARIANT_ORDER, OKABE_ITO))
    sns.boxplot(data=effect_long, x="variant", y="abs_pe_gap", hue="variant", order=VARIANT_ORDER, hue_order=VARIANT_ORDER, palette=palette, width=0.62, fliersize=0, legend=False, ax=axes[0])
    sns.stripplot(data=effect_long, x="variant", y="abs_pe_gap", order=VARIANT_ORDER, color="#222222", size=2.4, jitter=0.17, alpha=0.65, ax=axes[0])
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Absolute PE-ratio gap")
    axes[0].set_xticks(np.arange(len(VARIANT_ORDER)), ["Base", "I", "II", "III", "IV", "V"])
    axes[0].set_title("Policy gap by BAF variant", loc="left")

    x = np.arange(len(turnover))
    axes[1].errorbar(x, turnover["mean"], yerr=[turnover["mean"].to_numpy() - low, high - turnover["mean"].to_numpy()], fmt="o", color=OKABE_ITO[0], capsize=3, lw=1.2)
    axes[1].axhline(0.2, color="#777777", ls="--", lw=0.9, label="Gate threshold")
    axes[1].set_xticks(x, ["Base", "I", "II", "III", "IV", "V"])
    axes[1].set_ylim(0, 1.0)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Ranking turnover")
    axes[1].set_title("Capacity-ranking changes", loc="left")
    axes[1].legend(frameon=False, loc="upper right")
    for ax, label in zip(axes, ("a", "b")):
        ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontweight="bold", fontsize=10, va="top")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Threshold-transfer sensitivity at target FPR 0.05", y=1.02, fontsize=10)
    save(fig, output_dir, "figure1_policy_gap_and_turnover")


def wilson_interval(successes: np.ndarray, totals: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    p = successes / totals
    denominator = 1 + z**2 / totals
    center = (p + z**2 / (2 * totals)) / denominator
    half = z * np.sqrt((p * (1 - p) + z**2 / (4 * totals)) / totals) / denominator
    return np.clip(center - half, 0, 1), np.clip(center + half, 0, 1)


def figure2_target_sensitivity(rows: pd.DataFrame, output_dir: Path) -> None:
    clean = rows.loc[rows["scenario"].eq("clean") & rows["period"].eq("pooled_test")]
    effect_records = []
    turnover_records = []
    for target, group in clean.groupby("target_fpr", sort=True):
        effects = primary_effects(group)
        effects["target_fpr"] = float(target)
        effect_records.append(effects)
        units = capacity_ranking_units(group)
        turnover_records.append({"target_fpr": float(target), "changed": int(units["changed"].sum()), "units": len(units)})
    effects = pd.concat(effect_records, ignore_index=True)
    turnover = pd.DataFrame(turnover_records)
    medians = effects.groupby("target_fpr")["abs_pe_gap"].median().reset_index(name="median")
    lower = effects.groupby("target_fpr")["abs_pe_gap"].quantile(0.25).reset_index(name="lower")
    upper = effects.groupby("target_fpr")["abs_pe_gap"].quantile(0.75).reset_index(name="upper")
    medians = medians.merge(lower).merge(upper)
    low, high = wilson_interval(turnover["changed"].to_numpy(float), turnover["units"].to_numpy(float))

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0), gridspec_kw={"wspace": 0.36})
    x = medians["target_fpr"].to_numpy()
    axes[0].errorbar(x, medians["median"], yerr=[medians["median"] - medians["lower"], medians["upper"] - medians["median"]], fmt="o-", color=OKABE_ITO[1], capsize=3, lw=1.3)
    axes[0].set_xlabel("Target FPR")
    axes[0].set_ylabel("Median absolute PE-ratio gap")
    axes[0].set_xticks(x, [f"{v:.2f}" for v in x])
    axes[0].set_title("PE gap sensitivity", loc="left")

    x2 = turnover["target_fpr"].to_numpy()
    p = turnover["changed"].to_numpy() / turnover["units"].to_numpy()
    axes[1].errorbar(x2, p, yerr=[p - low, high - p], fmt="o-", color=OKABE_ITO[2], capsize=3, lw=1.3)
    axes[1].axhline(0.2, color="#777777", ls="--", lw=0.9)
    axes[1].set_ylim(0, 1.0)
    axes[1].set_xlabel("Target FPR")
    axes[1].set_ylabel("Ranking turnover")
    axes[1].set_xticks(x2, [f"{v:.2f}" for v in x2])
    axes[1].set_title("Capacity-order sensitivity", loc="left")
    for ax, label in zip(axes, ("a", "b")):
        ax.text(-0.13, 1.08, label, transform=ax.transAxes, fontweight="bold", fontsize=10, va="top")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Sensitivity to the predeclared operating-point target", y=1.02, fontsize=10)
    save(fig, output_dir, "figure2_target_fpr_sensitivity")


def figure3_noise_pressure(rows: pd.DataFrame, output_dir: Path) -> None:
    selected = rows.loc[
        rows["model_config"].eq("reference")
        & rows["period"].eq("pooled_test")
        & rows["target_fpr"].eq(0.05)
        & rows["scenario"].isin(("clean", "noise_0.25", "noise_0.50", "noise_1.00"))
    ].copy()
    order = ["clean", "noise_0.25", "noise_0.50", "noise_1.00"]
    labels = ["Clean", "0.25", "0.50", "1.00"]
    selected["severity"] = pd.Categorical(selected["scenario"], order, ordered=True)
    metrics = [("recall", "Recall"), ("global_fpr", "Global FPR"), ("pe_ratio", "PE ratio"), ("alert_rate", "Alert rate")]
    colors = dict(zip(POLICY_ORDER, OKABE_ITO[:3]))
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.4), sharex=True, gridspec_kw={"wspace": 0.28, "hspace": 0.38})
    for ax, (metric, label), panel in zip(axes.ravel(), metrics, ("a", "b", "c", "d")):
        for policy in POLICY_ORDER:
            group = selected.loc[selected["policy"].eq(policy)].groupby("severity", observed=False)[metric]
            summary = group.agg(["median", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]).reindex(order)
            med = summary["median"].to_numpy(float)
            q1 = summary["<lambda_0>"].to_numpy(float)
            q3 = summary["<lambda_1>"].to_numpy(float)
            x = np.arange(len(order)) + (POLICY_ORDER.index(policy) - 1) * 0.08
            ax.errorbar(x, med, yerr=[med - q1, q3 - med], fmt="o-", color=colors[policy], capsize=2.5, lw=1.1, ms=3.5, label=POLICY_LABELS[policy])
        ax.set_title(label, loc="left")
        ax.set_xticks(np.arange(len(order)), labels)
        ax.set_xlabel("Noise SD / fit-scale SD")
        ax.set_ylabel(label)
        ax.text(-0.12, 1.08, panel, transform=ax.transAxes, fontweight="bold", fontsize=10, va="top")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0, 1].legend(frameon=False, loc="best")
    fig.suptitle("Reference-model sensitivity to test-only feature noise", y=0.995, fontsize=10)
    save(fig, output_dir, "figure3_noise_pressure_test")


def main() -> None:
    input_path = ROOT / "results" / "full_threshold_policy_20260915" / "full_raw_results.json"
    output_dir = ROOT / "results" / "full_threshold_policy_20260915" / "figures"
    style()
    rows = load_rows(input_path)
    figure1_effect_and_turnover(rows, output_dir)
    figure2_target_sensitivity(rows, output_dir)
    figure3_noise_pressure(rows, output_dir)
    print(output_dir)


if __name__ == "__main__":
    main()
