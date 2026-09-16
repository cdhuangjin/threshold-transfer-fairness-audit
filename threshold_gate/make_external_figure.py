from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PAIR_ORDER = ["CA->PR", "CA->HI", "MS->HI", "MA->PR"]
COLORS = {"CA->PR": "#1b9e77", "CA->HI": "#d95f02", "MS->HI": "#7570b3", "MA->PR": "#e7298a"}


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials <= 0:
        return (float("nan"), float("nan"))
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    half = z * np.sqrt((p * (1 - p) + z * z / (4 * trials)) / trials) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def primary_effects(rows: pd.DataFrame) -> pd.DataFrame:
    keys = ["pair", "model_config", "seed"]
    selected = rows.loc[rows["target_fpr"].eq(0.05) & rows["policy"].isin(("target_oracle", "source_calibrated")), keys + ["policy", "pe_ratio"]]
    wide = selected.pivot_table(index=keys, columns="policy", values="pe_ratio", aggfunc="first").dropna().reset_index()
    wide["abs_pe_gap"] = (wide["target_oracle"] - wide["source_calibrated"]).abs()
    return wide


def ranking_units(rows: pd.DataFrame) -> pd.DataFrame:
    selected = rows.loc[rows["target_fpr"].eq(0.05) & rows["policy"].isin(("target_oracle", "source_calibrated")), ["pair", "seed", "model_config", "policy", "recall"]]
    records = []
    for (pair, seed), group in selected.groupby(["pair", "seed"], sort=True):
        pivot = group.pivot(index="model_config", columns="policy", values="recall").dropna()
        oracle_order = tuple(pivot.sort_values("target_oracle", ascending=False, kind="stable").index)
        source_order = tuple(pivot.sort_values("source_calibrated", ascending=False, kind="stable").index)
        records.append({"pair": pair, "seed": int(seed), "changed": int(oracle_order != source_order)})
    return pd.DataFrame(records)


def make_figure(raw_path: Path, output_dir: Path) -> None:
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    rows = pd.DataFrame(payload["rows"])
    effects = primary_effects(rows)
    units = ranking_units(rows)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), gridspec_kw={"width_ratios": [1.25, 1.0]})

    ax = axes[0]
    values = [effects.loc[effects["pair"].eq(pair), "abs_pe_gap"].to_numpy() for pair in PAIR_ORDER]
    box = ax.boxplot(values, positions=np.arange(len(PAIR_ORDER)), widths=0.5, patch_artist=True, showfliers=False, medianprops={"color": "#222222", "linewidth": 1.4})
    for patch, pair in zip(box["boxes"], PAIR_ORDER):
        patch.set_facecolor(COLORS[pair])
        patch.set_alpha(0.55)
    rng = np.random.default_rng(20260916)
    for index, (pair, vals) in enumerate(zip(PAIR_ORDER, values)):
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(np.full(len(vals), index) + jitter, vals, s=14, color=COLORS[pair], alpha=0.65, edgecolors="white", linewidths=0.25, zorder=3)
    ax.set_xticks(np.arange(len(PAIR_ORDER)), PAIR_ORDER, rotation=25, ha="right")
    ax.set_ylabel("Absolute PE-ratio gap")
    ax.set_title("(a) External PE-gap sensitivity")
    ax.grid(axis="y", alpha=0.22)

    ax = axes[1]
    proportions = []
    lower = []
    upper = []
    for pair in PAIR_ORDER:
        part = units.loc[units["pair"].eq(pair), "changed"]
        successes, trials = int(part.sum()), int(len(part))
        lo, hi = wilson_interval(successes, trials)
        proportions.append(successes / trials)
        lower.append(proportions[-1] - lo)
        upper.append(hi - proportions[-1])
    x = np.arange(len(PAIR_ORDER))
    ax.bar(x, proportions, yerr=np.vstack([lower, upper]), color=[COLORS[pair] for pair in PAIR_ORDER], alpha=0.72, capsize=3, edgecolor="#333333", linewidth=0.5)
    ax.axhline(0.5, color="#777777", linestyle="--", linewidth=0.9, label="50% reference")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x, PAIR_ORDER, rotation=25, ha="right")
    ax.set_ylabel("Ranking turnover")
    ax.set_title("(b) Recall-ranking turnover")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, loc="upper left")
    fig.suptitle("WhyShift external replication at target FPR = 0.05", y=1.01, fontsize=11)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "figure4_external_replication.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "figure4_external_replication.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    make_figure(args.raw, args.output_dir)


if __name__ == "__main__":
    main()
