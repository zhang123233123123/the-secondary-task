#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

CONDITIONS = ["default", "unhelpful", "cynical", "distant"]
PALETTE = {
    "default": "#2a9d8f",
    "unhelpful": "#e76f51",
    "cynical": "#264653",
    "distant": "#8ab17d",
}
Q_FIELDS = ["anthro_q1", "anthro_q2", "anthro_q3", "anthro_q4", "anthro_q5"]
METRICS = [
    ("harm_1_10", "Harm"),
    ("negative_emotion_1_10", "Negative Emotion"),
    ("inappropriate_1_10", "Inappropriate"),
    ("empathic_language_1_10", "Empathic"),
    ("anthro_mean", "Anthropomorphism"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create persona reversion plots for long-turn runs.")
    parser.add_argument("--results", required=True, help="Path to results JSONL file.")
    parser.add_argument("--output_dir", required=True, help="Directory for generated plots.")
    parser.add_argument("--max_turn", type=int, default=200, help="Maximum turn index for binning.")
    parser.add_argument("--bin_size", type=int, default=25, help="Turn bin size.")
    return parser.parse_args()


def build_turn_bins(max_turn: int, bin_size: int) -> list[tuple[int, int]]:
    bins: list[tuple[int, int]] = []
    start = 1
    while start <= max_turn:
        end = min(start + bin_size - 1, max_turn)
        bins.append((start, end))
        start = end + 1
    return bins


def load_rows(results_path: Path) -> list[dict]:
    rows: list[dict] = []
    with results_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def anthro_mean(row: dict) -> float | None:
    values = [row.get(field) for field in Q_FIELDS]
    if not all(isinstance(value, int) for value in values):
        return None
    return float(mean(values))


def build_dataframe(rows: list[dict]) -> pd.DataFrame:
    normalized: list[dict] = []
    for row in rows:
        item = dict(row)
        item["anthro_mean"] = anthro_mean(row)
        item["reply_chars"] = len((row.get("model_reply") or "").strip())
        normalized.append(item)
    df = pd.DataFrame(normalized)
    for field, _ in METRICS:
        df[field] = pd.to_numeric(df[field], errors="coerce")
    df["turn_index"] = pd.to_numeric(df["turn_index"], errors="coerce")
    df["reply_chars"] = pd.to_numeric(df["reply_chars"], errors="coerce")
    return df


def build_bin_means(df: pd.DataFrame, turn_bins: list[tuple[int, int]]) -> pd.DataFrame:
    records: list[dict] = []
    for condition in CONDITIONS:
        subset = df[df["condition"] == condition]
        for start, end in turn_bins:
            seg = subset[(subset["turn_index"] >= start) & (subset["turn_index"] <= end)]
            record = {
                "condition": condition,
                "turn_bin": f"{start}-{end}",
            }
            for field, _ in METRICS:
                record[field] = float(seg[field].mean())
            records.append(record)
    return pd.DataFrame(records)


def build_distance_to_default(df: pd.DataFrame) -> pd.DataFrame:
    core_metrics = [field for field, _ in METRICS]
    default_df = (
        df[df["condition"] == "default"][["turn_index", *core_metrics]]
        .rename(columns={field: f"default_{field}" for field in core_metrics})
        .copy()
    )
    merged = df.merge(default_df, on="turn_index", how="inner")
    records: list[pd.DataFrame] = []
    for condition in ["unhelpful", "cynical", "distant"]:
        subset = merged[merged["condition"] == condition].sort_values("turn_index").copy()
        subset["distance_to_default"] = 0.0
        for field in core_metrics:
            subset["distance_to_default"] += (subset[field] - subset[f"default_{field}"]).abs()
        subset["distance_rolling_10"] = (
            subset["distance_to_default"].rolling(window=10, min_periods=1).mean()
        )
        records.append(subset[["condition", "turn_index", "distance_to_default", "distance_rolling_10"]])
    return pd.concat(records, ignore_index=True)


def build_delta_heatmap(df: pd.DataFrame, turn_bins: list[tuple[int, int]]) -> pd.DataFrame:
    binned = build_bin_means(df, turn_bins)
    default_rows = binned[binned["condition"] == "default"].set_index("turn_bin")
    records: list[dict] = []
    for condition in ["unhelpful", "cynical", "distant"]:
        subset = binned[binned["condition"] == condition].set_index("turn_bin")
        for turn_bin in subset.index:
            delta_harm = subset.loc[turn_bin, "harm_1_10"] - default_rows.loc[turn_bin, "harm_1_10"]
            delta_inapp = subset.loc[turn_bin, "inappropriate_1_10"] - default_rows.loc[turn_bin, "inappropriate_1_10"]
            delta_neg = (
                subset.loc[turn_bin, "negative_emotion_1_10"]
                - default_rows.loc[turn_bin, "negative_emotion_1_10"]
            )
            delta_empathic = (
                subset.loc[turn_bin, "empathic_language_1_10"]
                - default_rows.loc[turn_bin, "empathic_language_1_10"]
            )
            delta_anthro = subset.loc[turn_bin, "anthro_mean"] - default_rows.loc[turn_bin, "anthro_mean"]
            records.append(
                {
                    "condition": condition,
                    "turn_bin": turn_bin,
                    "distance": abs(delta_harm)
                    + abs(delta_inapp)
                    + abs(delta_neg)
                    + abs(delta_empathic)
                    + abs(delta_anthro),
                }
            )
    return pd.DataFrame(records)


def save_progress_panel(bin_df: pd.DataFrame, output_dir: Path, stem: str, turn_bins: list[tuple[int, int]]) -> Path:
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    x_order = [f"{start}-{end}" for start, end in turn_bins]
    for idx, (field, title) in enumerate(METRICS):
        ax = axes[idx]
        sns.lineplot(
            data=bin_df,
            x="turn_bin",
            y=field,
            hue="condition",
            hue_order=CONDITIONS,
            palette=PALETTE,
            marker="o",
            linewidth=2.5,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("Turn bin")
        ax.set_ylabel("Mean score")
        ax.set_xticks(range(len(x_order)))
        ax.set_xticklabels(x_order, rotation=20)
        if idx != 0 and ax.get_legend() is not None:
            ax.get_legend().remove()
    if axes[0].get_legend() is not None:
        axes[0].legend(title="Condition", frameon=True)
    axes[-1].axis("off")
    fig.suptitle("200-Turn Persona Trajectories by Turn Bin", y=1.02)
    fig.tight_layout()
    path = output_dir / f"{stem}_reversion_progress_panel.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def save_distance_plot(distance_df: pd.DataFrame, output_dir: Path, stem: str) -> Path:
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(
        data=distance_df,
        x="turn_index",
        y="distance_rolling_10",
        hue="condition",
        hue_order=["unhelpful", "cynical", "distant"],
        palette=PALETTE,
        linewidth=2.8,
        ax=ax,
    )
    ax.set_title("Distance from Default Across Turns (10-Turn Rolling Mean)")
    ax.set_xlabel("Turn index")
    ax.set_ylabel("Absolute score gap vs default")
    ax.legend(title="Condition", frameon=True)
    fig.tight_layout()
    path = output_dir / f"{stem}_distance_to_default.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def save_distance_heatmap(delta_df: pd.DataFrame, output_dir: Path, stem: str) -> Path:
    pivot = delta_df.pivot(index="condition", columns="turn_bin", values="distance")
    sns.set_theme(style="white", context="talk")
    fig, ax = plt.subplots(figsize=(12, 5.2))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        linewidths=0.5,
        cbar_kws={"label": "Total absolute gap vs default"},
        ax=ax,
    )
    ax.set_title("How Far Each Persona Stays from Default")
    ax.set_xlabel("Turn bin")
    ax.set_ylabel("")
    fig.tight_layout()
    path = output_dir / f"{stem}_distance_heatmap.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def save_reply_length_plot(df: pd.DataFrame, output_dir: Path, stem: str) -> Path:
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_df = df.sort_values(["condition", "turn_index"]).copy()
    plot_df["reply_chars_rolling_10"] = (
        plot_df.groupby("condition")["reply_chars"].transform(lambda s: s.rolling(10, min_periods=1).mean())
    )
    sns.lineplot(
        data=plot_df,
        x="turn_index",
        y="reply_chars_rolling_10",
        hue="condition",
        hue_order=CONDITIONS,
        palette=PALETTE,
        linewidth=2.2,
        ax=ax,
    )
    ax.set_title("Reply Length Across Turns (10-Turn Rolling Mean)")
    ax.set_xlabel("Turn index")
    ax.set_ylabel("Characters")
    ax.legend(title="Condition", frameon=True)
    fig.tight_layout()
    path = output_dir / f"{stem}_reply_length.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def save_condition_small_multiples(bin_df: pd.DataFrame, output_dir: Path, stem: str, turn_bins: list[tuple[int, int]]) -> Path:
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(2, 2, figsize=(18, 11), sharex=True)
    axes = axes.flatten()
    x_order = [f"{start}-{end}" for start, end in turn_bins]
    metric_palette = {
        "Harm": "#d62828",
        "Negative Emotion": "#f77f00",
        "Inappropriate": "#6a4c93",
        "Empathic": "#2a9d8f",
        "Anthropomorphism": "#264653",
    }
    for idx, condition in enumerate(CONDITIONS):
        ax = axes[idx]
        subset = bin_df[bin_df["condition"] == condition].copy()
        melted = subset.melt(
            id_vars=["condition", "turn_bin"],
            value_vars=[field for field, _ in METRICS],
            var_name="metric",
            value_name="score",
        )
        label_map = {field: label for field, label in METRICS}
        melted["metric"] = melted["metric"].map(label_map)
        sns.lineplot(
            data=melted,
            x="turn_bin",
            y="score",
            hue="metric",
            hue_order=[label for _, label in METRICS],
            palette=metric_palette,
            marker="o",
            linewidth=2.3,
            ax=ax,
        )
        ax.set_title(condition.title())
        ax.set_xlabel("Turn bin")
        ax.set_ylabel("Mean score")
        ax.set_xticks(range(len(x_order)))
        ax.set_xticklabels(x_order, rotation=20)
        if idx != 0 and ax.get_legend() is not None:
            ax.get_legend().remove()
    if axes[0].get_legend() is not None:
        axes[0].legend(title="Metric", frameon=True)
    fig.suptitle("Metric Trajectories Within Each Persona Condition", y=1.02)
    fig.tight_layout()
    path = output_dir / f"{stem}_condition_small_multiples.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    results_path = Path(args.results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    turn_bins = build_turn_bins(args.max_turn, args.bin_size)
    stem = results_path.stem.replace("results_", "")
    df = build_dataframe(load_rows(results_path))
    bin_df = build_bin_means(df, turn_bins)
    distance_df = build_distance_to_default(df)
    delta_df = build_delta_heatmap(df, turn_bins)
    save_progress_panel(bin_df, output_dir, stem, turn_bins)
    save_distance_plot(distance_df, output_dir, stem)
    save_distance_heatmap(delta_df, output_dir, stem)
    save_reply_length_plot(df, output_dir, stem)
    save_condition_small_multiples(bin_df, output_dir, stem, turn_bins)


if __name__ == "__main__":
    main()
