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
Q_FIELDS = ["anthro_q1", "anthro_q2", "anthro_q3", "anthro_q4", "anthro_q5"]
SCORE_FIELDS = [
    "harm_1_10",
    "negative_emotion_1_10",
    "inappropriate_1_10",
    "empathic_language_1_10",
]
PROGRESS_BINS = [
    ("0-20%", 0.0, 0.2),
    ("20-40%", 0.2, 0.4),
    ("40-60%", 0.4, 0.6),
    ("60-80%", 0.6, 0.8),
    ("80-100%", 0.8, 1.0000001),
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create deduplicated analysis plots for a run")
    parser.add_argument("--results", required=True, help="Path to results_{run_id}.jsonl")
    parser.add_argument(
        "--output_dir",
        default="output",
        help="Directory for plot outputs",
    )
    return parser.parse_args()


def _load_rows(results_path: Path) -> list[dict]:
    rows: list[dict] = []
    with results_path.open("r", encoding="utf-8") as fh:
        for index, line in enumerate(fh):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            row["_row_index"] = index
            rows.append(row)
    return rows


def _deduplicate_rows(rows: list[dict]) -> list[dict]:
    latest: dict[tuple[str, str, int], dict] = {}
    for row in rows:
        key = (row["dialogue_id"], row["condition"], int(row["turn_index"]))
        latest[key] = row
    return list(latest.values())


def _anthro_mean(row: dict) -> float | None:
    vals = [row.get(field) for field in Q_FIELDS]
    if not all(isinstance(value, int) for value in vals):
        return None
    return float(mean(vals))


def _build_dataframe(rows: list[dict]) -> pd.DataFrame:
    normalized: list[dict] = []
    for row in rows:
        item = dict(row)
        item["anthro_mean"] = _anthro_mean(row)
        normalized.append(item)
    df = pd.DataFrame(normalized)
    for field in SCORE_FIELDS + ["anthro_mean"]:
        df[field] = pd.to_numeric(df[field], errors="coerce")
    df["turn_index"] = pd.to_numeric(df["turn_index"], errors="coerce")
    df["context_truncated"] = df["context_truncated"].fillna(False).astype(bool)
    df["refusal_detected"] = df["refusal_detected"].fillna(False).astype(bool)
    return df


def _dialogue_level_df(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    grouped = df.groupby(["dialogue_id", "condition"], sort=True)
    for (dialogue_id, condition), group in grouped:
        group = group.sort_values("turn_index")
        record = {
            "dialogue_id": dialogue_id,
            "condition": condition,
            "domain": group["domain"].iloc[0],
            "turns": int(len(group)),
            "refusal_rate": float(group["refusal_detected"].mean()),
            "truncation_rate": float(group["context_truncated"].mean()),
        }
        for field in SCORE_FIELDS + ["anthro_mean"]:
            record[field] = float(group[field].mean())
        records.append(record)
    return pd.DataFrame(records)


def _progress_bin_df(df: pd.DataFrame) -> pd.DataFrame:
    annotated: list[dict] = []
    grouped = df.groupby(["dialogue_id", "condition"], sort=True)
    for (_, condition), group in grouped:
        group = group.sort_values("turn_index").reset_index(drop=True)
        total_turns = len(group)
        for position, (_, row) in enumerate(group.iterrows(), start=1):
            relative = (position - 1) / total_turns if total_turns else 0.0
            label = "80-100%"
            for candidate, lower, upper in PROGRESS_BINS:
                if lower <= relative < upper:
                    label = candidate
                    break
            item = row.to_dict()
            item["progress_bin"] = label
            item["relative_progress"] = relative
            item["condition"] = condition
            annotated.append(item)
    progress_df = pd.DataFrame(annotated)
    summary = (
        progress_df.groupby(["condition", "progress_bin"], sort=False)
        .agg(
            harm_1_10=("harm_1_10", "mean"),
            negative_emotion_1_10=("negative_emotion_1_10", "mean"),
            inappropriate_1_10=("inappropriate_1_10", "mean"),
            empathic_language_1_10=("empathic_language_1_10", "mean"),
            anthro_mean=("anthro_mean", "mean"),
            truncation_rate=("context_truncated", "mean"),
        )
        .reset_index()
    )
    summary["progress_bin"] = pd.Categorical(
        summary["progress_bin"],
        categories=[label for label, _, _ in PROGRESS_BINS],
        ordered=True,
    )
    return summary.sort_values(["condition", "progress_bin"])


def _save_condition_bars(dialogue_df: pd.DataFrame, output_dir: Path, stem: str) -> Path:
    metrics = [
        ("harm_1_10", "Harm"),
        ("negative_emotion_1_10", "Negative Emotion"),
        ("inappropriate_1_10", "Inappropriate"),
        ("empathic_language_1_10", "Empathic"),
        ("anthro_mean", "Anthropomorphism"),
    ]
    melted = dialogue_df.melt(
        id_vars=["dialogue_id", "condition"],
        value_vars=[metric for metric, _ in metrics],
        var_name="metric",
        value_name="score",
    )
    label_map = {metric: label for metric, label in metrics}
    melted["metric"] = melted["metric"].map(label_map)

    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    palette = {
        "default": "#2a9d8f",
        "unhelpful": "#e76f51",
        "cynical": "#264653",
        "distant": "#8ab17d",
    }
    for idx, label in enumerate(label_map.values()):
        ax = axes[idx]
        subset = melted[melted["metric"] == label]
        sns.barplot(
            data=subset,
            x="condition",
            y="score",
            hue="condition",
            order=CONDITIONS,
            palette=palette,
            errorbar=None,
            legend=False,
            ax=ax,
        )
        ax.set_title(label)
        ax.set_xlabel("")
        ax.set_ylabel("Dialogue-level mean")
        ax.tick_params(axis="x", rotation=15)
    axes[-1].axis("off")
    fig.suptitle("Deduplicated Dialogue-Level Comparison Across Conditions", y=1.02)
    fig.tight_layout()
    output_path = output_dir / f"{stem}_condition_bars.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _save_domain_heatmap(
    dialogue_df: pd.DataFrame,
    output_dir: Path,
    stem: str,
    *,
    metric: str,
    title: str,
    colorbar_label: str,
    filename_suffix: str,
) -> Path:
    heatmap_df = (
        dialogue_df.groupby(["domain", "condition"])[metric]
        .mean()
        .reset_index()
        .pivot(index="domain", columns="condition", values=metric)
        .reindex(columns=CONDITIONS)
    )
    sns.set_theme(style="white", context="talk")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        linewidths=0.5,
        cbar_kws={"label": colorbar_label},
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    output_path = output_dir / f"{stem}_{filename_suffix}.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _save_domain_heatmap_panel(dialogue_df: pd.DataFrame, output_dir: Path, stem: str) -> Path:
    metrics = [
        ("harm_1_10", "Harm"),
        ("negative_emotion_1_10", "Negative Emotion"),
        ("inappropriate_1_10", "Inappropriate"),
        ("empathic_language_1_10", "Empathic"),
        ("anthro_mean", "Anthropomorphism"),
    ]
    sns.set_theme(style="white", context="talk")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for idx, (metric, title) in enumerate(metrics):
        heatmap_df = (
            dialogue_df.groupby(["domain", "condition"])[metric]
            .mean()
            .reset_index()
            .pivot(index="domain", columns="condition", values=metric)
            .reindex(columns=CONDITIONS)
        )
        sns.heatmap(
            heatmap_df,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            linewidths=0.5,
            cbar=True,
            ax=axes[idx],
        )
        axes[idx].set_title(title)
        axes[idx].set_xlabel("")
        axes[idx].set_ylabel("")

    axes[-1].axis("off")
    fig.suptitle("Domain Sensitivity Heatmaps Across Metrics (Dialogue-Level)", y=1.02)
    fig.tight_layout()
    output_path = output_dir / f"{stem}_domain_heatmap_panel.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _save_progress_lines(progress_df: pd.DataFrame, output_dir: Path, stem: str) -> Path:
    metrics = [
        ("harm_1_10", "Harm"),
        ("negative_emotion_1_10", "Negative Emotion"),
        ("inappropriate_1_10", "Inappropriate"),
        ("empathic_language_1_10", "Empathic"),
        ("anthro_mean", "Anthropomorphism"),
    ]
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    axes = axes.flatten()
    palette = {
        "default": "#2a9d8f",
        "unhelpful": "#e76f51",
        "cynical": "#264653",
        "distant": "#8ab17d",
    }
    for idx, (metric, label) in enumerate(metrics):
        ax = axes[idx]
        sns.lineplot(
            data=progress_df,
            x="progress_bin",
            y=metric,
            hue="condition",
            hue_order=CONDITIONS,
            palette=palette,
            marker="o",
            linewidth=2.5,
            ax=ax,
        )
        ax.set_title(label)
        ax.set_xlabel("Relative dialogue progress")
        ax.set_ylabel("Mean score")
        if idx != 0:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
    axes[-1].axis("off")
    first_legend = axes[0].get_legend()
    if first_legend is not None:
        first_legend.set_title("Condition")
    fig.suptitle("Progress-Normalized Drift on Deduplicated Turns", y=1.02)
    fig.tight_layout()
    output_path = output_dir / f"{stem}_progress_lines.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _save_truncation_plot(progress_df: pd.DataFrame, output_dir: Path, stem: str) -> Path:
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(9, 5))
    palette = {
        "default": "#2a9d8f",
        "unhelpful": "#e76f51",
        "cynical": "#264653",
        "distant": "#8ab17d",
    }
    sns.lineplot(
        data=progress_df,
        x="progress_bin",
        y="truncation_rate",
        hue="condition",
        hue_order=CONDITIONS,
        palette=palette,
        marker="o",
        linewidth=2.5,
        ax=ax,
    )
    ax.set_title("Truncation Rate by Relative Dialogue Progress")
    ax.set_xlabel("Relative dialogue progress")
    ax.set_ylabel("Truncation rate")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Condition")
    fig.tight_layout()
    output_path = output_dir / f"{stem}_truncation_lines.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> int:
    args = _parse_args()
    results_path = Path(args.results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = _load_rows(results_path)
    deduplicated_rows = _deduplicate_rows(raw_rows)
    stem = results_path.stem.replace("results_", "")

    df = _build_dataframe(deduplicated_rows)
    dialogue_df = _dialogue_level_df(df)
    progress_df = _progress_bin_df(df)

    outputs = [
        _save_condition_bars(dialogue_df, output_dir, stem),
        _save_domain_heatmap(
            dialogue_df,
            output_dir,
            stem,
            metric="harm_1_10",
            title="Domain Sensitivity Heatmap (Harm, Dialogue-Level)",
            colorbar_label="Mean harm",
            filename_suffix="domain_heatmap_harm",
        ),
        _save_domain_heatmap(
            dialogue_df,
            output_dir,
            stem,
            metric="negative_emotion_1_10",
            title="Domain Sensitivity Heatmap (Negative Emotion, Dialogue-Level)",
            colorbar_label="Mean negative emotion",
            filename_suffix="domain_heatmap_negative_emotion",
        ),
        _save_domain_heatmap(
            dialogue_df,
            output_dir,
            stem,
            metric="inappropriate_1_10",
            title="Domain Sensitivity Heatmap (Inappropriate, Dialogue-Level)",
            colorbar_label="Mean inappropriate score",
            filename_suffix="domain_heatmap_inappropriate",
        ),
        _save_domain_heatmap(
            dialogue_df,
            output_dir,
            stem,
            metric="empathic_language_1_10",
            title="Domain Sensitivity Heatmap (Empathic, Dialogue-Level)",
            colorbar_label="Mean empathic score",
            filename_suffix="domain_heatmap_empathic",
        ),
        _save_domain_heatmap(
            dialogue_df,
            output_dir,
            stem,
            metric="anthro_mean",
            title="Domain Sensitivity Heatmap (Anthropomorphism, Dialogue-Level)",
            colorbar_label="Mean anthropomorphism",
            filename_suffix="domain_heatmap_anthropomorphism",
        ),
        _save_domain_heatmap_panel(dialogue_df, output_dir, stem),
        _save_progress_lines(progress_df, output_dir, stem),
        _save_truncation_plot(progress_df, output_dir, stem),
    ]

    print("Created plots:")
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
