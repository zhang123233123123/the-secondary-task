#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean

Q_FIELDS = ["anthro_q1", "anthro_q2", "anthro_q3", "anthro_q4", "anthro_q5"]
AI_FIELDS = [
    "harm_1_10",
    "negative_emotion_1_10",
    "inappropriate_1_10",
    "empathic_language_1_10",
    *Q_FIELDS,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build balanced expert review samples.")
    parser.add_argument("--results", required=True, help="Aggregate success JSONL.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--seed", type=int, default=20260325, help="Sampling seed.")
    parser.add_argument(
        "--scheme",
        choices=["balanced_48", "assigned_12x20", "assigned_12x20_no_unhelpful"],
        default="balanced_48",
        help="Sampling scheme to build.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        turn_index = int(row["turn_index"])
        if turn_index <= 33:
            phase = "early_1_33"
        elif turn_index <= 66:
            phase = "mid_34_66"
        else:
            phase = "late_67_100"
        rows.append(
            {
                "domain": row["domain"],
                "condition": row["condition"],
                "phase": phase,
                "dialogue_id": row["dialogue_id"],
                "turn_index": turn_index,
                "user_text": row["user_text"],
                "model_reply": row["model_reply"],
                "ai_scores": {field: row[field] for field in AI_FIELDS},
                "ai_anthro_mean": mean(row[field] for field in Q_FIELDS),
            }
        )
    return rows


def build_samples(rows: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    domains = ["creative", "finance", "mental_health", "medicine"]
    conditions = ["default", "unhelpful", "cynical", "distant"]
    phases = ["early_1_33", "mid_34_66", "late_67_100"]
    samples: list[dict] = []
    sample_num = 1
    for domain in domains:
        for condition in conditions:
            for phase in phases:
                bucket = [
                    row
                    for row in rows
                    if row["domain"] == domain
                    and row["condition"] == condition
                    and row["phase"] == phase
                ]
                if not bucket:
                    continue
                chosen = dict(rng.choice(bucket))
                chosen["sample_id"] = f"S{sample_num:03d}"
                samples.append(chosen)
                sample_num += 1
    return samples


def build_assigned_samples(rows: list[dict], seed: int) -> tuple[list[dict], dict[str, list[str]], dict[str, dict[str, str]]]:
    rng = random.Random(seed)
    domains = ["creative", "finance", "mental_health", "medicine"]
    conditions = ["default", "unhelpful", "cynical", "distant"]
    phases = ["early_1_33", "mid_34_66", "late_67_100"]

    reviewers_by_domain = {
        "creative": ["expert01", "expert02", "expert03"],
        "finance": ["expert04", "expert05", "expert06"],
        "mental_health": ["expert07", "expert08", "expert09"],
        "medicine": ["expert10", "expert11", "expert12"],
    }
    split_patterns = {
        "early_1_33": [2, 2, 1],
        "mid_34_66": [2, 1, 2],
        "late_67_100": [1, 2, 2],
    }

    assignments: dict[str, list[str]] = {
        reviewer: []
        for reviewers in reviewers_by_domain.values()
        for reviewer in reviewers
    }
    reviewer_meta = {
        reviewer: {"focus_domain": domain}
        for domain, reviewers in reviewers_by_domain.items()
        for reviewer in reviewers
    }

    samples: list[dict] = []
    sample_num = 1
    for domain in domains:
        domain_reviewers = reviewers_by_domain[domain]
        for condition in conditions:
            for phase in phases:
                bucket = [
                    row
                    for row in rows
                    if row["domain"] == domain
                    and row["condition"] == condition
                    and row["phase"] == phase
                ]
                selected = rng.sample(bucket, 5)
                idx = 0
                for reviewer, count in zip(domain_reviewers, split_patterns[phase]):
                    for _ in range(count):
                        chosen = dict(selected[idx])
                        chosen["sample_id"] = f"S{sample_num:03d}"
                        chosen["assigned_reviewer"] = reviewer
                        samples.append(chosen)
                        assignments[reviewer].append(chosen["sample_id"])
                        sample_num += 1
                        idx += 1
    return samples, assignments, reviewer_meta


def build_assigned_samples_no_unhelpful(
    rows: list[dict], seed: int
) -> tuple[list[dict], dict[str, list[str]], dict[str, dict[str, str]]]:
    rng = random.Random(seed)
    domains = ["creative", "finance", "mental_health", "medicine"]
    conditions = ["default", "cynical", "distant"]
    phases = ["early_1_33", "mid_34_66", "late_67_100"]

    reviewers_by_domain = {
        "creative": ["expert01", "expert02", "expert03"],
        "finance": ["expert04", "expert05", "expert06"],
        "mental_health": ["expert07", "expert08", "expert09"],
        "medicine": ["expert10", "expert11", "expert12"],
    }
    phase_patterns_by_condition = {
        "default": {
            "early_1_33": [3, 2, 2],
            "mid_34_66": [2, 2, 2],
            "late_67_100": [2, 3, 2],
        },
        "cynical": {
            "early_1_33": [2, 3, 2],
            "mid_34_66": [2, 2, 2],
            "late_67_100": [2, 2, 3],
        },
        "distant": {
            "early_1_33": [2, 2, 3],
            "mid_34_66": [2, 2, 2],
            "late_67_100": [3, 2, 2],
        },
    }

    assignments: dict[str, list[str]] = {
        reviewer: []
        for reviewers in reviewers_by_domain.values()
        for reviewer in reviewers
    }
    reviewer_meta = {
        reviewer: {"focus_domain": domain}
        for domain, reviewers in reviewers_by_domain.items()
        for reviewer in reviewers
    }

    samples: list[dict] = []
    sample_num = 1
    for domain in domains:
        domain_reviewers = reviewers_by_domain[domain]
        for condition in conditions:
            for phase in phases:
                bucket = [
                    row
                    for row in rows
                    if row["domain"] == domain
                    and row["condition"] == condition
                    and row["phase"] == phase
                ]
                counts = phase_patterns_by_condition[condition][phase]
                selected = rng.sample(bucket, sum(counts))
                idx = 0
                for reviewer, count in zip(domain_reviewers, counts):
                    for _ in range(count):
                        chosen = dict(selected[idx])
                        chosen["sample_id"] = f"S{sample_num:03d}"
                        chosen["assigned_reviewer"] = reviewer
                        samples.append(chosen)
                        assignments[reviewer].append(chosen["sample_id"])
                        sample_num += 1
                        idx += 1
    return samples, assignments, reviewer_meta


def main() -> int:
    args = parse_args()
    results_path = Path(args.results)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_rows(results_path)
    assignments: dict[str, list[str]] = {}
    reviewer_meta: dict[str, dict[str, str]] = {}
    if args.scheme == "balanced_48":
        samples = build_samples(rows, seed=args.seed)
    elif args.scheme == "assigned_12x20":
        samples, assignments, reviewer_meta = build_assigned_samples(rows, seed=args.seed)
    else:
        samples, assignments, reviewer_meta = build_assigned_samples_no_unhelpful(rows, seed=args.seed)
    payload = {
        "source_results": str(results_path),
        "seed": args.seed,
        "scheme": args.scheme,
        "sample_count": len(samples),
        "samples": samples,
        "assignments": assignments,
        "reviewer_meta": reviewer_meta,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
