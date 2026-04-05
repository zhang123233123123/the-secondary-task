#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from build_review_samples import load_rows


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "data" / "review_samples_assigned_12x20.json"
REVIEWS_DIR = ROOT / "data" / "reviews"
ARCHIVE_DIR = ROOT / "data" / "archive"

DOMAINS = ["creative", "finance", "mental_health", "medicine"]
CONDITIONS = ["default", "cynical", "distant"]
PHASES = ["early_1_33", "mid_34_66", "late_67_100"]
PHASE_TARGETS = {"early_1_33": 7, "mid_34_66": 6, "late_67_100": 7}
REVIEWERS_BY_DOMAIN = {
    "creative": ["expert01", "expert02", "expert03"],
    "finance": ["expert04", "expert05", "expert06"],
    "mental_health": ["expert07", "expert08", "expert09"],
    "medicine": ["expert10", "expert11", "expert12"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove unhelpful samples and preserve existing non-unhelpful reviews.")
    parser.add_argument(
        "--results",
        default="stress_test_100turn_deepseek/output_aggregate/results_aggregate_success.jsonl",
        help="Source aggregate success JSONL.",
    )
    parser.add_argument("--seed", type=int, default=20260327, help="Sampling seed.")
    parser.add_argument(
        "--source-archive",
        default="",
        help="Optional previous archive folder containing the old dataset and reviews to preserve from.",
    )
    return parser.parse_args()


def timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def archive_current_state(tag: str) -> Path:
    archive_root = ARCHIVE_DIR / f"remove_unhelpful_{tag}"
    archive_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DATASET_PATH, archive_root / DATASET_PATH.name)
    reviews_archive = archive_root / "reviews"
    reviews_archive.mkdir(parents=True, exist_ok=True)
    for path in REVIEWS_DIR.glob("*.json"):
        shutil.copy2(path, reviews_archive / path.name)
    return archive_root


def load_source_state(source_archive: str) -> tuple[dict, Path]:
    if source_archive:
        archive_root = Path(source_archive)
        dataset_path = archive_root / DATASET_PATH.name
        reviews_dir = archive_root / "reviews"
    else:
        dataset_path = DATASET_PATH
        reviews_dir = REVIEWS_DIR
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    return payload, reviews_dir


def collect_preserved_reviews(old_payload: dict, reviews_dir: Path) -> tuple[list[dict], dict[str, dict]]:
    old_sample_map = {sample["sample_id"]: sample for sample in old_payload["samples"]}
    preserved_samples: list[dict] = []
    preserved_reviews: dict[str, dict] = {}

    for review_path in reviews_dir.glob("*.json"):
        if review_path.name == ".gitkeep":
            continue
        reviewer = review_path.stem
        if reviewer not in {r for vals in REVIEWERS_BY_DOMAIN.values() for r in vals}:
            continue
        data = json.loads(review_path.read_text(encoding="utf-8"))
        for sample_id, review in data.get("reviews", {}).items():
            if not review.get("blind_submitted"):
                continue
            sample = old_sample_map.get(sample_id)
            if not sample or sample["condition"] == "unhelpful":
                continue
            preserved = dict(sample)
            preserved["assigned_reviewer"] = reviewer
            preserved_samples.append(preserved)
            preserved_reviews[sample_id] = {
                "blind_submitted": True,
                "blind_submitted_at": review.get("blind_submitted_at"),
                "blind_scores": review.get("blind_scores"),
                "blind_notes": review.get("blind_notes", ""),
                "final_submitted": True,
                "final_submitted_at": review.get("blind_submitted_at"),
            }
    return preserved_samples, preserved_reviews


def build_new_dataset(rows: list[dict], preserved_samples: list[dict], seed: int) -> dict:
    rng = random.Random(seed)
    row_buckets: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row["condition"] not in CONDITIONS:
            continue
        row_buckets[(row["domain"], row["condition"], row["phase"])].append(dict(row))

    used_keys = {
        (sample["dialogue_id"], int(sample["turn_index"]), sample["condition"])
        for sample in preserved_samples
    }
    reviewer_totals = Counter(sample["assigned_reviewer"] for sample in preserved_samples)
    domain_phase_counts = Counter((sample["domain"], sample["condition"], sample["phase"]) for sample in preserved_samples)

    samples = [dict(sample) for sample in preserved_samples]
    reviewer_meta = {
        reviewer: {"focus_domain": domain}
        for domain, reviewers in REVIEWERS_BY_DOMAIN.items()
        for reviewer in reviewers
    }

    existing_ids = {sample["sample_id"] for sample in samples}
    next_id = 1

    def next_sample_id() -> str:
        nonlocal next_id
        while True:
            candidate = f"S{next_id:03d}"
            next_id += 1
            if candidate not in existing_ids:
                existing_ids.add(candidate)
                return candidate

    for domain in DOMAINS:
        reviewers = REVIEWERS_BY_DOMAIN[domain]
        for condition in CONDITIONS:
            for phase in PHASES:
                target = PHASE_TARGETS[phase]
                current = domain_phase_counts[(domain, condition, phase)]
                needed = target - current
                if needed < 0:
                    raise ValueError(f"Preserved reviews exceed target for {domain}/{condition}/{phase}")
                bucket = [
                    row
                    for row in row_buckets[(domain, condition, phase)]
                    if (row["dialogue_id"], int(row["turn_index"]), row["condition"]) not in used_keys
                ]
                if len(bucket) < needed:
                    raise ValueError(f"Not enough samples to fill {domain}/{condition}/{phase}")
                selected = rng.sample(bucket, needed)
                for row in selected:
                    reviewer = min(
                        [r for r in reviewers if reviewer_totals[r] < 20],
                        key=lambda r: (reviewer_totals[r], r),
                    )
                    sample = dict(row)
                    sample["sample_id"] = next_sample_id()
                    sample["assigned_reviewer"] = reviewer
                    samples.append(sample)
                    reviewer_totals[reviewer] += 1
                    domain_phase_counts[(domain, condition, phase)] += 1
                    used_keys.add((sample["dialogue_id"], int(sample["turn_index"]), sample["condition"]))

    assignments = {reviewer: [] for reviewers in REVIEWERS_BY_DOMAIN.values() for reviewer in reviewers}
    for sample in sorted(samples, key=lambda item: item["sample_id"]):
        assignments[sample["assigned_reviewer"]].append(sample["sample_id"])

    for reviewer, sample_ids in assignments.items():
        if len(sample_ids) != 20:
            raise ValueError(f"{reviewer} does not have 20 assigned samples")

    return {
        "source_results": "",
        "seed": seed,
        "scheme": "assigned_12x20_no_unhelpful",
        "sample_count": len(samples),
        "samples": sorted(samples, key=lambda item: item["sample_id"]),
        "assignments": assignments,
        "reviewer_meta": reviewer_meta,
    }


def rewrite_review_files(assignments: dict[str, list[str]], preserved_reviews: dict[str, dict]) -> dict[str, int]:
    migration_counts: dict[str, int] = {}
    active_reviewers = set(assignments)
    for reviewer in active_reviewers:
        review_state = {
            "reviewer": reviewer,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "reviews": {},
        }
        for sample_id in assignments[reviewer]:
            if sample_id in preserved_reviews:
                review_state["reviews"][sample_id] = preserved_reviews[sample_id]
        migration_counts[reviewer] = len(review_state["reviews"])
        review_path = REVIEWS_DIR / f"{reviewer}.json"
        review_path.write_text(json.dumps(review_state, ensure_ascii=False, indent=2), encoding="utf-8")
    return migration_counts


def main() -> int:
    args = parse_args()
    archive_root = archive_current_state(timestamp())
    old_payload, source_reviews_dir = load_source_state(args.source_archive)
    preserved_samples, preserved_reviews = collect_preserved_reviews(old_payload, source_reviews_dir)
    rows = load_rows(Path(args.results))
    new_payload = build_new_dataset(rows, preserved_samples, args.seed)
    new_payload["source_results"] = args.results
    DATASET_PATH.write_text(json.dumps(new_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    migrated_counts = rewrite_review_files(new_payload["assignments"], preserved_reviews)

    summary = {
        "archive_root": str(archive_root),
        "source_archive": args.source_archive or "",
        "new_dataset_path": str(DATASET_PATH),
        "new_scheme": new_payload["scheme"],
        "new_sample_count": new_payload["sample_count"],
        "preserved_review_count": len(preserved_reviews),
        "migrated_counts": migrated_counts,
    }
    (archive_root / "migration_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
