"""
Step 1: Build stratified calibration set for human annotation.

Sources:
  - Primary: stress_test_100turn_deepseek aggregate (52x100, deepseek-chat)
  - Supplement: stress_test_150turn_deepseek Stage 1 results (200x150, new models)

Stratification: condition (3) x domain (4) x turn_epoch (3) = 36 cells
Target: ~50 rows total (conditions: default / cynical / distant only)
"""
import argparse
import json
import random
from pathlib import Path
from collections import defaultdict

CONDITIONS = ("default", "cynical", "distant")
DOMAINS = ("creative", "finance", "mental_health", "medicine")
EPOCHS = [("early", 1, 33), ("mid", 34, 67), ("late", 68, 100)]


def epoch_of(turn_index: int) -> str:
    for name, lo, hi in EPOCHS:
        if lo <= turn_index <= hi:
            return name
    return "late"


def load_valid(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if (
                not r.get("error_stage")
                and r.get("model_reply")
                and r.get("condition") in CONDITIONS
                and r.get("domain") in DOMAINS
            ):
                rows.append(r)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--primary", default=
        "stress_test_100turn_deepseek/output_aggregate_20260322/"
        "results_aggregate_success_20260322.jsonl")
    p.add_argument("--supplement", nargs="*", default=[
        "stress_test_150turn_deepseek/results_generate_001.jsonl",
        "stress_test_150turn_deepseek/results_test_gemini.jsonl",
        "stress_test_150turn_deepseek/results_test_gpt4omini.jsonl",
    ])
    p.add_argument("--per_cell", type=int, default=1)
    p.add_argument("--supplement_per_model", type=int, default=3)
    p.add_argument("--output", default="few_shot_design/calibration_set.jsonl")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)

    # Load primary data
    print(f"Loading primary: {args.primary}")
    primary_rows = load_valid(args.primary)
    print(f"  Valid rows: {len(primary_rows)}")

    # Stratified sample from primary
    cells: dict[tuple, list] = defaultdict(list)
    for r in primary_rows:
        key = (r["condition"], r["domain"], epoch_of(r["turn_index"]))
        cells[key].append(r)

    selected = []
    for key in sorted(cells.keys()):
        pool = cells[key]
        rng.shuffle(pool)
        for r in pool[:args.per_cell]:
            selected.append(r)

    print(f"  Stratified sample: {len(selected)} rows ({len(cells)} strata)")

    # Supplement from new models
    for sup_path in args.supplement:
        path = Path(sup_path)
        if not path.exists():
            print(f"  Skipping (not found): sup_path")
            continue
        sup_rows = load_valid(str(path))
        if not sup_rows:
            continue
        # Sample per condition
        by_cond = defaultdict(list)
        for r in sup_rows:
            by_cond[r["condition"]].append(r)
        for cond, pool in by_cond.items():
            rng.shuffle(pool)
            selected.extend(pool[:args.supplement_per_model])
        print(f"  Supplement {path.name}: +{args.supplement_per_model * len(by_cond)} rows")

    # Deduplicate by (dialogue_id, condition, turn_index)
    seen = set()
    deduped = []
    for r in selected:
        key = (r.get("dialogue_id"), r.get("condition"), r.get("turn_index"))
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    rng.shuffle(deduped)

    # Build annotation rows (blank human score fields)
    out_rows = []
    for i, r in enumerate(deduped):
        out_rows.append({
            "cal_id": f"cal_{i+1:03d}",
            "condition": r["condition"],
            "domain": r["domain"],
            "turn_index": r["turn_index"],
            "epoch": epoch_of(r["turn_index"]),
            "dialogue_id": r.get("dialogue_id", ""),
            "llm3_model": r.get("llm3_model", r.get("model", "unknown")),
            "user_text": r["user_text"],
            "model_reply": r["model_reply"],
            # Existing LLM4 scores (for reference after annotation)
            "llm4_harm": r.get("harm_1_10"),
            "llm4_neg_emo": r.get("negative_emotion_1_10"),
            "llm4_inapp": r.get("inappropriate_1_10"),
            "llm4_empathic": r.get("empathic_language_1_10"),
            "llm4_anthro_q1": r.get("anthro_q1"),
            "llm4_anthro_q2": r.get("anthro_q2"),
            "llm4_anthro_q3": r.get("anthro_q3"),
            "llm4_anthro_q4": r.get("anthro_q4"),
            "llm4_anthro_q5": r.get("anthro_q5"),
            # Human annotation fields (to be filled)
            "human_harm_1_10": None,
            "human_neg_emo_1_10": None,
            "human_inapp_1_10": None,
            "human_empathic_1_10": None,
            "human_anthro_1_5": None,
            "annotator_id": None,
            "notes": None,
        })

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nCalibration set saved: {len(out_rows)} rows → {out_path}")

    # Coverage summary
    print("\nCoverage:")
    cov = defaultdict(int)
    for r in out_rows:
        cov[(r["condition"], r["domain"])] += 1
    for k, n in sorted(cov.items()):
        print(f"  {k[0]:10s} | {k[1]:15s} → {n} rows")


if __name__ == "__main__":
    main()
