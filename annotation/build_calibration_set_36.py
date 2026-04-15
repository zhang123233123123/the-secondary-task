"""
Build 36-item stratified calibration set from primary deepseek-chat data.

Source : stress_test_100turn_deepseek aggregate (20,755 rows, deepseek-chat)
Strata : condition (3) × domain (4) × epoch (3) = 36 cells, 1 row each
Output : annotation/calibration_set_36.jsonl
"""
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict

CONDITIONS = ("default", "cynical", "distant")
DOMAINS    = ("creative", "finance", "mental_health", "medicine")
EPOCHS     = [("early", 1, 33), ("mid", 34, 67), ("late", 68, 100)]


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
                and r.get("harm_1_10") is not None   # must have LLM4 scores
            ):
                rows.append(r)
    return rows


def scale_10_to_7(v):
    if v is None:
        return None
    return round(1 + (v - 1) * 6 / 9)


def scale_5_to_7(v):
    if v is None:
        return None
    return round(1 + (v - 1) * 6 / 4)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=(
        "stress_test_100turn_deepseek/output_aggregate_20260322/"
        "results_aggregate_success_20260322.jsonl"
    ))
    p.add_argument("--output", default="annotation/calibration_set_36.jsonl")
    p.add_argument("--per_cell", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)

    print(f"Loading: {args.source}")
    rows = load_valid(args.source)
    print(f"Valid rows (with LLM4 scores): {len(rows)}")

    # Stratified sample: condition × domain × epoch
    cells: dict[tuple, list] = defaultdict(list)
    for r in rows:
        key = (r["condition"], r["domain"], epoch_of(r["turn_index"]))
        cells[key].append(r)

    selected = []
    missing = []
    for cond in CONDITIONS:
        for domain in DOMAINS:
            for epoch_name, _, _ in EPOCHS:
                key = (cond, domain, epoch_name)
                pool = cells.get(key, [])
                if not pool:
                    missing.append(key)
                    continue
                rng.shuffle(pool)
                selected.extend(pool[:args.per_cell])

    if missing:
        print(f"WARNING: {len(missing)} empty cells: {missing}")

    # Shuffle final order (hide condition from annotators)
    rng.shuffle(selected)

    # Build output rows
    out_rows = []
    for i, r in enumerate(selected, 1):
        out_rows.append({
            "cal_id":       f"cal_{i:03d}",
            "condition":    r["condition"],   # hidden from annotators in app
            "domain":       r["domain"],
            "turn_index":   r["turn_index"],
            "epoch":        epoch_of(r["turn_index"]),
            "dialogue_id":  r.get("dialogue_id", ""),
            "llm3_model":   r.get("llm3_model", r.get("model", "unknown")),
            "user_text":    r["user_text"],
            "model_reply":  r["model_reply"],
            # LLM4 reference scores converted to 1-7 (NOT shown to annotators)
            "_llm4_harm_1_7":      scale_10_to_7(r.get("harm_1_10")),
            "_llm4_neg_emo_1_7":   scale_10_to_7(r.get("negative_emotion_1_10")),
            "_llm4_inapp_1_7":     scale_10_to_7(r.get("inappropriate_1_10")),
            "_llm4_empathic_1_7":  scale_10_to_7(r.get("empathic_language_1_10")),
            "_llm4_anthro_q1_1_7": scale_5_to_7(r.get("anthro_q1")),
            "_llm4_anthro_q2_1_7": scale_5_to_7(r.get("anthro_q2")),
            "_llm4_anthro_q3_1_7": scale_5_to_7(r.get("anthro_q3")),
            "_llm4_anthro_q4_1_7": scale_5_to_7(r.get("anthro_q4")),
            "_llm4_anthro_q5_1_7": scale_5_to_7(r.get("anthro_q5")),
        })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(out_rows)} rows → {out}")
    print("\nCoverage (condition × domain × epoch):")
    cov = defaultdict(int)
    for r in out_rows:
        cov[(r["condition"], r["domain"], r["epoch"])] += 1
    for k, n in sorted(cov.items()):
        print(f"  {k[0]:10} | {k[1]:15} | {k[2]:6} → {n}")


if __name__ == "__main__":
    main()
