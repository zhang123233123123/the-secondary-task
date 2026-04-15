"""
Merge all annotator responses and compute inter-rater agreement.

Input : annotation/responses/annotator_*.jsonl
        annotation/calibration_set_36.jsonl  (for LLM4 reference scores)
Output: annotation/responses/merged.jsonl
        annotation/responses/agreement_report.txt
"""
import json
from pathlib import Path
from collections import defaultdict
import statistics

RESP_DIR  = Path(__file__).parent / "responses"
DATA_FILE = Path(__file__).parent / "calibration_set_36.jsonl"

SCORE_KEYS = [
    "human_harm_1_7", "human_neg_emo_1_7", "human_inapp_1_7",
    "human_empathic_1_7",
    "human_anthro_q1_1_7", "human_anthro_q2_1_7", "human_anthro_q3_1_7",
    "human_anthro_q4_1_7", "human_anthro_q5_1_7",
]

LLM4_KEYS = [
    "_llm4_harm_1_7", "_llm4_neg_emo_1_7", "_llm4_inapp_1_7",
    "_llm4_empathic_1_7",
    "_llm4_anthro_q1_1_7", "_llm4_anthro_q2_1_7", "_llm4_anthro_q3_1_7",
    "_llm4_anthro_q4_1_7", "_llm4_anthro_q5_1_7",
]


def load_items() -> dict[str, dict]:
    items = {}
    with DATA_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                items[r["cal_id"]] = r
    return items


def load_all_responses() -> dict[str, list[dict]]:
    """Load all responses, grouped by cal_id."""
    by_item: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(RESP_DIR.glob("annotator_*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    by_item[r["cal_id"]].append(r)
    return by_item


def pearson_r(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx  = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy  = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def main():
    items    = load_items()
    by_item  = load_all_responses()

    if not by_item:
        print("No responses found in annotation/responses/")
        return

    print(f"Loaded responses for {len(by_item)} items from "
          f"{len(list(RESP_DIR.glob('annotator_*.jsonl')))} annotators\n")

    # --- Merged rows (mean score per item) ---
    merged = []
    for cal_id, responses in sorted(by_item.items()):
        item = items.get(cal_id, {})
        row = {
            "cal_id":      cal_id,
            "condition":   item.get("condition"),
            "domain":      item.get("domain"),
            "turn_index":  item.get("turn_index"),
            "epoch":       item.get("epoch"),
            "n_annotators": len(responses),
        }
        # Mean human score per dimension
        for key in SCORE_KEYS:
            vals = [r[key] for r in responses if r.get(key) is not None]
            row[key + "_mean"] = round(statistics.mean(vals), 3) if vals else None
            row[key + "_std"]  = round(statistics.stdev(vals), 3) if len(vals) > 1 else None
        # LLM4 scores
        for key in LLM4_KEYS:
            row[key] = item.get(key)
        merged.append(row)

    # Save merged
    out = RESP_DIR / "merged.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Merged saved → {out}  ({len(merged)} items)")

    # --- Agreement report ---
    report_lines = ["=" * 60, "LLM4 vs Human Agreement (Pearson r)", "=" * 60]

    dim_pairs = list(zip(SCORE_KEYS, LLM4_KEYS))
    for h_key, l_key in dim_pairs:
        h_vals = [r[h_key + "_mean"] for r in merged if r.get(h_key + "_mean") is not None]
        l_vals = [r[l_key] for r in merged if r.get(l_key) is not None and
                  r.get(h_key + "_mean") is not None]
        r_val = pearson_r(h_vals, l_vals)
        label = h_key.replace("human_", "").replace("_1_7", "")
        r_str = f"{r_val:.3f}" if r_val is not None else "N/A"
        flag  = "✅" if r_val and r_val >= 0.7 else ("⚠️" if r_val and r_val >= 0.5 else "❌")
        report_lines.append(f"  {label:<22} r = {r_str}  {flag}")

    report_lines += ["", "Inter-rater agreement (pairwise mean absolute deviation)", "-" * 60]
    all_responses = load_all_responses()
    for cal_id, responses in sorted(all_responses.items()):
        if len(responses) < 2:
            continue
        for key in SCORE_KEYS:
            vals = [r[key] for r in responses if r.get(key) is not None]
            if len(vals) >= 2:
                pairs = [(vals[i], vals[j])
                         for i in range(len(vals))
                         for j in range(i + 1, len(vals))]
                mad = statistics.mean(abs(a - b) for a, b in pairs)
                if mad >= 3:
                    label = key.replace("human_", "").replace("_1_7", "")
                    report_lines.append(
                        f"  ⚠️ {cal_id} {label}: MAD={mad:.1f} "
                        f"(scores: {vals}) — needs discussion"
                    )

    report = "\n".join(report_lines)
    print("\n" + report)

    report_path = RESP_DIR / "agreement_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved → {report_path}")


if __name__ == "__main__":
    main()
