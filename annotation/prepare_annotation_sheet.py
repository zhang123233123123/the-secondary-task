"""
Prepare calibration set for human annotation with 1-7 scale fields.

Reads:  few_shot_design/calibration_set.jsonl
Output: annotation/calibration_set_for_annotation.jsonl
        annotation/calibration_set_for_annotation.csv  (for Google Sheets)
"""
import json
import csv
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="few_shot_design/calibration_set.jsonl")
    p.add_argument("--output_jsonl", default="annotation/calibration_set_for_annotation.jsonl")
    p.add_argument("--output_csv",   default="annotation/calibration_set_for_annotation.csv")
    args = p.parse_args()

    rows = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    out_rows = []
    for r in rows:
        out_rows.append({
            "cal_id":       r["cal_id"],
            "domain":       r["domain"],
            "turn_index":   r["turn_index"],
            "epoch":        r["epoch"],
            "llm3_model":   r.get("llm3_model", ""),
            "user_text":    r["user_text"],
            "model_reply":  r["model_reply"],
            # LLM4 reference scores (for post-hoc comparison only, NOT shown to annotators)
            "_llm4_harm":       r.get("llm4_harm"),
            "_llm4_neg_emo":    r.get("llm4_neg_emo"),
            "_llm4_inapp":      r.get("llm4_inapp"),
            "_llm4_empathic":   r.get("llm4_empathic"),
            # Human annotation fields — 1-7 scale
            "human_harm_1_7":        None,
            "human_neg_emo_1_7":     None,
            "human_inapp_1_7":       None,
            "human_empathic_1_7":    None,
            "human_anthro_q1_1_7":   None,
            "human_anthro_q2_1_7":   None,
            "human_anthro_q3_1_7":   None,
            "human_anthro_q4_1_7":   None,
            "human_anthro_q5_1_7":   None,
            "annotator_id":          None,
            "notes":                 None,
        })

    # Save JSONL
    out = Path(args.output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"JSONL saved → {out}  ({len(out_rows)} rows)")

    # Save CSV (hide _llm4 fields for annotators)
    csv_fields = [
        "cal_id", "domain", "turn_index", "epoch", "llm3_model",
        "user_text", "model_reply",
        "human_harm_1_7", "human_neg_emo_1_7", "human_inapp_1_7",
        "human_empathic_1_7",
        "human_anthro_q1_1_7", "human_anthro_q2_1_7", "human_anthro_q3_1_7",
        "human_anthro_q4_1_7", "human_anthro_q5_1_7",
        "annotator_id", "notes",
    ]
    csv_out = Path(args.output_csv)
    with csv_out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"CSV  saved → {csv_out}  (ready for Google Sheets)")


if __name__ == "__main__":
    main()
