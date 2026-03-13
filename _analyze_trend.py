#!/usr/bin/env python3
"""Analyze score trends across turns for each condition."""
import json
from collections import defaultdict

RESULTS_FILE = "output/results_run_20260311_073316_34ae2c.jsonl"

rows = []
with open(RESULTS_FILE) as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

print(f"=== 轮次-评分趋势分析 (共 {len(rows)} 行) ===\n")

METRICS = [
    ("harm_1_10", "Harm"),
    ("negative_emotion_1_10", "Emotion"),
    ("inappropriate_1_10", "Inappr"),
    ("empathic_language_1_10", "Empathic"),
]
ANTHRO_KEYS = ["anthro_q1", "anthro_q2", "anthro_q3", "anthro_q4", "anthro_q5"]

# 按 condition -> turn_index -> list of scores
cond_turn = defaultdict(lambda: defaultdict(lambda: {m[0]: [] for m in METRICS} | {"anthro_avg": []}))

for r in rows:
    c = r.get("condition", "unknown")
    t = r.get("turn_index")
    if t is None:
        continue
    bucket = cond_turn[c][t]
    for key, _ in METRICS:
        v = r.get(key)
        if v is not None and v != "":
            try:
                bucket[key].append(float(v))
            except (ValueError, TypeError):
                pass
    aq = []
    for qi in ANTHRO_KEYS:
        v = r.get(qi)
        if v is not None and v != "":
            try:
                aq.append(float(v))
            except (ValueError, TypeError):
                pass
    if aq:
        bucket["anthro_avg"].append(sum(aq) / len(aq))


def avg(lst):
    return sum(lst) / len(lst) if lst else None

def fmt(v):
    return f"{v:>5.1f}" if v is not None else "    -"


for cond in ["default", "unhelpful", "cynical", "distant"]:
    if cond not in cond_turn:
        continue
    turns_data = cond_turn[cond]
    sorted_turns = sorted(turns_data.keys())
    if not sorted_turns:
        continue

    print(f"--- {cond.upper()} ---")
    header = f"{'Turn':>5} | {'Harm':>5} {'Emo':>5} {'Inap':>5} {'Empa':>5} | {'Anthro':>6}"
    print(header)
    print("-" * len(header))

    # Collect per-turn averages for trend calculation
    turn_scores = {m[0]: [] for m in METRICS}
    turn_scores["anthro_avg"] = []
    turn_indices = []

    for t in sorted_turns:
        b = turns_data[t]
        harm = avg(b["harm_1_10"])
        emo = avg(b["negative_emotion_1_10"])
        inap = avg(b["inappropriate_1_10"])
        empa = avg(b["empathic_language_1_10"])
        anth = avg(b["anthro_avg"])
        print(f"{t:>5} | {fmt(harm)} {fmt(emo)} {fmt(inap)} {fmt(empa)} | {fmt(anth)}")

        turn_indices.append(t)
        for key in turn_scores:
            val = avg(b[key])
            turn_scores[key].append(val)

    # Simple linear trend (slope) via least squares
    print()
    print("  趋势 (线性斜率/轮):")
    for key, label in METRICS + [("anthro_avg", "Anthro")]:
        vals = turn_scores[key]
        pairs = [(turn_indices[i], vals[i]) for i in range(len(vals)) if vals[i] is not None]
        if len(pairs) < 3:
            print(f"    {label:>8}: 数据不足")
            continue
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        n = len(xs)
        x_mean = sum(xs) / n
        y_mean = sum(ys) / n
        ss_xy = sum((xs[i] - x_mean) * (ys[i] - y_mean) for i in range(n))
        ss_xx = sum((xs[i] - x_mean) ** 2 for i in range(n))
        slope = ss_xy / ss_xx if ss_xx != 0 else 0
        # direction arrow
        if abs(slope) < 0.05:
            arrow = "→ 稳定"
        elif slope > 0:
            arrow = "↑ 上升"
        else:
            arrow = "↓ 下降"
        # first half vs second half comparison
        mid = len(ys) // 2
        first_half = [v for v in ys[:mid] if v is not None]
        second_half = [v for v in ys[mid:] if v is not None]
        fh_avg = sum(first_half) / len(first_half) if first_half else 0
        sh_avg = sum(second_half) / len(second_half) if second_half else 0
        delta = sh_avg - fh_avg
        print(f"    {label:>8}: slope={slope:+.3f}/turn  {arrow}  (前半均值={fh_avg:.2f} → 后半均值={sh_avg:.2f}, Δ={delta:+.2f})")

    print()
