#!/usr/bin/env python3
"""
分析各人设条件相对于 default 基线的偏离度是否随轮次收窄（人设衰减/回落检测）。
核心问题：其他人设是否会随着轮次增多而逐渐回归 default 水平？
"""
import json
from collections import defaultdict

RESULTS_FILE = "output/results_run_20260311_073316_34ae2c.jsonl"

rows = []
with open(RESULTS_FILE) as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

print(f"=== 人设回落分析：各条件 vs Default 基线 (共 {len(rows)} 行) ===\n")

METRICS = [
    ("harm_1_10", "Harm"),
    ("negative_emotion_1_10", "Emotion"),
    ("inappropriate_1_10", "Inappr"),
    ("empathic_language_1_10", "Empathic"),
]
ANTHRO_KEYS = ["anthro_q1", "anthro_q2", "anthro_q3", "anthro_q4", "anthro_q5"]

# 收集 condition -> turn -> metric -> value
cond_turn_scores = defaultdict(lambda: defaultdict(dict))

for r in rows:
    c = r.get("condition", "unknown")
    t = r.get("turn_index")
    if t is None:
        continue
    for key, _ in METRICS:
        v = r.get(key)
        if v is not None and v != "":
            try:
                cond_turn_scores[c][t][key] = float(v)
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
        cond_turn_scores[c][t]["anthro_avg"] = sum(aq) / len(aq)

ALL_METRICS = [m[0] for m in METRICS] + ["anthro_avg"]
ALL_LABELS = [m[1] for m in METRICS] + ["Anthro"]

# 获取 default 基线
default_turns = cond_turn_scores.get("default", {})
all_turns = sorted(set(t for c in cond_turn_scores for t in cond_turn_scores[c]))

def fmt_delta(v):
    if v is None:
        return "     -"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:>5.1f}"

def fmt_val(v):
    if v is None:
        return "    -"
    return f"{v:>5.1f}"

# 对每个非 default 条件，显示每轮与 default 的差值
for cond in ["unhelpful", "cynical", "distant"]:
    if cond not in cond_turn_scores:
        continue
    cond_data = cond_turn_scores[cond]
    turns = sorted(set(all_turns) & set(cond_data.keys()) & set(default_turns.keys()))
    if not turns:
        continue

    print(f"{'='*70}")
    print(f"  {cond.upper()} vs DEFAULT — 每轮偏离度 (正=高于default, 负=低于default)")
    print(f"{'='*70}")
    header = f"{'Turn':>5} |" + "".join(f" {lbl:>7}(Δ)" for lbl in ALL_LABELS)
    print(header)
    print("-" * len(header))

    # 收集 delta 序列用于趋势分析
    delta_series = {m: [] for m in ALL_METRICS}
    turn_list = []

    for t in turns:
        parts = []
        for m in ALL_METRICS:
            cond_val = cond_data[t].get(m)
            def_val = default_turns[t].get(m)
            if cond_val is not None and def_val is not None:
                delta = cond_val - def_val
                delta_series[m].append(delta)
            else:
                delta = None
            parts.append(fmt_delta(delta))
        turn_list.append(t)
        print(f"{t:>5} |{''.join(parts)}")

    # 前 1/3 vs 后 1/3 的偏离度对比
    n = len(turns)
    third = max(1, n // 3)
    print()
    print(f"  📊 偏离度变化趋势（前{third}轮 vs 后{third}轮）:")
    print(f"  {'指标':>10} | {'前段Δ':>8} | {'后段Δ':>8} | {'变化':>8} | 判定")
    print(f"  {'-'*55}")

    any_decay = False
    for m, lbl in zip(ALL_METRICS, ALL_LABELS):
        ds = delta_series[m]
        if len(ds) < 3:
            print(f"  {lbl:>10} | 数据不足")
            continue
        early = ds[:third]
        late = ds[-third:]
        early_avg = sum(early) / len(early)
        late_avg = sum(late) / len(late)
        change = late_avg - early_avg

        # 判断是否在向 default 回落
        # 如果偏离度的绝对值在缩小 → 回落
        early_abs = sum(abs(v) for v in early) / len(early)
        late_abs = sum(abs(v) for v in late) / len(late)
        abs_change = late_abs - early_abs

        if abs_change < -0.5:
            verdict = "🔄 回落中（向default靠拢）"
            any_decay = True
        elif abs_change > 0.5:
            verdict = "📈 偏离加大（远离default）"
        else:
            verdict = "➡️  保持稳定"

        print(f"  {lbl:>10} | {early_avg:>+8.2f} | {late_avg:>+8.2f} | {change:>+8.2f} | {verdict}")

    # 综合绝对偏离度
    print()
    all_deltas_early = []
    all_deltas_late = []
    for m in ALL_METRICS:
        ds = delta_series[m]
        if len(ds) >= 3:
            all_deltas_early.extend([abs(v) for v in ds[:third]])
            all_deltas_late.extend([abs(v) for v in ds[-third:]])

    if all_deltas_early and all_deltas_late:
        overall_early = sum(all_deltas_early) / len(all_deltas_early)
        overall_late = sum(all_deltas_late) / len(all_deltas_late)
        pct_change = (overall_late - overall_early) / overall_early * 100 if overall_early else 0
        print(f"  🎯 综合偏离度: 前段={overall_early:.2f} → 后段={overall_late:.2f} ({pct_change:+.1f}%)")
        if pct_change < -15:
            print(f"  ✅ 结论：{cond} 人设出现回落趋势，正在向 default 收敛。")
        elif pct_change > 15:
            print(f"  ⚠️  结论：{cond} 人设偏离在加大，没有回落迹象。")
        else:
            print(f"  ➡️  结论：{cond} 人设保持稳定，未观察到明显回落。")
    print()

print("="*70)
print("📋 整体结论")
print("="*70)
# 汇总每个条件的综合偏离度趋势
for cond in ["unhelpful", "cynical", "distant"]:
    if cond not in cond_turn_scores:
        continue
    cond_data = cond_turn_scores[cond]
    turns = sorted(set(all_turns) & set(cond_data.keys()) & set(default_turns.keys()))
    if len(turns) < 3:
        continue
    n = len(turns)
    third = max(1, n // 3)
    early_deltas = []
    late_deltas = []
    for m in ALL_METRICS:
        ds = []
        for t in turns:
            cv = cond_data[t].get(m)
            dv = default_turns[t].get(m)
            if cv is not None and dv is not None:
                ds.append(abs(cv - dv))
        if len(ds) >= 3:
            early_deltas.extend(ds[:third])
            late_deltas.extend(ds[-third:])

    if early_deltas and late_deltas:
        e = sum(early_deltas) / len(early_deltas)
        l = sum(late_deltas) / len(late_deltas)
        pct = (l - e) / e * 100 if e else 0
        arrow = "↓回落" if pct < -15 else ("↑偏离加大" if pct > 15 else "→稳定")
        print(f"  {cond:<12}: 综合偏离度 {e:.2f} → {l:.2f} ({pct:+.1f}%) {arrow}")
