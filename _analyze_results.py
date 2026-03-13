#!/usr/bin/env python3
"""Quick analysis of current experiment results."""
import json
from collections import defaultdict

RESULTS_FILE = "output/results_run_20260311_073316_34ae2c.jsonl"

rows = []
with open(RESULTS_FILE) as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

print(f"=== 总行数: {len(rows)} ===")
print()

# 按 condition 分组统计
cond_stats = defaultdict(lambda: {
    "count": 0, "harm": [], "emotion": [], "inappropriate": [],
    "empathic": [], "anthro": [], "errors": 0, "refusals": 0
})

for r in rows:
    c = r.get("condition", "unknown")
    s = cond_stats[c]
    s["count"] += 1
    if r.get("error_stage"):
        s["errors"] += 1
    if r.get("refusal_detected"):
        s["refusals"] += 1
    for key, field in [
        ("harm", "harm_1_10"),
        ("emotion", "negative_emotion_1_10"),
        ("inappropriate", "inappropriate_1_10"),
        ("empathic", "empathic_language_1_10"),
    ]:
        v = r.get(field)
        if v is not None and v != "":
            try:
                s[key].append(float(v))
            except (ValueError, TypeError):
                pass
    aq = []
    for qi in ["anthro_q1", "anthro_q2", "anthro_q3", "anthro_q4", "anthro_q5"]:
        v = r.get(qi)
        if v is not None and v != "":
            try:
                aq.append(float(v))
            except (ValueError, TypeError):
                pass
    if aq:
        s["anthro"].append(sum(aq) / len(aq))


def avg(lst):
    return sum(lst) / len(lst) if lst else None

def fmt(v):
    return f"{v:.2f}" if v is not None else "-"

header = f"{'Condition':<14} {'N':>4} {'Err':>4} {'Ref':>4} | {'Harm':>6} {'Emotion':>8} {'Inappr':>7} {'Empathic':>8} | {'Anthro':>7}"
print(header)
print("-" * len(header))
for c in ["default", "unhelpful", "cynical", "distant"]:
    s = cond_stats.get(c)
    if not s:
        continue
    print(
        f"{c:<14} {s['count']:>4} {s['errors']:>4} {s['refusals']:>4} "
        f"| {fmt(avg(s['harm'])):>6} {fmt(avg(s['emotion'])):>8} "
        f"{fmt(avg(s['inappropriate'])):>7} {fmt(avg(s['empathic'])):>8} "
        f"| {fmt(avg(s['anthro'])):>7}"
    )
for c, s in cond_stats.items():
    if c not in ["default", "unhelpful", "cynical", "distant"]:
        print(
            f"{c:<14} {s['count']:>4} {s['errors']:>4} {s['refusals']:>4} "
            f"| {fmt(avg(s['harm'])):>6} {fmt(avg(s['emotion'])):>8} "
            f"{fmt(avg(s['inappropriate'])):>7} {fmt(avg(s['empathic'])):>8} "
            f"| {fmt(avg(s['anthro'])):>7}"
        )

print()
print("=== 每条对话的完成进度 ===")
dlg_cond = defaultdict(lambda: defaultdict(list))
for r in rows:
    dlg_cond[r.get("dialogue_id", "?")][r.get("condition", "?")].append(r.get("turn_index"))

for dlg_id in sorted(dlg_cond.keys()):
    parts = []
    for cond in sorted(dlg_cond[dlg_id].keys()):
        turns = dlg_cond[dlg_id][cond]
        parts.append(f"{cond}: turns={sorted(turns)}")
    print(f"  {dlg_id}: {' | '.join(parts)}")

print()
print("=== 错误详情 ===")
errs = [r for r in rows if r.get("error_stage")]
if errs:
    for r in errs:
        msg = (r.get("error_message", "") or "")[:150]
        print(f"  {r.get('dialogue_id','?')} / {r.get('condition','?')} / turn {r.get('turn_index','?')} => {r.get('error_stage')}: {msg}")
else:
    print("  无错误！")

print()
print("=== 最新 5 条结果 ===")
for r in rows[-5:]:
    cond = r.get("condition", "?")
    turn = r.get("turn_index", "?")
    dlg = r.get("dialogue_id", "?")[:25]
    user = (r.get("user_text", "") or "")[:80]
    reply = (r.get("model_reply", "") or "")[:80]
    harm = r.get("harm_1_10", "-")
    emo = r.get("negative_emotion_1_10", "-")
    inap = r.get("inappropriate_1_10", "-")
    emp = r.get("empathic_language_1_10", "-")
    err = r.get("error_stage", "")
    print(f"  [{cond}] {dlg} turn={turn}  harm={harm} emo={emo} inap={inap} emp={emp} {'ERR:'+err if err else ''}")
    print(f"    User: {user}")
    print(f"    Model: {reply}")
    print()

# 整体统计
total_errors = sum(1 for r in rows if r.get("error_stage"))
total_refusals = sum(1 for r in rows if r.get("refusal_detected"))
all_harm = [float(r["harm_1_10"]) for r in rows if r.get("harm_1_10") is not None and r.get("harm_1_10") != ""]
all_emo = [float(r["negative_emotion_1_10"]) for r in rows if r.get("negative_emotion_1_10") is not None and r.get("negative_emotion_1_10") != ""]

print("=== 整体摘要 ===")
print(f"  总行数: {len(rows)}")
print(f"  错误数: {total_errors}  ({total_errors/len(rows)*100:.1f}%)" if rows else "")
print(f"  拒绝数: {total_refusals}")
print(f"  涉及 conditions: {sorted(cond_stats.keys())}")
print(f"  涉及对话数: {len(dlg_cond)}")
print(f"  全局平均 harm: {fmt(avg(all_harm))}")
print(f"  全局平均 emotion: {fmt(avg(all_emo))}")
