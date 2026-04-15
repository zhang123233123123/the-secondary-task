"""
Select few-shot anchor examples from existing LLM4-scored data.

Source: stress_test_100turn_deepseek aggregate (20,755 rows, scores in 1-10 scale)
Output: annotation/fewshot_candidates.jsonl

Score conversion (Malu, 2026-04-13):
  harm / neg_emo / inapp / empathic : 1-10 → 1-7  (linear)
  anthro_q1~q5                      : 1-5  (kept as-is, Godspeed validated scale)

Selection strategy:
  - 1 high-scoring example  (cynical/distant: inapp+neg_emo high)
  - 1 low-scoring example   (default: empathic high, harm/inapp low)
  - 1 mid-range example     (any condition: scores near midpoint 4)
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict


def scale_10_to_7(v):
    """Linear transform 1-10 → 1-7, rounded to nearest integer."""
    if v is None:
        return None
    return round(1 + (v - 1) * 6 / 9)


def scale_5_to_7(v):
    """Linear transform 1-5 → 1-7, rounded to nearest integer."""
    if v is None:
        return None
    return round(1 + (v - 1) * 6 / 4)


def convert_row(r: dict) -> dict:
    """Return row with all scores converted to 1-7."""
    return {
        **r,
        "harm_1_7":     scale_10_to_7(r.get("harm_1_10")),
        "neg_emo_1_7":  scale_10_to_7(r.get("negative_emotion_1_10")),
        "inapp_1_7":    scale_10_to_7(r.get("inappropriate_1_10")),
        "empathic_1_7": scale_10_to_7(r.get("empathic_language_1_10")),
        "anthro_q1_1_7": scale_5_to_7(r.get("anthro_q1")),
        "anthro_q2_1_7": scale_5_to_7(r.get("anthro_q2")),
        "anthro_q3_1_7": scale_5_to_7(r.get("anthro_q3")),
        "anthro_q4_1_7": scale_5_to_7(r.get("anthro_q4")),
        "anthro_q5_1_7": scale_5_to_7(r.get("anthro_q5")),
    }


def composite_high(r: dict) -> float:
    """Score for selecting high-harm/neg/inapp examples."""
    return (r.get("harm_1_7") or 0) + (r.get("neg_emo_1_7") or 0) + (r.get("inapp_1_7") or 0)


def composite_low(r: dict) -> float:
    """Score for selecting low-harm, high-empathic examples."""
    return (r.get("empathic_1_7") or 0) - (r.get("harm_1_7") or 0) - (r.get("inapp_1_7") or 0)


def composite_mid(r: dict) -> float:
    """Distance from midpoint 4 on all dimensions (lower = more mid-range)."""
    vals = [
        r.get("harm_1_7"), r.get("neg_emo_1_7"),
        r.get("inapp_1_7"), r.get("empathic_1_7"),
    ]
    vals = [v for v in vals if v is not None]
    return -sum(abs(v - 4) for v in vals)  # negative so higher = closer to mid


def composite_low_anthro(r: dict) -> float:
    """Score for selecting low-anthro (machine-like) examples from distant condition.
    Prioritise low anthro mean; secondary: low harm/inapp (clean signal)."""
    anthro_vals = [
        r.get(f"anthro_q{j}_1_7") for j in range(1, 6)
    ]
    anthro_vals = [v for v in anthro_vals if v is not None]
    anthro_mean = sum(anthro_vals) / len(anthro_vals) if anthro_vals else 7
    # Lower anthro = better; lower harm/inapp = cleaner signal
    return -(anthro_mean + 0.2 * (r.get("harm_1_7") or 4) + 0.2 * (r.get("inapp_1_7") or 4))


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
                and r.get("condition") in ("default", "cynical", "distant")
                and r.get("harm_1_10") is not None
            ):
                rows.append(convert_row(r))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=
        "stress_test_100turn_deepseek/output_aggregate_20260322/"
        "results_aggregate_success_20260322.jsonl")
    p.add_argument("--output", default="annotation/fewshot_candidates.jsonl")
    p.add_argument("--n_high", type=int, default=2,
                   help="Number of high-scoring examples (cynical/distant)")
    p.add_argument("--n_low", type=int, default=2,
                   help="Number of low-scoring examples (default)")
    p.add_argument("--n_mid", type=int, default=1,
                   help="Number of mid-range examples")
    p.add_argument("--n_low_anthro", type=int, default=1,
                   help="Number of low-anthro (machine-like) examples from distant")
    args = p.parse_args()

    rows = load_valid(args.source)
    print(f"Loaded {len(rows)} valid rows")

    by_cond = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)

    selected = []

    # HIGH: cynical + distant, ranked by harm+neg+inapp
    for cond in ("cynical", "distant"):
        pool = sorted(by_cond[cond], key=composite_high, reverse=True)
        picked = pool[:args.n_high]
        for r in picked:
            r["_selection_type"] = f"high_{cond}"
        selected.extend(picked)
        print(f"High ({cond}): harm={picked[0].get('harm_1_7')} "
              f"neg={picked[0].get('neg_emo_1_7')} "
              f"inapp={picked[0].get('inapp_1_7')} "
              f"empathic={picked[0].get('empathic_1_7')}")

    # LOW: default, ranked by empathic - harm - inapp
    pool = sorted(by_cond["default"], key=composite_low, reverse=True)
    picked = pool[:args.n_low]
    for r in picked:
        r["_selection_type"] = "low_default"
    selected.extend(picked)
    print(f"Low (default): harm={picked[0].get('harm_1_7')} "
          f"neg={picked[0].get('neg_emo_1_7')} "
          f"inapp={picked[0].get('inapp_1_7')} "
          f"empathic={picked[0].get('empathic_1_7')}")

    # LOW ANTHRO: distant condition, lowest anthro mean — covers machine-like end
    pool = sorted(by_cond["distant"], key=composite_low_anthro, reverse=True)
    # Exclude rows already selected
    pool = [r for r in pool if r not in selected]
    picked = pool[:args.n_low_anthro]
    for r in picked:
        r["_selection_type"] = "low_anthro_distant"
    selected.extend(picked)
    anthro_vals = [picked[0].get(f"anthro_q{j}_1_7") for j in range(1, 6)]
    anthro_mean = sum(v for v in anthro_vals if v) / max(1, sum(1 for v in anthro_vals if v))
    print(f"Low anthro (distant): harm={picked[0].get('harm_1_7')} "
          f"neg={picked[0].get('neg_emo_1_7')} "
          f"inapp={picked[0].get('inapp_1_7')} "
          f"anthro_mean={anthro_mean:.1f}")

    # MID: any condition, closest to midpoint 4
    all_rows = [r for r in rows if r not in selected]
    pool = sorted(all_rows, key=composite_mid, reverse=True)
    picked = pool[:args.n_mid]
    for r in picked:
        r["_selection_type"] = "mid"
    selected.extend(picked)
    print(f"Mid: harm={picked[0].get('harm_1_7')} "
          f"neg={picked[0].get('neg_emo_1_7')} "
          f"inapp={picked[0].get('inapp_1_7')} "
          f"empathic={picked[0].get('empathic_1_7')}")

    # Save
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in selected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{len(selected)} examples saved → {out}")
    print("\nSummary:")
    print(f"{'type':<20} {'cond':<10} {'domain':<15} {'turn':>5} "
          f"{'harm':>5} {'neg':>5} {'inapp':>5} {'emp':>5}")
    print("-" * 70)
    for r in selected:
        print(f"{r['_selection_type']:<20} {r['condition']:<10} "
              f"{r['domain']:<15} {r['turn_index']:>5} "
              f"{r.get('harm_1_7'):>5} {r.get('neg_emo_1_7'):>5} "
              f"{r.get('inapp_1_7'):>5} {r.get('empathic_1_7'):>5}")


if __name__ == "__main__":
    main()
