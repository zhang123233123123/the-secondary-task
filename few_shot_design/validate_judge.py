"""
Step 5: Validate LLM4 judge against human scores on held-out calibration items.

Two modes:
  --mode existing : compare existing llm4_* scores in the calibration set
                    (quick check before deploying few-shot prompt)
  --mode rerun    : call LLM4 with the new few-shot judge prompt on held-out items
                    (true validation of the updated prompt)

Output: per-dimension Pearson r, printed to stdout and written to a JSON report.
Target: r > 0.80 following AlpacaEval validation practice.

Usage examples:
  # Quick check using stored LLM4 scores
  python few_shot_design/validate_judge.py \\
    --annotated few_shot_design/calibration_set_annotated.jsonl \\
    --anchor_ids cal_001 cal_005 cal_012 cal_018 cal_022 cal_027 cal_034 cal_040 \\
    --mode existing

  # Full re-run validation with few-shot prompt
  python few_shot_design/validate_judge.py \\
    --annotated few_shot_design/calibration_set_annotated.jsonl \\
    --anchor_ids cal_001 cal_005 cal_012 cal_018 cal_022 cal_027 cal_034 cal_040 \\
    --mode rerun \\
    --config stress_test_150turn_deepseek/config_150turn.yaml \\
    --judge_prompt few_shot_design/judge_system_fewshot.txt \\
    --output few_shot_design/validation_report.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Statistics helpers (pure Python — no numpy/scipy required)
# ---------------------------------------------------------------------------

def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _pearson_r(xs: list[float], ys: list[float]) -> float | None:
    """Pearson r. Returns None if variance is zero in either series."""
    n = len(xs)
    if n < 3:
        return None
    mx, my = _mean(xs), _mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def _krippendorff_alpha_ordinal(ratings: list[list[float | None]]) -> float:
    """
    Krippendorff's alpha for ordinal data (simplified).
    ratings[i][j] = annotator j's score for item i  (None = missing).
    """
    # Collect all paired differences
    Do = 0.0
    De = 0.0
    n_pairs_o = 0
    n_pairs_e = 0

    # Observed disagreement
    for item_ratings in ratings:
        vals = [v for v in item_ratings if v is not None]
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                Do += (vals[i] - vals[j]) ** 2
                n_pairs_o += 1

    if n_pairs_o == 0:
        return 1.0

    # Expected disagreement (all values pooled)
    all_vals: list[float] = []
    for item_ratings in ratings:
        all_vals.extend(v for v in item_ratings if v is not None)
    n = len(all_vals)
    if n < 2:
        return 1.0
    mu = _mean(all_vals)
    De = sum((v - mu) ** 2 for v in all_vals) * n / (n - 1)

    if De == 0:
        return 1.0
    return 1.0 - (Do / n_pairs_o) / (De / len(all_vals))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

SCORE_MAP = [
    # (human_field,         llm4_field,              display_name)
    ("human_harm_1_10",     "llm4_harm",             "harm_1_10"),
    ("human_neg_emo_1_10",  "llm4_neg_emo",          "negative_emotion_1_10"),
    ("human_inapp_1_10",    "llm4_inapp",             "inappropriate_1_10"),
    ("human_empathic_1_10", "llm4_empathic",          "empathic_language_1_10"),
    ("human_anthro_1_5",    "_llm4_anthro_mean",      "anthro_mean"),
]


def _llm4_anthro_mean(row: dict) -> float | None:
    vals = [row.get(f"llm4_anthro_q{i}") for i in range(1, 6)]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def load_annotated(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def merge_human_scores(rows: list[dict]) -> dict[str, dict]:
    """Group by cal_id, return mean human scores (averaging annotators)."""
    by_id: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_id[r["cal_id"]].append(r)

    merged: dict[str, dict] = {}
    for cal_id, group in by_id.items():
        base = dict(group[0])  # keep metadata from first annotator
        for hf, _, _ in SCORE_MAP:
            if hf == "human_anthro_1_5":
                vals = [g.get("human_anthro_1_5") for g in group
                        if g.get("human_anthro_1_5") is not None]
            else:
                vals = [g.get(hf) for g in group if g.get(hf) is not None]
            base[hf] = (sum(vals) / len(vals)) if vals else None

        # Compute LLM4 anthro mean helper field
        base["_llm4_anthro_mean"] = _llm4_anthro_mean(base)
        merged[cal_id] = base

    return merged


# ---------------------------------------------------------------------------
# LLM4 re-run helpers
# ---------------------------------------------------------------------------

def _call_llm4(client, judge_system: str, user_text: str, model_reply: str,
               retries: int = 3, timeout: int = 60) -> dict | None:
    """Call LLM4 with few-shot judge prompt, return parsed scores or None."""
    from backend.llm_clients import make_llm_client, parse_judge_json  # noqa: E402

    messages = [
        {"role": "system", "content": judge_system},
        {"role": "user", "content": (
            f"User message: {user_text}\n\nAssistant reply: {model_reply}"
        )},
    ]

    for attempt in range(retries + 1):
        try:
            result = client.chat(messages, timeout)
            parsed = parse_judge_json(result.text)
            if parsed:
                return parsed
        except Exception as exc:  # noqa: BLE001
            if attempt < retries:
                time.sleep(0.5 * (2 ** attempt))
            else:
                print(f"    LLM4 call failed: {exc}", file=sys.stderr)
    return None


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_report(results: dict, n_held_out: int, mode: str) -> None:
    print(f"\n{'='*60}")
    print(f"Validation report  |  mode={mode}  |  n={n_held_out} held-out items")
    print(f"{'='*60}")
    print(f"{'Dimension':<28} {'Pearson r':>10}  {'N pairs':>8}  {'Status':>10}")
    print(f"{'-'*60}")
    for dim, info in results.items():
        r = info.get("pearson_r")
        n = info.get("n")
        if r is None:
            status = "N/A"
            r_str = "—"
        else:
            status = "PASS (r>0.80)" if r > 0.80 else "WARN (r≤0.80)"
            r_str = f"{r:.3f}"
        print(f"  {dim:<26} {r_str:>10}  {n:>8}  {status:>10}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--annotated", required=True,
                   help="Annotated JSONL (human scores filled in)")
    p.add_argument("--anchor_ids", nargs="*", default=[],
                   help="cal_ids used as few-shot anchors (excluded from validation)")
    p.add_argument("--anchor_ids_file", default=None,
                   help="Text file listing anchor cal_ids, one per line")
    p.add_argument("--mode", choices=["existing", "rerun"], default="existing",
                   help="existing=use stored llm4_* scores; rerun=call LLM4 with new prompt")
    p.add_argument("--config", default=None,
                   help="(rerun mode) Path to config YAML with llm4 settings")
    p.add_argument("--judge_prompt", default=None,
                   help="(rerun mode) Path to few-shot judge_system text file")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--output", default="few_shot_design/validation_report.json")
    args = p.parse_args()

    # Anchor IDs
    anchor_ids: set[str] = set(args.anchor_ids)
    if args.anchor_ids_file:
        af = Path(args.anchor_ids_file)
        if af.exists():
            for line in af.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    anchor_ids.add(line)

    # Load and merge
    rows = load_annotated(args.annotated)
    print(f"Loaded {len(rows)} annotation rows")
    merged = merge_human_scores(rows)
    print(f"Merged into {len(merged)} unique items")

    # Held-out: has human scores, not an anchor, no generate error
    held_out = [
        r for cal_id, r in merged.items()
        if cal_id not in anchor_ids
        and r.get("human_harm_1_10") is not None  # has been annotated
        and not r.get("error_stage")
    ]
    print(f"Held-out items (annotated, not anchors): {len(held_out)}")

    if not held_out:
        print("ERROR: no held-out items found. Check --anchor_ids or annotation completeness.")
        return 1

    # -----------------------------------------------------------------------
    # Collect LLM4 scores
    # -----------------------------------------------------------------------
    if args.mode == "existing":
        print("Mode: existing — using stored llm4_* scores")
        for r in held_out:
            r["_llm4_anthro_mean"] = _llm4_anthro_mean(r)

    else:  # rerun
        if not args.config or not args.judge_prompt:
            print("ERROR: --config and --judge_prompt required for rerun mode")
            return 1

        from backend.llm_clients import make_llm_client  # noqa: E402
        from backend.runtime_config import load_config  # noqa: E402
        from concurrent.futures import ThreadPoolExecutor, as_completed

        config = load_config(args.config)
        client = make_llm_client(config.llm4)
        judge_system = Path(args.judge_prompt).read_text(encoding="utf-8")

        print(f"Mode: rerun — calling LLM4 ({config.llm4.model}) on {len(held_out)} items")

        def _judge_item(r: dict) -> dict:
            scores = _call_llm4(
                client, judge_system,
                r["user_text"], r["model_reply"],
                retries=3, timeout=getattr(config, "timeout_seconds", 60),
            )
            if scores:
                r["llm4_harm"] = scores.get("harm_1_10")
                r["llm4_neg_emo"] = scores.get("negative_emotion_1_10")
                r["llm4_inapp"] = scores.get("inappropriate_1_10")
                r["llm4_empathic"] = scores.get("empathic_language_1_10")
                for i in range(1, 6):
                    r[f"llm4_anthro_q{i}"] = scores.get(f"anthro_q{i}")
                r["_llm4_anthro_mean"] = _llm4_anthro_mean(r)
                r["_rerun_error"] = None
            else:
                r["_rerun_error"] = "failed"
            return r

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_judge_item, r): r["cal_id"] for r in held_out}
            done = 0
            for future in as_completed(futures):
                future.result()
                done += 1
                if done % 5 == 0 or done == len(held_out):
                    print(f"  Judged {done}/{len(held_out)}")

        rerun_errors = sum(1 for r in held_out if r.get("_rerun_error"))
        if rerun_errors:
            print(f"WARNING: {rerun_errors} items had LLM4 errors (excluded from r)")

    # -----------------------------------------------------------------------
    # Compute Pearson r per dimension
    # -----------------------------------------------------------------------
    results: dict[str, dict] = {}

    for hf, lf, dim in SCORE_MAP:
        pairs = [
            (r[hf], r[lf])
            for r in held_out
            if r.get(hf) is not None and r.get(lf) is not None
        ]
        if not pairs:
            results[dim] = {"pearson_r": None, "n": 0}
            continue
        hs = [p[0] for p in pairs]
        ls = [p[1] for p in pairs]
        r_val = _pearson_r(hs, ls)
        results[dim] = {
            "pearson_r": round(r_val, 4) if r_val is not None else None,
            "n": len(pairs),
            "human_mean": round(_mean(hs), 3),
            "llm4_mean": round(_mean(ls), 3),
        }

    # Krippendorff's alpha between human and LLM4 for each dimension
    for hf, lf, dim in SCORE_MAP:
        pairs = [
            (r[hf], r[lf])
            for r in held_out
            if r.get(hf) is not None and r.get(lf) is not None
        ]
        if len(pairs) < 3:
            results[dim]["krippendorff_alpha"] = None
            continue
        # Two raters: human and LLM4
        ratings = [[p[0], p[1]] for p in pairs]
        alpha = _krippendorff_alpha_ordinal(ratings)
        results[dim]["krippendorff_alpha"] = round(alpha, 4)

    _print_report(results, len(held_out), args.mode)

    # -----------------------------------------------------------------------
    # Save report
    # -----------------------------------------------------------------------
    report = {
        "mode": args.mode,
        "n_held_out": len(held_out),
        "anchor_ids": sorted(anchor_ids),
        "dimensions": results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report saved → {out}")

    # Overall pass/fail
    r_values = [v["pearson_r"] for v in results.values() if v["pearson_r"] is not None]
    if r_values and min(r_values) >= 0.80:
        print("OVERALL: PASS — all dimensions r > 0.80")
        return 0
    elif r_values:
        failing = [dim for dim, v in results.items()
                   if v.get("pearson_r") is not None and v["pearson_r"] < 0.80]
        print(f"OVERALL: WARN — dimensions below 0.80: {failing}")
        return 0
    else:
        print("OVERALL: INSUFFICIENT DATA")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
