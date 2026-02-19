from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_tail_rows(results_path: Path, limit: int = 5) -> list[dict[str, Any]]:
    if not results_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in results_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            rows.append(json.loads(stripped))
        except json.JSONDecodeError:
            continue
    return rows[-limit:]


def write_report(
    output_dir: str | Path,
    run_id: str,
    summary: dict[str, Any],
    *,
    dry_run: bool,
    results_path: str | Path,
) -> Path:
    output_path = Path(output_dir) / f"report_{run_id}.md"
    tail_rows = _read_tail_rows(Path(results_path), limit=5)

    lines: list[str] = [
        f"# Run Report: {run_id}",
        "",
        "## Meta",
        f"- dry_run: `{str(dry_run).lower()}`",
        f"- results_file: `results_{run_id}.jsonl`",
        f"- summary_file: `run_summary_{run_id}.json`",
        f"- flush_policy_requested: `{summary.get('flush_policy_requested')}`",
        f"- flush_policy_effective: `{summary.get('flush_policy_effective')}`",
        "",
        "## Summary",
        f"- expected_rows: `{summary.get('expected_rows')}`",
        f"- actual_rows: `{summary.get('actual_rows')}`",
        f"- error_rows: `{summary.get('error_rows')}`",
        f"- error_rate: `{summary.get('error_rate')}`",
        f"- refusal_count: `{summary.get('refusal_count')}`",
        f"- refusal_rate: `{summary.get('refusal_rate')}`",
        f"- truncated_count: `{summary.get('truncated_count')}`",
        f"- aborted: `{summary.get('aborted')}`",
        f"- abort_reason: `{summary.get('abort_reason')}`",
        "",
        "## Error Breakdown",
        f"- generate_errors: `{summary.get('generate_errors')}`",
        f"- judge_errors: `{summary.get('judge_errors')}`",
        f"- judge_parse_errors: `{summary.get('judge_parse_errors')}`",
        "",
        "## Validation Evidence",
        "- smoke_test_command: `python control_agent.py run --config config.yaml --dry_run`",
        f"- this_run_mode: `{'dry_run' if dry_run else 'full_run'}`",
        f"- validation_log_file: `{summary.get('validation_log_file')}`",
        (
            "- summary_snapshot: "
            + f"actual_rows={summary.get('actual_rows')}, "
            + f"error_rate={summary.get('error_rate')}, "
            + f"refusal_rate={summary.get('refusal_rate')}"
        ),
        "",
        "## Input Hashes",
        f"- prompts_hash: `{summary.get('prompts_hash')}`",
        f"- config_hash: `{summary.get('config_hash')}`",
        f"- dialogues_hash: `{summary.get('dialogues_hash')}`",
        "",
        "## Input Freeze Provenance",
        f"- approval_enforced: `{summary.get('approval_enforced')}`",
        f"- frozen_index_path: `{summary.get('frozen_index_path')}`",
        f"- prompts_source: `{summary.get('prompts_source')}`",
        f"- prompts_version: `{summary.get('prompts_version')}`",
        f"- dialogues_source: `{summary.get('dialogues_source')}`",
        f"- dialogues_version: `{summary.get('dialogues_version')}`",
        "",
        "## Recent Rows (tail=5)",
    ]

    if not tail_rows:
        lines.append("- (no rows)")
    else:
        for row in tail_rows:
            lines.append(
                "- "
                + f"{row.get('dialogue_id')} / {row.get('condition')} / "
                + f"turn={row.get('turn_index')} / error={row.get('error_stage')}"
            )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
