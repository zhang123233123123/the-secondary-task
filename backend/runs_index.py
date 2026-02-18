from __future__ import annotations

import datetime as dt
import json
from pathlib import Path


def _iso_utc_from_mtime(path: Path) -> str:
    return dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc).isoformat()


def build_runs_index(output_dir: str | Path) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    runs = []
    for summary_file in sorted(output_path.glob("run_summary_*.json")):
        stem = summary_file.stem  # run_summary_<run_id>
        run_id = stem.removeprefix("run_summary_")
        results_name = f"results_{run_id}.jsonl"
        results_file = output_path / results_name
        runs.append(
            {
                "run_id": run_id,
                "created_at_utc": _iso_utc_from_mtime(summary_file),
                "summary_file": summary_file.name,
                "results_file": results_name,
                "results_exists": results_file.exists(),
            }
        )

    runs.sort(key=lambda item: item["created_at_utc"], reverse=True)
    latest_run_id = runs[0]["run_id"] if runs else None
    return {"latest_run_id": latest_run_id, "runs": runs}


def write_runs_index(output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    index = build_runs_index(output_path)
    index_path = output_path / "runs_index.json"
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return index_path
