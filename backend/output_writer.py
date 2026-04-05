from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any


class JsonlWriter:
    def __init__(self, output_dir: str | Path, run_id: str, flush_policy: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_path = self.output_dir / f"results_{run_id}.jsonl"
        self.requested_flush_policy = flush_policy
        # Runtime always guarantees per-turn durability for resumability.
        self.effective_flush_policy = "per_turn"
        self._fh = self.results_path.open("a", encoding="utf-8")
        self.rows_written = 0
        self._lock = threading.Lock()

    def write(self, row: dict[str, Any]) -> None:
        with self._lock:
            self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.rows_written += 1
            self._fh.flush()

    def close(self) -> None:
        self._fh.flush()
        self._fh.close()


def write_summary(output_dir: str | Path, run_id: str, summary: dict[str, Any]) -> Path:
    output_path = Path(output_dir) / f"run_summary_{run_id}.json"
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path
