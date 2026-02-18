from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RuntimeConfig:
    max_turns: int
    output_dir: str
    resume_strategy: str
    abort_on_error: bool
    retries: int
    timeout_seconds: float
    truncation_policy: str
    flush_policy: str


def load_config(path: str | Path) -> RuntimeConfig:
    config_path = Path(path)
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    return RuntimeConfig(
        max_turns=int(raw.get("max_turns", 10)),
        output_dir=str(raw.get("output_dir", "output")),
        resume_strategy=str(raw.get("resume_strategy", "reconstruct")),
        abort_on_error=bool(raw.get("abort_on_error", False)),
        retries=int(raw.get("retries", 3)),
        timeout_seconds=float(raw.get("timeout", 30)),
        truncation_policy=str(raw.get("truncation_policy", "sliding_window")),
        flush_policy=str(raw.get("flush_policy", "per_turn")),
    )
