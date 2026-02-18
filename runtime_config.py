from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class LLMConfig:
    provider: str
    model: str
    api_key_env: str
    base_url: str
    temperature: float
    top_p: float
    seed: int | None


@dataclass
class RuntimeConfig:
    dialogues_path: str
    prompts_path: str
    max_turns: int
    output_dir: str
    resume_strategy: str
    abort_on_error: bool
    retries: int
    timeout_seconds: float
    truncation_policy: str
    flush_policy: str
    input_compatibility_mode: bool
    max_history_messages: int
    llm3: LLMConfig
    llm4: LLMConfig


def _load_llm_config(raw: dict[str, Any], default_model: str) -> LLMConfig:
    return LLMConfig(
        provider=str(raw.get("provider", "deepseek")),
        model=str(raw.get("model", default_model)),
        api_key_env=str(raw.get("api_key_env", "DEEPSEEK_API_KEY")),
        base_url=str(raw.get("base_url", "https://api.deepseek.com/v1")),
        temperature=float(raw.get("temperature", 0.7)),
        top_p=float(raw.get("top_p", 1.0)),
        seed=raw.get("seed"),
    )


def load_config(path: str | Path) -> RuntimeConfig:
    config_path = Path(path)
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    return RuntimeConfig(
        dialogues_path=str(raw.get("dialogues_path", "dialogues.jsonl")),
        prompts_path=str(raw.get("prompts_path", "prompts.json")),
        max_turns=int(raw.get("max_turns", 10)),
        output_dir=str(raw.get("output_dir", "output")),
        resume_strategy=str(raw.get("resume_strategy", "reconstruct")),
        abort_on_error=bool(raw.get("abort_on_error", False)),
        retries=int(raw.get("retries", 3)),
        timeout_seconds=float(raw.get("timeout", 30)),
        truncation_policy=str(raw.get("truncation_policy", "sliding_window")),
        flush_policy=str(raw.get("flush_policy", "per_turn")),
        input_compatibility_mode=bool(raw.get("input_compatibility_mode", False)),
        max_history_messages=int(raw.get("max_history_messages", 20)),
        llm3=_load_llm_config(raw.get("llm3", {}), default_model="deepseek-chat"),
        llm4=_load_llm_config(raw.get("llm4", {}), default_model="deepseek-chat"),
    )
