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
    max_new_tokens: int = 256
    device_map: str = "auto"
    load_in_4bit: bool = False
    torch_dtype: str = "auto"


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
    max_context_chars: int
    frozen_index_path: str
    require_approved_prompts: bool
    require_approved_dialogues: bool
    prepare_dialogue_count: int
    prepare_dialogue_min_turns: int
    prepare_dialogue_turns: int
    prepare_domain_distribution: dict[str, float]
    llm1: LLMConfig | None
    llm2: LLMConfig
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
        max_new_tokens=int(raw.get("max_new_tokens", 256)),
        device_map=str(raw.get("device_map", "auto")),
        load_in_4bit=bool(raw.get("load_in_4bit", False)),
        torch_dtype=str(raw.get("torch_dtype", "auto")),
    )


def load_config(path: str | Path) -> RuntimeConfig:
    config_path = Path(path)
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    prepare_distribution = raw.get("prepare_domain_distribution")
    if not isinstance(prepare_distribution, dict):
        prepare_distribution = {
            "creative": 0.25,
            "finance": 0.25,
            "mental_health": 0.25,
            "medicine": 0.25,
        }
    normalized_distribution = {
        str(key): float(value) for key, value in prepare_distribution.items()
    }

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
        max_context_chars=int(raw.get("max_context_chars", 8000)),
        frozen_index_path=str(raw.get("frozen_index_path", "frozen_inputs/index.json")),
        require_approved_prompts=bool(raw.get("require_approved_prompts", True)),
        require_approved_dialogues=bool(raw.get("require_approved_dialogues", True)),
        prepare_dialogue_count=int(raw.get("prepare_dialogue_count", 2000)),
        prepare_dialogue_min_turns=int(raw.get("prepare_dialogue_min_turns", 1)),
        prepare_dialogue_turns=int(raw.get("prepare_dialogue_turns", 6)),
        prepare_domain_distribution=normalized_distribution,
        llm1=_load_llm_config(raw["llm1"], default_model="deepseek-chat") if "llm1" in raw else None,
        llm2=_load_llm_config(raw.get("llm2", {}), default_model="deepseek-chat"),
        llm3=_load_llm_config(raw.get("llm3", {}), default_model="deepseek-chat"),
        llm4=_load_llm_config(raw.get("llm4", {}), default_model="deepseek-chat"),
    )
