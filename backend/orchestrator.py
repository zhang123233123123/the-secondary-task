from __future__ import annotations

import datetime as dt
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .frozen_registry import find_approved_version_for_file
from .input_loader import Dialogue, PromptsBundle, compute_sha256, load_dialogues, load_prompts
from .llm_client_factory import build_chat_client
from .llm_clients import (
    LLMError,
    detect_refusal,
    parse_judge_json,
)
from .output_writer import JsonlWriter, write_summary
from .report_writer import write_report
from .resume import load_resume_state
from .runs_index import write_runs_index
from .schema_validation import validate_with_simple_schema
from .runtime_config import RuntimeConfig

CONDITIONS_ORDER = ("default", "unhelpful", "cynical", "distant")
JUDGE_KEYS_1_TO_10 = (
    "harm_1_10",
    "negative_emotion_1_10",
    "inappropriate_1_10",
    "empathic_language_1_10",
)
JUDGE_KEYS_1_TO_5 = (
    "anthro_q1",
    "anthro_q2",
    "anthro_q3",
    "anthro_q4",
    "anthro_q5",
)


@dataclass
class RunStats:
    expected_rows: int = 0
    actual_rows: int = 0
    generate_errors: int = 0
    judge_errors: int = 0
    judge_parse_errors: int = 0
    refusal_count: int = 0
    truncated_count: int = 0


def _build_run_id() -> str:
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"run_{ts}_{uuid.uuid4().hex[:6]}"


def _truncate_history(
    messages: list[dict[str, str]],
    policy: str,
    max_history_messages: int,
    max_context_chars: int,
) -> tuple[list[dict[str, str]], bool]:
    if policy == "token_budget":
        # Approximate token budget using character budget to keep implementation dependency-free.
        kept = [messages[0]]
        used_chars = len(messages[0].get("content", ""))
        tail: list[dict[str, str]] = []
        for item in reversed(messages[1:]):
            item_chars = len(item.get("content", ""))
            if used_chars + item_chars > max_context_chars:
                break
            tail.append(item)
            used_chars += item_chars
        tail.reverse()
        kept.extend(tail)
        return kept, len(kept) != len(messages)

    if len(messages) <= max_history_messages:
        return messages, False
    # Keep system prompt and last N-1 exchanges/messages.
    return [messages[0], *messages[-(max_history_messages - 1) :]], True


def _timestamp_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _judge_messages(
    prompts: PromptsBundle,
    model_reply: str,
    *,
    user_text: str,
) -> list[dict[str, str]]:
    all_keys = [*JUDGE_KEYS_1_TO_10, *JUDGE_KEYS_1_TO_5]
    user_content = (
        f"User text:\n{user_text}\n\n"
        f"Rubric:\n{prompts.judge_rubric}\n\n"
        f"Model reply to evaluate:\n{model_reply}\n\n"
        f"Return only JSON with keys: {', '.join(all_keys)}."
    )
    return [
        {"role": "system", "content": prompts.judge_system},
        {"role": "user", "content": user_content},
    ]


def _call_with_retry(
    call,
    retries: int,
    backoff_seconds: float = 0.5,
):
    attempts = retries + 1
    last_error: Exception | None = None
    for index in range(attempts):
        try:
            return call()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if index == attempts - 1:
                break
            time.sleep(backoff_seconds * (2**index))
    if last_error is None:
        raise RuntimeError("Unexpected retry state")
    raise last_error


def _validate_judge_score_range(values: dict[str, int]) -> None:
    for key in JUDGE_KEYS_1_TO_10:
        value = values[key]
        if value < 1 or value > 10:
            raise ValueError(f"{key} out of range: {value}")
    for key in JUDGE_KEYS_1_TO_5:
        value = values[key]
        if value < 1 or value > 5:
            raise ValueError(f"{key} out of range: {value}")


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


def _compute_anthro_item_means_by_condition(
    results_path: Path,
) -> dict[str, dict[str, float | None]]:
    sums: dict[str, dict[str, float]] = {}
    counts: dict[str, int] = {}
    for condition in CONDITIONS_ORDER:
        sums[condition] = {key: 0.0 for key in JUDGE_KEYS_1_TO_5}
        counts[condition] = 0

    if not results_path.exists():
        return {
            condition: {key: None for key in JUDGE_KEYS_1_TO_5}
            for condition in CONDITIONS_ORDER
        }

    with results_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            condition = str(row.get("condition", ""))
            if condition not in sums:
                continue
            if row.get("error_stage") is not None:
                continue
            values = [row.get(key) for key in JUDGE_KEYS_1_TO_5]
            if not all(isinstance(value, int) for value in values):
                continue
            for key, value in zip(JUDGE_KEYS_1_TO_5, values):
                sums[condition][key] += float(value)
            counts[condition] += 1

    means: dict[str, dict[str, float | None]] = {}
    for condition in CONDITIONS_ORDER:
        condition_count = counts[condition]
        means[condition] = {}
        for key in JUDGE_KEYS_1_TO_5:
            if condition_count == 0:
                means[condition][key] = None
            else:
                means[condition][key] = sums[condition][key] / condition_count
    return means


def _write_validation_log(
    *,
    output_dir: str | Path,
    run_id: str,
    command: str,
    summary: dict[str, Any],
    results_path: Path,
) -> Path:
    output_path = Path(output_dir) / f"validation_{run_id}.log"
    tail_rows = _read_tail_rows(results_path, limit=5)
    lines = [
        f"validation_run_id={run_id}",
        f"generated_at_utc={_timestamp_utc()}",
        f"command={command}",
        f"actual_rows={summary.get('actual_rows')}",
        f"error_rate={summary.get('error_rate')}",
        f"refusal_rate={summary.get('refusal_rate')}",
        "recent_rows_tail=5",
    ]
    for row in tail_rows:
        lines.append(
            f"- {row.get('dialogue_id')} / {row.get('condition')} / "
            f"turn={row.get('turn_index')} / error={row.get('error_stage')}"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def run_experiment(
    config: RuntimeConfig,
    config_path: str,
    dry_run: bool,
    run_id: str | None = None,
) -> dict[str, Any]:
    actual_run_id = run_id or _build_run_id()
    config_file = Path(config_path).resolve()
    config_dir = config_file.parent
    prompts_file = Path(config.prompts_path)
    dialogues_file = Path(config.dialogues_path)
    index_file = Path(config.frozen_index_path)
    if not prompts_file.is_absolute():
        prompts_file = (config_dir / prompts_file).resolve()
    if not dialogues_file.is_absolute():
        dialogues_file = (config_dir / dialogues_file).resolve()
    if not index_file.is_absolute():
        index_file = (config_dir / index_file).resolve()

    prompts_version = find_approved_version_for_file(
        index_path=index_file,
        kind="prompts",
        file_path=prompts_file,
    )
    dialogues_version = find_approved_version_for_file(
        index_path=index_file,
        kind="dialogues",
        file_path=dialogues_file,
    )
    if config.require_approved_prompts and prompts_version is None:
        raise RuntimeError(
            f"prompts_path is not an approved frozen version: {prompts_file}. "
            "Run prepare + approve-prompts + use-frozen first."
        )
    if config.require_approved_dialogues and dialogues_version is None:
        raise RuntimeError(
            f"dialogues_path is not an approved frozen version: {dialogues_file}. "
            "Run prepare + approve-dialogues + use-frozen first."
        )
    prompts_source = "frozen" if prompts_version is not None else "manual"
    dialogues_source = "frozen" if dialogues_version is not None else "manual"
    approval_enforced = config.require_approved_prompts or config.require_approved_dialogues

    dialogues: list[Dialogue] = load_dialogues(
        dialogues_file,
        compatibility_mode=config.input_compatibility_mode,
    )
    prompts = load_prompts(prompts_file)
    if dry_run:
        dialogues = dialogues[:5]

    hashes = {
        "prompts_hash": compute_sha256(prompts_file),
        "config_hash": compute_sha256(config_file),
        "dialogues_hash": compute_sha256(dialogues_file),
    }
    stats = RunStats()

    writer = JsonlWriter(config.output_dir, actual_run_id, config.flush_policy)
    resume_state = load_resume_state(writer.results_path)
    llm3_client = build_chat_client(config.llm3)
    llm4_client = build_chat_client(config.llm4)
    abort_reason: str | None = None

    try:
        for dialogue in dialogues:
            for condition in CONDITIONS_ORDER:
                system_prompt = prompts.conditions[condition]
                combo_key = (dialogue.dialogue_id, condition)
                combo_state = resume_state.combo_states.get(combo_key)
                if (
                    config.resume_strategy == "skip"
                    and combo_state is not None
                    and combo_state.processed_turns
                ):
                    continue

                history: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
                processed_turns: set[int] = set()
                if config.resume_strategy == "reconstruct" and combo_state is not None:
                    history.extend(combo_state.history)
                    processed_turns = set(combo_state.processed_turns)

                max_turns = min(len(dialogue.turns), config.max_turns)
                stats.expected_rows += max_turns

                for turn_index in range(1, max_turns + 1):
                    if turn_index in processed_turns:
                        continue

                    user_text = dialogue.turns[turn_index - 1].text
                    history_with_current_turn = [*history, {"role": "user", "content": user_text}]
                    prompt_messages, context_truncated = _truncate_history(
                        history_with_current_turn,
                        policy=config.truncation_policy,
                        max_history_messages=config.max_history_messages,
                        max_context_chars=config.max_context_chars,
                    )

                    model_reply = ""
                    gen_latency_ms: int | None = None
                    judge_latency_ms: int | None = None
                    harm_1_10: int | None = None
                    negative_emotion_1_10: int | None = None
                    inappropriate_1_10: int | None = None
                    empathic_language_1_10: int | None = None
                    anthro_q1: int | None = None
                    anthro_q2: int | None = None
                    anthro_q3: int | None = None
                    anthro_q4: int | None = None
                    anthro_q5: int | None = None
                    error_stage: str | None = None
                    error_message: str | None = None
                    judge_raw: str | None = None
                    refusal_detected = False

                    try:
                        generate_result = _call_with_retry(
                            lambda: llm3_client.chat(prompt_messages, config.timeout_seconds),
                            retries=config.retries,
                        )
                        model_reply = generate_result.text
                        gen_latency_ms = generate_result.latency_ms
                        refusal_detected = detect_refusal(model_reply)
                    except Exception as exc:  # noqa: BLE001
                        error_stage = "generate"
                        error_message = str(exc)
                        stats.generate_errors += 1

                    if error_stage is None:
                        try:
                            judge_result = _call_with_retry(
                                lambda: llm4_client.chat(
                                    _judge_messages(
                                        prompts,
                                        model_reply,
                                        user_text=user_text,
                                    ),
                                    config.timeout_seconds,
                                ),
                                retries=config.retries,
                            )
                            judge_latency_ms = judge_result.latency_ms
                            judge_raw = judge_result.text
                            parsed = parse_judge_json(judge_result.text)
                            validate_with_simple_schema(parsed, prompts.judge_schema)
                            scores = {
                                "harm_1_10": int(parsed.get("harm_1_10")),
                                "negative_emotion_1_10": int(parsed.get("negative_emotion_1_10")),
                                "inappropriate_1_10": int(parsed.get("inappropriate_1_10")),
                                "empathic_language_1_10": int(parsed.get("empathic_language_1_10")),
                                "anthro_q1": int(parsed.get("anthro_q1")),
                                "anthro_q2": int(parsed.get("anthro_q2")),
                                "anthro_q3": int(parsed.get("anthro_q3")),
                                "anthro_q4": int(parsed.get("anthro_q4")),
                                "anthro_q5": int(parsed.get("anthro_q5")),
                            }
                            _validate_judge_score_range(scores)
                            harm_1_10 = scores["harm_1_10"]
                            negative_emotion_1_10 = scores["negative_emotion_1_10"]
                            inappropriate_1_10 = scores["inappropriate_1_10"]
                            empathic_language_1_10 = scores["empathic_language_1_10"]
                            anthro_q1 = scores["anthro_q1"]
                            anthro_q2 = scores["anthro_q2"]
                            anthro_q3 = scores["anthro_q3"]
                            anthro_q4 = scores["anthro_q4"]
                            anthro_q5 = scores["anthro_q5"]
                        except (TypeError, ValueError) as exc:
                            error_stage = "judge_parse"
                            error_message = str(exc)
                            harm_1_10 = None
                            negative_emotion_1_10 = None
                            inappropriate_1_10 = None
                            empathic_language_1_10 = None
                            anthro_q1 = None
                            anthro_q2 = None
                            anthro_q3 = None
                            anthro_q4 = None
                            anthro_q5 = None
                            stats.judge_parse_errors += 1
                        except (LLMError, RuntimeError) as exc:
                            error_stage = "judge"
                            error_message = str(exc)
                            stats.judge_errors += 1
                        except Exception as exc:  # noqa: BLE001
                            error_stage = "judge"
                            error_message = str(exc)
                            stats.judge_errors += 1

                    row = {
                        "run_id": actual_run_id,
                        "dialogue_id": dialogue.dialogue_id,
                        "domain": dialogue.domain,
                        "condition": condition,
                        "turn_index": turn_index,
                        "timestamp_utc": _timestamp_utc(),
                        "user_text": user_text,
                        "model_reply": model_reply,
                        "harm_1_10": harm_1_10,
                        "negative_emotion_1_10": negative_emotion_1_10,
                        "inappropriate_1_10": inappropriate_1_10,
                        "empathic_language_1_10": empathic_language_1_10,
                        "anthro_q1": anthro_q1,
                        "anthro_q2": anthro_q2,
                        "anthro_q3": anthro_q3,
                        "anthro_q4": anthro_q4,
                        "anthro_q5": anthro_q5,
                        "error_stage": error_stage,
                        "error_message": error_message,
                        "refusal_detected": refusal_detected,
                        "context_truncated": context_truncated,
                        "gen_latency_ms": gen_latency_ms,
                        "judge_latency_ms": judge_latency_ms,
                        "llm3_model": config.llm3.model,
                        "llm3_provider": config.llm3.provider,
                        "llm4_provider": config.llm4.provider,
                        "llm3_params": {
                            "temperature": config.llm3.temperature,
                            "top_p": config.llm3.top_p,
                            "seed": config.llm3.seed,
                            "max_new_tokens": config.llm3.max_new_tokens,
                            "load_in_4bit": config.llm3.load_in_4bit,
                        },
                        "resume_strategy": config.resume_strategy,
                        "input_schema_variant": dialogue.input_schema_variant,
                        "prompts_version": prompts_version,
                        "dialogues_version": dialogues_version,
                        "judge_raw": judge_raw,
                        **hashes,
                    }
                    writer.write(row)
                    stats.actual_rows += 1

                    if context_truncated:
                        stats.truncated_count += 1
                    if refusal_detected:
                        stats.refusal_count += 1

                    # Keep history clean from failed generate turns.
                    if error_stage != "generate":
                        history = history_with_current_turn
                        history.append({"role": "assistant", "content": model_reply})

                    if error_stage == "generate" and config.abort_on_error:
                        raise RuntimeError(error_message or "Generation failed")
                    if error_stage in {"judge", "judge_parse"} and config.abort_on_error:
                        raise RuntimeError(error_message or "Judge failed")
    except RuntimeError as exc:
        abort_reason = str(exc)

    finally:
        writer.close()

    final_resume_state = load_resume_state(writer.results_path)
    total_rows = final_resume_state.existing_rows
    error_rows = final_resume_state.error_rows
    summary = {
        "run_id": actual_run_id,
        "expected_rows": stats.expected_rows,
        "actual_rows": total_rows,
        "new_rows_written": writer.rows_written,
        "flush_policy_requested": writer.requested_flush_policy,
        "flush_policy_effective": writer.effective_flush_policy,
        "error_rows": error_rows,
        "error_rate": (error_rows / total_rows) if total_rows else 0.0,
        "generate_errors": final_resume_state.generate_errors,
        "judge_errors": final_resume_state.judge_errors,
        "judge_parse_errors": final_resume_state.judge_parse_errors,
        "new_generate_errors": stats.generate_errors,
        "new_judge_errors": stats.judge_errors,
        "new_judge_parse_errors": stats.judge_parse_errors,
        "truncated_count": final_resume_state.truncated_count,
        "refusal_count": final_resume_state.refusal_count,
        "refusal_rate": (final_resume_state.refusal_count / total_rows) if total_rows else 0.0,
        "prompts_version": prompts_version,
        "dialogues_version": dialogues_version,
        "prompts_source": prompts_source,
        "dialogues_source": dialogues_source,
        "approval_enforced": approval_enforced,
        "frozen_index_path": str(index_file),
        "dry_run": bool(dry_run),
        "validation_log_file": None,
        "aborted": abort_reason is not None,
        "abort_reason": abort_reason,
        "anthro_item_means_by_condition": _compute_anthro_item_means_by_condition(
            writer.results_path
        ),
        **hashes,
    }
    if dry_run:
        command = f"python control_agent.py run --config {config_path} --dry_run"
        validation_path = _write_validation_log(
            output_dir=config.output_dir,
            run_id=actual_run_id,
            command=command,
            summary=summary,
            results_path=writer.results_path,
        )
        summary["validation_log_file"] = validation_path.name
    summary_path = write_summary(config.output_dir, actual_run_id, summary)
    report_path = write_report(
        config.output_dir,
        actual_run_id,
        summary,
        dry_run=dry_run,
        results_path=writer.results_path,
    )
    write_runs_index(config.output_dir)
    if abort_reason is not None:
        raise RuntimeError(abort_reason)
    return {
        "run_id": actual_run_id,
        "summary_path": str(summary_path),
        "report_path": str(report_path),
        "results_path": str(writer.results_path),
        "summary": summary,
    }
