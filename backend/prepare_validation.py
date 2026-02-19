from __future__ import annotations

from typing import Any

from .input_loader import VALID_DOMAINS


def _normalize_distribution(raw: dict[str, float]) -> dict[str, float]:
    keys = {str(k) for k in raw.keys()}
    missing = VALID_DOMAINS - keys
    unexpected = keys - VALID_DOMAINS
    if missing:
        raise ValueError(f"prepare_domain_distribution missing domains: {sorted(missing)}")
    if unexpected:
        raise ValueError(f"prepare_domain_distribution has unexpected domains: {sorted(unexpected)}")

    normalized_raw = {str(k): float(v) for k, v in raw.items()}
    total = sum(normalized_raw.values())
    if total <= 0:
        raise ValueError("prepare_domain_distribution total weight must be > 0")
    return {k: normalized_raw[k] / total for k in VALID_DOMAINS}


def validate_prepared_dialogues(
    dialogues_payload: list[dict[str, Any]],
    *,
    expected_count: int,
    expected_turns: int,
    expected_distribution: dict[str, float],
) -> None:
    if expected_count <= 0:
        raise ValueError("prepare_dialogue_count must be > 0")
    if expected_turns <= 0:
        raise ValueError("prepare_dialogue_turns must be > 0")

    actual_count = len(dialogues_payload)
    if actual_count != expected_count:
        raise ValueError(
            f"dialogue_count_mismatch: expected={expected_count}, actual={actual_count}"
        )

    normalized_distribution = _normalize_distribution(expected_distribution)
    domain_counts = {domain: 0 for domain in VALID_DOMAINS}

    for idx, item in enumerate(dialogues_payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"dialogue #{idx} must be an object")
        dialogue_id = str(item.get("dialogue_id", f"index_{idx}"))
        domain = item.get("domain")
        if not isinstance(domain, str) or domain not in VALID_DOMAINS:
            raise ValueError(f"dialogue {dialogue_id}: invalid domain {domain!r}")

        turns = item.get("turns")
        if not isinstance(turns, list):
            raise ValueError(f"dialogue {dialogue_id}: turns must be an array")
        if len(turns) != expected_turns:
            raise ValueError(
                f"dialogue {dialogue_id}: turn_count_mismatch expected={expected_turns}, actual={len(turns)}"
            )

        for turn_index, turn in enumerate(turns, start=1):
            if not isinstance(turn, dict):
                raise ValueError(f"dialogue {dialogue_id}: turn#{turn_index} must be an object")
            role = turn.get("role")
            text = turn.get("text")
            if role != "user":
                raise ValueError(
                    f"dialogue {dialogue_id}: turn#{turn_index} role must be 'user', got {role!r}"
                )
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"dialogue {dialogue_id}: turn#{turn_index} text must be non-empty")

        domain_counts[domain] += 1

    tolerance = max(1, round(expected_count * 0.02))
    violations: list[str] = []
    for domain in sorted(VALID_DOMAINS):
        expected_domain_count = expected_count * normalized_distribution[domain]
        actual_domain_count = domain_counts[domain]
        delta = abs(actual_domain_count - expected_domain_count)
        if delta > tolerance:
            violations.append(
                (
                    f"{domain}: expected~{expected_domain_count:.2f}, actual={actual_domain_count}, "
                    f"delta={delta:.2f}, tolerance={tolerance}"
                )
            )
    if violations:
        raise ValueError("domain_distribution_mismatch: " + "; ".join(violations))

