import pytest

from backend.prepare_validation import validate_prepared_dialogues


def test_prepare_validation_count_mismatch():
    payload = [
        {
            "dialogue_id": "D1",
            "domain": "creative",
            "turns": [{"role": "user", "text": "a"}],
        }
    ]
    with pytest.raises(ValueError, match="dialogue_count_mismatch"):
        validate_prepared_dialogues(
            payload,
            expected_count=2,
            expected_turns=1,
            expected_distribution={
                "creative": 0.25,
                "finance": 0.25,
                "mental_health": 0.25,
                "medicine": 0.25,
            },
        )


def test_prepare_validation_distribution_violation():
    payload = []
    for i in range(20):
        payload.append(
            {
                "dialogue_id": f"D{i}",
                "domain": "creative",
                "turns": [{"role": "user", "text": "a"}],
            }
        )
    with pytest.raises(ValueError, match="domain_distribution_mismatch"):
        validate_prepared_dialogues(
            payload,
            expected_count=20,
            expected_turns=1,
            expected_distribution={
                "creative": 0.25,
                "finance": 0.25,
                "mental_health": 0.25,
                "medicine": 0.25,
            },
        )


def test_prepare_validation_turn_count_violation():
    payload = [
        {
            "dialogue_id": "D1",
            "domain": "creative",
            "turns": [
                {"role": "user", "text": "a"},
                {"role": "user", "text": "b"},
            ],
        }
    ]
    with pytest.raises(ValueError, match="turn_count_mismatch"):
        validate_prepared_dialogues(
            payload,
            expected_count=1,
            expected_turns=3,
            expected_distribution={
                "creative": 1.0,
                "finance": 0.0,
                "mental_health": 0.0,
                "medicine": 0.0,
            },
        )

