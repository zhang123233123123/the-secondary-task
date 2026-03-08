import pytest

from backend.schema_validation import validate_with_simple_schema


def test_schema_validation_accepts_valid_payload():
    schema = {
        "type": "object",
        "required": ["harm_1_10", "negative_emotion_1_10", "anthro_q1"],
        "properties": {
            "harm_1_10": {"type": "integer", "minimum": 1, "maximum": 10},
            "negative_emotion_1_10": {"type": "integer", "minimum": 1, "maximum": 10},
            "anthro_q1": {"type": "integer", "minimum": 1, "maximum": 5},
        },
    }
    payload = {"harm_1_10": 2, "negative_emotion_1_10": 3, "anthro_q1": 1}
    validate_with_simple_schema(payload, schema)


def test_schema_validation_rejects_missing_required_field():
    schema = {
        "type": "object",
        "required": ["harm_1_10", "negative_emotion_1_10"],
        "properties": {
            "harm_1_10": {"type": "integer"},
            "negative_emotion_1_10": {"type": "integer"},
        },
    }
    with pytest.raises(ValueError, match="schema violation"):
        validate_with_simple_schema({"harm_1_10": 2}, schema)


def test_schema_validation_rejects_additional_properties_when_forbidden():
    schema = {
        "type": "object",
        "required": ["harm_1_10", "negative_emotion_1_10", "anthro_q1"],
        "properties": {
            "harm_1_10": {"type": "integer", "minimum": 1, "maximum": 10},
            "negative_emotion_1_10": {"type": "integer", "minimum": 1, "maximum": 10},
            "anthro_q1": {"type": "integer", "minimum": 1, "maximum": 5},
        },
        "additionalProperties": False,
    }
    with pytest.raises(ValueError, match="schema violation"):
        validate_with_simple_schema(
            {"harm_1_10": 2, "negative_emotion_1_10": 3, "anthro_q1": 1, "unexpected": 7},
            schema,
        )
