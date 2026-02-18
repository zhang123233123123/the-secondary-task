import pytest

from backend.schema_validation import validate_with_simple_schema


def test_schema_validation_accepts_valid_payload():
    schema = {
        "type": "object",
        "required": ["harm", "emotion", "anthro"],
        "properties": {
            "harm": {"type": "integer", "minimum": 1, "maximum": 5},
            "emotion": {"type": "integer", "minimum": 1, "maximum": 5},
            "anthro": {"type": "integer", "minimum": 1, "maximum": 5},
        },
    }
    payload = {"harm": 2, "emotion": 3, "anthro": 1}
    validate_with_simple_schema(payload, schema)


def test_schema_validation_rejects_missing_required_field():
    schema = {
        "type": "object",
        "required": ["harm", "emotion"],
        "properties": {"harm": {"type": "integer"}, "emotion": {"type": "integer"}},
    }
    with pytest.raises(ValueError, match="missing required field"):
        validate_with_simple_schema({"harm": 2}, schema)
