from __future__ import annotations

from typing import Any


def validate_with_simple_schema(data: Any, schema: dict[str, Any]) -> None:
    expected_type = schema.get("type")
    if expected_type == "object":
        if not isinstance(data, dict):
            raise ValueError("Judge output must be an object")
        required = schema.get("required", [])
        for key in required:
            if key not in data:
                raise ValueError(f"Judge output missing required field: {key}")
        properties = schema.get("properties", {})
        for key, prop in properties.items():
            if key not in data:
                continue
            value = data[key]
            prop_type = prop.get("type")
            if prop_type == "integer":
                if not isinstance(value, int):
                    raise ValueError(f"Judge field must be integer: {key}")
                minimum = prop.get("minimum")
                maximum = prop.get("maximum")
                if minimum is not None and value < minimum:
                    raise ValueError(f"Judge field below minimum: {key}")
                if maximum is not None and value > maximum:
                    raise ValueError(f"Judge field above maximum: {key}")
