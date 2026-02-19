from __future__ import annotations

from typing import Any

from jsonschema import ValidationError, validate


def validate_with_simple_schema(data: Any, schema: dict[str, Any]) -> None:
    try:
        validate(instance=data, schema=schema)
    except ValidationError as exc:
        path = ".".join(str(part) for part in exc.path)
        if path:
            raise ValueError(f"Judge output schema violation at {path}: {exc.message}") from exc
        raise ValueError(f"Judge output schema violation: {exc.message}") from exc
