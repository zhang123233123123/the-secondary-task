from __future__ import annotations

from pathlib import Path


def _unquote(value: str) -> str:
    trimmed = value.strip()
    if len(trimmed) >= 2 and trimmed[0] == trimmed[-1] and trimmed[0] in {'"', "'"}:
        inner = trimmed[1:-1]
        inner = inner.replace('\\"', '"').replace("\\'", "'").replace("\\\\", "\\")
        return inner
    return trimmed


def _escape_for_double_quotes(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def read_local_env(path: str | Path) -> dict[str, str]:
    env_path = Path(path)
    if not env_path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        values[key] = _unquote(raw_value)
    return values


def upsert_local_env(path: str | Path, key: str, value: str) -> Path:
    env_path = Path(path)
    env_path.parent.mkdir(parents=True, exist_ok=True)
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []

    rendered = f'{key}="{_escape_for_double_quotes(value)}"'
    updated = False
    output_lines: list[str] = []

    for raw_line in lines:
        stripped = raw_line.strip()
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].strip()
        if "=" in stripped:
            existing_key = stripped.split("=", 1)[0].strip()
            if existing_key == key:
                output_lines.append(rendered)
                updated = True
                continue
        output_lines.append(raw_line)

    if not updated:
        output_lines.append(rendered)

    env_path.write_text("\n".join(output_lines).rstrip() + "\n", encoding="utf-8")
    return env_path


def mask_secret(value: str | None) -> str | None:
    if value is None:
        return None
    secret = value.strip()
    if not secret:
        return None
    if len(secret) <= 8:
        return "*" * len(secret)
    return secret[:4] + "*" * (len(secret) - 8) + secret[-4:]


def resolve_secret(key: str, env_path: str | Path) -> str | None:
    from os import environ

    direct = environ.get(key)
    if direct:
        return direct
    local_values = read_local_env(env_path)
    return local_values.get(key)
