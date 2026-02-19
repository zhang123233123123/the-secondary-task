from __future__ import annotations

import datetime as dt
import json
import shutil
from pathlib import Path
from typing import Any, Literal

import yaml

from .input_loader import compute_sha256, load_dialogues, load_prompts

FrozenKind = Literal["prompts", "dialogues"]


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _default_index() -> dict[str, Any]:
    return {
        "active": {"prompts_version": None, "dialogues_version": None},
        "prompts_versions": [],
        "dialogues_versions": [],
    }


def load_frozen_index(index_path: str | Path) -> dict[str, Any]:
    path = Path(index_path)
    if not path.exists():
        return _default_index()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return _default_index()
    raw.setdefault("active", {"prompts_version": None, "dialogues_version": None})
    raw.setdefault("prompts_versions", [])
    raw.setdefault("dialogues_versions", [])
    return raw


def save_frozen_index(index_path: str | Path, index_data: dict[str, Any]) -> Path:
    path = Path(index_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index_data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def init_frozen_layout(index_path: str | Path) -> None:
    path = Path(index_path)
    root = path.parent
    (root / "prompts").mkdir(parents=True, exist_ok=True)
    (root / "dialogues").mkdir(parents=True, exist_ok=True)
    (root / "candidates").mkdir(parents=True, exist_ok=True)
    (root / "reviews").mkdir(parents=True, exist_ok=True)
    if not path.exists():
        save_frozen_index(path, _default_index())


def _project_root_from_index(index_path: Path) -> Path:
    return index_path.parent.parent


def _entries_key(kind: FrozenKind) -> str:
    return "prompts_versions" if kind == "prompts" else "dialogues_versions"


def _review_file_name(kind: FrozenKind, version: str) -> str:
    return f"review_{kind}_{version}.json"


def _frozen_file_name(kind: FrozenKind, version: str) -> str:
    suffix = "json" if kind == "prompts" else "jsonl"
    return f"{kind}_{version}.{suffix}"


def _validate_candidate(kind: FrozenKind, candidate_path: Path) -> None:
    if kind == "prompts":
        load_prompts(candidate_path)
        return
    load_dialogues(candidate_path, compatibility_mode=False)


def _find_entry(index_data: dict[str, Any], kind: FrozenKind, version: str) -> dict[str, Any] | None:
    entries = index_data.get(_entries_key(kind), [])
    for item in entries:
        if isinstance(item, dict) and item.get("version") == version:
            return item
    return None


def approve_candidate(
    *,
    index_path: str | Path,
    kind: FrozenKind,
    candidate_path: str | Path,
    version: str,
    reviewer: str,
    note: str | None = None,
) -> dict[str, Any]:
    index_file = Path(index_path)
    init_frozen_layout(index_file)
    index_data = load_frozen_index(index_file)
    if _find_entry(index_data, kind, version) is not None:
        raise ValueError(f"{kind} version already exists: {version}")

    candidate = Path(candidate_path).resolve()
    if not candidate.exists():
        raise ValueError(f"Candidate file not found: {candidate}")
    _validate_candidate(kind, candidate)

    kind_dir = index_file.parent / kind
    kind_dir.mkdir(parents=True, exist_ok=True)
    frozen_file = kind_dir / _frozen_file_name(kind, version)
    shutil.copyfile(candidate, frozen_file)
    frozen_hash = compute_sha256(frozen_file)
    candidate_hash = compute_sha256(candidate)
    approved_at = _utc_now()
    project_root = _project_root_from_index(index_file)

    entry = {
        "version": version,
        "file": str(frozen_file.resolve().relative_to(project_root)),
        "sha256": frozen_hash,
        "approved_by": reviewer,
        "approved_at_utc": approved_at,
        "review_note": note,
        "candidate_file": str(candidate.relative_to(project_root))
        if project_root in candidate.parents
        else str(candidate),
        "candidate_sha256": candidate_hash,
    }

    entries_key = _entries_key(kind)
    index_data[entries_key].append(entry)
    save_frozen_index(index_file, index_data)

    review_payload = {
        "kind": kind,
        "version": version,
        "reviewer": reviewer,
        "review_note": note,
        "candidate_file": entry["candidate_file"],
        "candidate_sha256": candidate_hash,
        "frozen_file": entry["file"],
        "frozen_sha256": frozen_hash,
        "approved_at_utc": approved_at,
    }
    review_path = index_file.parent / "reviews" / _review_file_name(kind, version)
    review_path.write_text(json.dumps(review_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return entry


def set_active_versions(
    *,
    index_path: str | Path,
    prompts_version: str | None = None,
    dialogues_version: str | None = None,
) -> dict[str, Any]:
    index_file = Path(index_path)
    init_frozen_layout(index_file)
    index_data = load_frozen_index(index_file)

    if prompts_version is not None and _find_entry(index_data, "prompts", prompts_version) is None:
        raise ValueError(f"Unknown prompts version: {prompts_version}")
    if dialogues_version is not None and _find_entry(index_data, "dialogues", dialogues_version) is None:
        raise ValueError(f"Unknown dialogues version: {dialogues_version}")

    active = index_data.setdefault("active", {})
    if prompts_version is not None:
        active["prompts_version"] = prompts_version
    if dialogues_version is not None:
        active["dialogues_version"] = dialogues_version

    save_frozen_index(index_file, index_data)
    return active


def resolve_frozen_file(
    *,
    index_path: str | Path,
    kind: FrozenKind,
    version: str,
) -> Path:
    index_file = Path(index_path)
    index_data = load_frozen_index(index_file)
    entry = _find_entry(index_data, kind, version)
    if entry is None:
        raise ValueError(f"Unknown {kind} version: {version}")
    file_path = entry.get("file")
    if not isinstance(file_path, str) or not file_path:
        raise ValueError(f"Invalid file path for {kind} version: {version}")
    project_root = _project_root_from_index(index_file)
    return (project_root / file_path).resolve()


def find_approved_version_for_file(
    *,
    index_path: str | Path,
    kind: FrozenKind,
    file_path: str | Path,
) -> str | None:
    index_file = Path(index_path)
    index_data = load_frozen_index(index_file)
    target = Path(file_path).resolve()
    project_root = _project_root_from_index(index_file)
    for item in index_data.get(_entries_key(kind), []):
        if not isinstance(item, dict):
            continue
        rel = item.get("file")
        if not isinstance(rel, str) or not rel:
            continue
        candidate = (project_root / rel).resolve()
        if candidate == target:
            version = item.get("version")
            return str(version) if isinstance(version, str) else None
    return None


def apply_versions_to_config(
    *,
    config_path: str | Path,
    index_path: str | Path,
    prompts_version: str,
    dialogues_version: str,
) -> dict[str, str]:
    config_file = Path(config_path)
    raw = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("config yaml must be an object")

    prompts_path = resolve_frozen_file(
        index_path=index_path,
        kind="prompts",
        version=prompts_version,
    )
    dialogues_path = resolve_frozen_file(
        index_path=index_path,
        kind="dialogues",
        version=dialogues_version,
    )
    set_active_versions(
        index_path=index_path,
        prompts_version=prompts_version,
        dialogues_version=dialogues_version,
    )

    config_dir = config_file.resolve().parent
    try:
        raw["prompts_path"] = str(prompts_path.relative_to(config_dir))
    except ValueError:
        raw["prompts_path"] = str(prompts_path)
    try:
        raw["dialogues_path"] = str(dialogues_path.relative_to(config_dir))
    except ValueError:
        raw["dialogues_path"] = str(dialogues_path)
    raw["frozen_index_path"] = str(Path(index_path))
    raw["require_approved_prompts"] = True
    raw["require_approved_dialogues"] = True
    config_file.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=False), encoding="utf-8")
    return {
        "prompts_path": raw["prompts_path"],
        "dialogues_path": raw["dialogues_path"],
    }
