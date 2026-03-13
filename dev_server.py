from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import threading
import uuid
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import yaml

from backend.frozen_registry import (
    apply_versions_to_config,
    approve_candidate,
    load_frozen_index,
    set_active_versions,
)
from backend.prepare_orchestrator import prepare_inputs
from backend.runs_index import build_runs_index, write_runs_index
from backend.runtime_config import load_config


class DevRequestHandler(SimpleHTTPRequestHandler):
    jobs: dict[str, dict] = {}
    prepare_jobs: dict[str, dict] = {}
    last_prepare_task_id: str | None = None
    jobs_lock = threading.Lock()
    deepseek_key_name = "DEEPSEEK_API_KEY"

    @staticmethod
    def _resolve_repo_path(root: Path, raw_path: str) -> Path:
        candidate = (root / raw_path).resolve()
        if root != candidate and root not in candidate.parents:
            raise ValueError("path must stay inside repository root")
        return candidate

    @staticmethod
    def _load_yaml_dict(path: Path) -> dict:
        if not path.exists():
            return {}
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _write_runtime_config(root: Path, run_id: str, payload: dict) -> str:
        runtime_dir = root / "output" / "runtime_configs"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        config_path = runtime_dir / f"runtime_config_{run_id}.yaml"
        
        # Convert relative paths to absolute paths to avoid resolution issues
        # when control_agent reads the config from the runtime_configs subdirectory
        patched_payload = json.loads(json.dumps(payload))  # deep copy
        for path_key in ["dialogues_path", "prompts_path", "frozen_index_path"]:
            if path_key in patched_payload:
                raw_value = str(patched_payload[path_key])
                # Only convert if it's a relative path
                if not Path(raw_value).is_absolute():
                    patched_payload[path_key] = str(root / raw_value)
        
        config_path.write_text(
            yaml.safe_dump(patched_payload, sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )
        return str(config_path.relative_to(root))

    @staticmethod
    def _apply_overrides(base_config: dict, overrides: dict) -> dict:
        merged = json.loads(json.dumps(base_config))
        llm3 = merged.setdefault("llm3", {})
        llm4 = merged.setdefault("llm4", {})

        if "max_turns" in overrides:
            merged["max_turns"] = max(1, int(overrides["max_turns"]))
        if "temperature" in overrides:
            llm3["temperature"] = float(overrides["temperature"])
        if "generator_model" in overrides and overrides["generator_model"]:
            llm3["model"] = str(overrides["generator_model"])
        if "judge_model" in overrides and overrides["judge_model"]:
            llm4["model"] = str(overrides["judge_model"])
        if "resume_strategy" in overrides:
            strategy = str(overrides["resume_strategy"])
            if strategy in {"reconstruct", "skip"}:
                merged["resume_strategy"] = strategy
        if "abort_on_error" in overrides:
            merged["abort_on_error"] = bool(overrides["abort_on_error"])
        return merged

    @staticmethod
    def _resolve_config_file_path(root: Path, config_path: Path, raw_path: str) -> Path:
        base_dir = config_path.parent
        candidate = (base_dir / raw_path).resolve()
        if root != candidate and root not in candidate.parents:
            raise ValueError("path must stay inside repository root")
        return candidate

    @staticmethod
    def _file_status(root: Path, path: Path, fallback_relative_path: str | None = None) -> dict:
        try:
            relative_path = str(path.resolve().relative_to(root))
        except ValueError:
            relative_path = fallback_relative_path or str(path)
        if not path.exists():
            return {
                "exists": False,
                "size_bytes": 0,
                "modified_at_utc": None,
                "relative_path": relative_path,
            }
        stat = path.stat()
        return {
            "exists": True,
            "size_bytes": stat.st_size,
            "modified_at_utc": dt.datetime.fromtimestamp(
                stat.st_mtime,
                tz=dt.timezone.utc,
            ).isoformat(),
            "relative_path": relative_path,
        }

    @staticmethod
    def _mask_secret(value: str | None) -> str | None:
        if value is None:
            return None
        secret = value.strip()
        if not secret:
            return None
        if len(secret) <= 8:
            return "*" * len(secret)
        return secret[:4] + "*" * (len(secret) - 8) + secret[-4:]

    @classmethod
    def _settings_payload(cls) -> dict:
        process_value = os.environ.get(cls.deepseek_key_name)
        return {
            "configured": bool(process_value),
            "masked_key": cls._mask_secret(process_value),
            "source": "process_env" if process_value else None,
        }

    @staticmethod
    def _config_editor_payload(root: Path, config_path: Path, fallback_path: str) -> dict:
        if not config_path.exists():
            return {
                "config_path": fallback_path,
                "exists": False,
                "size_bytes": 0,
                "modified_at_utc": None,
                "content": "",
                "top_level_keys": [],
            }
        content = config_path.read_text(encoding="utf-8")
        parsed = yaml.safe_load(content)
        top_level_keys: list[str] = []
        if isinstance(parsed, dict):
            top_level_keys = sorted(str(key) for key in parsed.keys())
        stat = config_path.stat()
        return {
            "config_path": str(config_path.relative_to(root)),
            "exists": True,
            "size_bytes": stat.st_size,
            "modified_at_utc": dt.datetime.fromtimestamp(
                stat.st_mtime,
                tz=dt.timezone.utc,
            ).isoformat(),
            "content": content,
            "top_level_keys": top_level_keys,
        }

    @classmethod
    def _missing_api_keys_for_roles(cls, config_payload: dict, roles: tuple[str, ...]) -> list[str]:
        required: set[str] = set()
        for role in roles:
            role_config = config_payload.get(role, {})
            if isinstance(role_config, dict):
                required.add(str(role_config.get("api_key_env", cls.deepseek_key_name)))
            else:
                required.add(cls.deepseek_key_name)
        missing: list[str] = []
        for key in sorted(required):
            if not key:
                continue
            if not os.environ.get(key):
                missing.append(key)
        return missing

    @classmethod
    def _missing_api_keys(cls, config_payload: dict) -> list[str]:
        return cls._missing_api_keys_for_roles(config_payload, ("llm3", "llm4"))

    @staticmethod
    def _relative_to_root(root: Path, path: Path, fallback: str) -> str:
        try:
            return str(path.resolve().relative_to(root))
        except ValueError:
            return fallback

    def _resolve_config_and_index_paths(
        self,
        root: Path,
        *,
        config_path_raw: str,
        index_path_raw: str | None,
    ) -> tuple[Path, Path, dict, str]:
        config_path = self._resolve_repo_path(root, config_path_raw)
        config_payload = self._load_yaml_dict(config_path)
        raw_index = index_path_raw
        if not raw_index:
            raw_index = str(config_payload.get("frozen_index_path", "frozen_inputs/index.json"))
        index_path = self._resolve_config_file_path(root, config_path, raw_index)
        return config_path, index_path, config_payload, raw_index

    @staticmethod
    def _index_entry_by_version(entries: list[dict], version: str | None) -> dict | None:
        if not version:
            return None
        for item in entries:
            if isinstance(item, dict) and item.get("version") == version:
                return item
        return None

    def _frozen_index_payload(
        self,
        *,
        root: Path,
        config_path: Path,
        index_path: Path,
        index_path_raw: str,
    ) -> dict:
        index_data = load_frozen_index(index_path)
        prompts_versions = index_data.get("prompts_versions", [])
        dialogues_versions = index_data.get("dialogues_versions", [])
        active = index_data.get("active", {})
        prompts_active = self._index_entry_by_version(prompts_versions, active.get("prompts_version"))
        dialogues_active = self._index_entry_by_version(
            dialogues_versions,
            active.get("dialogues_version"),
        )
        return {
            "config_path": self._relative_to_root(root, config_path, str(config_path)),
            "index_path": self._relative_to_root(root, index_path, index_path_raw),
            "active": active,
            "active_entries": {
                "prompts": prompts_active,
                "dialogues": dialogues_active,
            },
            "prompts_versions": prompts_versions,
            "dialogues_versions": dialogues_versions,
        }

    @classmethod
    def _prepare_status_payload(cls, task_id: str, root: Path) -> dict:
        with cls.jobs_lock:
            job = cls.prepare_jobs.get(task_id)
        if job is None:
            return {"task_id": task_id, "status": "unknown"}
        manifest_file = job.get("manifest_file")
        manifest_relative = None
        if isinstance(manifest_file, Path):
            try:
                manifest_relative = str(manifest_file.resolve().relative_to(root))
            except ValueError:
                manifest_relative = str(manifest_file)
        payload = {
            "task_id": task_id,
            "status": job.get("status", "unknown"),
            "prepare_id": job.get("prepare_id"),
            "manifest_file": manifest_relative,
            "config_path": job.get("config_path"),
            "target_version": job.get("target_version"),
            "skip_llm1": job.get("skip_llm1", False),
            "started_at_utc": job.get("started_at_utc"),
            "finished_at_utc": job.get("finished_at_utc"),
            "error": job.get("error"),
            "progress": job.get("progress"),
        }
        return payload

    @classmethod
    def _run_prepare_job(
        cls,
        *,
        root: Path,
        task_id: str,
        config_path: Path,
        target_version: str | None,
        skip_llm1: bool,
    ) -> None:
        def progress_callback(progress: dict) -> None:
            with cls.jobs_lock:
                if task_id in cls.prepare_jobs:
                    cls.prepare_jobs[task_id]["progress"] = progress
        
        try:
            config = load_config(config_path)
            manifest = prepare_inputs(
                config=config,
                config_path=str(config_path),
                target_version=target_version,
                skip_llm1=skip_llm1,
                progress_callback=progress_callback,
            )
            prepare_id = str(manifest.get("prepare_id", target_version or ""))
            manifest_file = Path(manifest.get("prompts_candidate", "")).parent / (
                f"prepare_manifest_{prepare_id}.json"
            )
            with cls.jobs_lock:
                job = cls.prepare_jobs[task_id]
                job["status"] = "succeeded" if manifest.get("status") == "complete" else "partial"
                job["prepare_id"] = prepare_id
                job["manifest_file"] = manifest_file
                job["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
                job["error"] = None
        except Exception as exc:  # noqa: BLE001
            # Try to find partial results manifest
            prepare_id = None
            manifest_file = None
            try:
                config = load_config(config_path)
                config_dir = config_path.parent
                frozen_index = Path(config.frozen_index_path)
                if not frozen_index.is_absolute():
                    frozen_index = (config_dir / frozen_index).resolve()
                candidates_dir = frozen_index.parent / "candidates"
                # Look for manifest matching target_version or most recent
                expected_id = target_version or ""
                if expected_id:
                    manifest_candidate = candidates_dir / f"prepare_manifest_{expected_id}.json"
                    if manifest_candidate.exists():
                        prepare_id = expected_id
                        manifest_file = manifest_candidate
            except Exception:  # noqa: BLE001, S110
                pass  # Failed to find partial results
            
            with cls.jobs_lock:
                job = cls.prepare_jobs[task_id]
                job["status"] = "partial" if manifest_file else "failed"
                job["prepare_id"] = prepare_id
                job["manifest_file"] = manifest_file
                job["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
                job["error"] = str(exc)

    def _json_response(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        return json.loads(raw.decode("utf-8"))

    @classmethod
    def _runtime_status(cls, run_id: str, root: Path) -> str:
        job = cls.jobs.get(run_id)
        summary_path = root / "output" / f"run_summary_{run_id}.json"
        if job is None:
            return "succeeded" if summary_path.exists() else "unknown"

        process: subprocess.Popen = job["process"]
        code = process.poll()
        if code is None:
            return "running"
        if summary_path.exists() and code == 0:
            return "succeeded"
        return "failed"

    @classmethod
    def _job_payload(cls, run_id: str, root: Path) -> dict:
        job = cls.jobs.get(run_id)
        summary_file = f"run_summary_{run_id}.json"
        results_file = f"results_{run_id}.jsonl"
        status = cls._runtime_status(run_id, root)
        payload = {
            "run_id": run_id,
            "runtime_status": status,
            "summary_file": summary_file,
            "results_file": results_file,
            "summary_exists": (root / "output" / summary_file).exists(),
            "results_exists": (root / "output" / results_file).exists(),
        }
        if job is not None:
            payload["pid"] = job["pid"]
            payload["started_at_utc"] = job["started_at_utc"]
            payload["dry_run"] = job["dry_run"]
            payload["config_path"] = job["config_path"]
        return payload

    def do_GET(self) -> None:  # noqa: N802
        root = Path(self.directory or ".").resolve()
        if self.path == "/health":
            self._json_response(HTTPStatus.OK, {"ok": True})
            return
        parsed = urlparse(self.path)
        if parsed.path == "/prepare/status":
            query = parse_qs(parsed.query)
            task_id = query.get("task_id", [None])[0]
            if not task_id:
                task_id = self.last_prepare_task_id
            if not task_id:
                self._json_response(HTTPStatus.OK, {"task_id": None, "status": "idle"})
                return
            payload = self._prepare_status_payload(str(task_id), root)
            self._json_response(HTTPStatus.OK, payload)
            return
        if parsed.path == "/prepare/manifest":
            query = parse_qs(parsed.query)
            manifest_raw = str(query.get("manifest_file", [""])[0]).strip()
            if not manifest_raw:
                self._json_response(HTTPStatus.BAD_REQUEST, {"error": "manifest_file is required"})
                return
            try:
                manifest_path = self._resolve_repo_path(root, manifest_raw)
            except ValueError as exc:
                self._json_response(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            if not manifest_path.exists():
                self._json_response(HTTPStatus.NOT_FOUND, {"error": "manifest file not found"})
                return
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                self._json_response(HTTPStatus.BAD_REQUEST, {"error": f"invalid manifest json: {exc}"})
                return
            self._json_response(
                HTTPStatus.OK,
                {
                    "manifest_file": self._relative_to_root(root, manifest_path, manifest_raw),
                    "manifest": payload,
                },
            )
            return
        if parsed.path == "/prepare/candidate/prompts":
            query = parse_qs(parsed.query)
            candidate_raw = str(query.get("file", [""])[0]).strip()
            if not candidate_raw:
                self._json_response(HTTPStatus.BAD_REQUEST, {"error": "file is required"})
                return
            try:
                candidate_path = self._resolve_repo_path(root, candidate_raw)
            except ValueError as exc:
                self._json_response(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            if not candidate_path.exists():
                self._json_response(HTTPStatus.NOT_FOUND, {"error": "candidate file not found"})
                return
            try:
                content = json.loads(candidate_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                self._json_response(
                    HTTPStatus.BAD_REQUEST,
                    {"error": f"invalid prompts json: {exc}"},
                )
                return
            self._json_response(
                HTTPStatus.OK,
                {
                    "file": self._relative_to_root(root, candidate_path, candidate_raw),
                    "content": content,
                },
            )
            return
        if parsed.path == "/prepare/candidate/dialogues":
            query = parse_qs(parsed.query)
            candidate_raw = str(query.get("file", [""])[0]).strip()
            if not candidate_raw:
                self._json_response(HTTPStatus.BAD_REQUEST, {"error": "file is required"})
                return
            sample_raw = str(query.get("sample", ["20"])[0]).strip()
            try:
                sample_size = max(1, min(100, int(sample_raw)))
            except ValueError:
                self._json_response(HTTPStatus.BAD_REQUEST, {"error": "sample must be an integer"})
                return
            try:
                candidate_path = self._resolve_repo_path(root, candidate_raw)
            except ValueError as exc:
                self._json_response(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            if not candidate_path.exists():
                self._json_response(HTTPStatus.NOT_FOUND, {"error": "candidate file not found"})
                return

            total = 0
            invalid_rows = 0
            domain_counts: dict[str, int] = {}
            turn_count_min: int | None = None
            turn_count_max: int | None = None
            sample_rows: list[dict] = []

            with candidate_path.open("r", encoding="utf-8") as fh:
                for line_no, line in enumerate(fh, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    total += 1
                    try:
                        row = json.loads(stripped)
                    except json.JSONDecodeError:
                        invalid_rows += 1
                        continue
                    domain = str(row.get("domain", ""))
                    domain_counts[domain] = domain_counts.get(domain, 0) + 1
                    turns = row.get("turns")
                    turn_count = len(turns) if isinstance(turns, list) else 0
                    if turn_count_min is None or turn_count < turn_count_min:
                        turn_count_min = turn_count
                    if turn_count_max is None or turn_count > turn_count_max:
                        turn_count_max = turn_count
                    if len(sample_rows) < sample_size:
                        sample_rows.append(
                            {
                                "line": line_no,
                                "dialogue_id": row.get("dialogue_id"),
                                "domain": row.get("domain"),
                                "turn_count": turn_count,
                            }
                        )

            self._json_response(
                HTTPStatus.OK,
                {
                    "file": self._relative_to_root(root, candidate_path, candidate_raw),
                    "total": total,
                    "invalid_rows": invalid_rows,
                    "domain_counts": domain_counts,
                    "turn_count_min": turn_count_min,
                    "turn_count_max": turn_count_max,
                    "sample_rows": sample_rows,
                },
            )
            return
        if parsed.path == "/frozen/index":
            query = parse_qs(parsed.query)
            config_path_raw = str(query.get("config_path", ["config.yaml"])[0])
            index_path_raw = query.get("index_path", [None])[0]
            try:
                config_path, index_path, _, resolved_index_raw = self._resolve_config_and_index_paths(
                    root,
                    config_path_raw=config_path_raw,
                    index_path_raw=str(index_path_raw) if index_path_raw else None,
                )
            except ValueError as exc:
                self._json_response(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            payload = self._frozen_index_payload(
                root=root,
                config_path=config_path,
                index_path=index_path,
                index_path_raw=resolved_index_raw,
            )
            self._json_response(HTTPStatus.OK, payload)
            return
        if parsed.path == "/inputs/prompts":
            query = parse_qs(parsed.query)
            config_path_raw = str(query.get("config_path", ["config.yaml"])[0])
            try:
                config_path = self._resolve_repo_path(root, config_path_raw)
            except ValueError as exc:
                self._json_response(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            raw_config = self._load_yaml_dict(config_path)
            prompts_path_raw = str(raw_config.get("prompts_path", "prompts.json"))
            try:
                prompts_path = self._resolve_config_file_path(root, config_path, prompts_path_raw)
            except ValueError:
                prompts_path = root / prompts_path_raw
            if not prompts_path.exists():
                self._json_response(HTTPStatus.NOT_FOUND, {"error": f"Prompts file not found: {prompts_path_raw}"})
                return
            try:
                content = json.loads(prompts_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                self._json_response(HTTPStatus.BAD_REQUEST, {"error": f"Invalid prompts JSON: {exc}"})
                return
            self._json_response(HTTPStatus.OK, {
                "file": self._relative_to_root(root, prompts_path, prompts_path_raw),
                "conditions": content.get("conditions", {}),
                "judge_system": content.get("judge_system", ""),
                "judge_rubric": content.get("judge_rubric", ""),
                "judge_schema": content.get("judge_schema", {}),
            })
            return
        if parsed.path == "/inputs/status":
            query = parse_qs(parsed.query)
            config_path_raw = str(query.get("config_path", ["config.yaml"])[0])
            try:
                config_path = self._resolve_repo_path(root, config_path_raw)
            except ValueError as exc:
                self._json_response(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return

            raw_config = self._load_yaml_dict(config_path)
            dialogues_path_raw = str(raw_config.get("dialogues_path", "dialogues.jsonl"))
            prompts_path_raw = str(raw_config.get("prompts_path", "prompts.json"))
            path_errors: dict[str, str] = {}
            try:
                dialogues_path = self._resolve_config_file_path(root, config_path, dialogues_path_raw)
            except ValueError as exc:
                dialogues_path = root / dialogues_path_raw
                path_errors["dialogues"] = str(exc)
            try:
                prompts_path = self._resolve_config_file_path(root, config_path, prompts_path_raw)
            except ValueError as exc:
                prompts_path = root / prompts_path_raw
                path_errors["prompts"] = str(exc)

            files = {
                "config": self._file_status(root, config_path, fallback_relative_path=config_path_raw),
                "dialogues": self._file_status(
                    root,
                    dialogues_path,
                    fallback_relative_path=dialogues_path_raw,
                ),
                "prompts": self._file_status(
                    root,
                    prompts_path,
                    fallback_relative_path=prompts_path_raw,
                ),
            }
            llm3 = raw_config.get("llm3", {}) if isinstance(raw_config.get("llm3"), dict) else {}
            llm4 = raw_config.get("llm4", {}) if isinstance(raw_config.get("llm4"), dict) else {}
            self._json_response(
                HTTPStatus.OK,
                {
                    "config_path": str(config_path.relative_to(root))
                    if config_path.exists()
                    else config_path_raw,
                    "files": files,
                    "path_errors": path_errors,
                    "defaults": {
                        "generator_model": str(llm3.get("model", "deepseek-chat")),
                        "judge_model": str(llm4.get("model", "deepseek-chat")),
                        "temperature": float(llm3.get("temperature", 0.7)),
                        "max_turns": int(raw_config.get("max_turns", 10)),
                        "resume_strategy": str(raw_config.get("resume_strategy", "reconstruct")),
                        "abort_on_error": bool(raw_config.get("abort_on_error", False)),
                    },
                },
            )
            return
        if parsed.path == "/settings/apikey/status":
            self._json_response(HTTPStatus.OK, self._settings_payload())
            return
        if parsed.path == "/settings/config":
            query = parse_qs(parsed.query)
            config_path_raw = str(query.get("config_path", ["config.yaml"])[0])
            try:
                config_path = self._resolve_repo_path(root, config_path_raw)
            except ValueError as exc:
                self._json_response(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            payload = self._config_editor_payload(root, config_path, config_path_raw)
            self._json_response(HTTPStatus.OK, payload)
            return
        if parsed.path == "/runs":
            output_dir = root / "output"
            index_path = output_dir / "runs_index.json"
            if not index_path.exists():
                write_runs_index(output_dir)
            index = build_runs_index(output_dir)

            by_id = {item["run_id"]: item for item in index["runs"]}
            for run_item in by_id.values():
                run_item["runtime_status"] = self._runtime_status(run_item["run_id"], root)
            for run_id in self.jobs:
                if run_id not in by_id:
                    by_id[run_id] = self._job_payload(run_id, root)

            runs = sorted(
                by_id.values(),
                key=lambda item: item.get("created_at_utc", item.get("started_at_utc", "")),
                reverse=True,
            )
            latest_run_id = runs[0]["run_id"] if runs else None
            self._json_response(HTTPStatus.OK, {"latest_run_id": latest_run_id, "runs": runs})
            return
        if parsed.path == "/run/status":
            query = parse_qs(parsed.query)
            run_id = query.get("run_id", [None])[0]
            if not run_id:
                self._json_response(HTTPStatus.BAD_REQUEST, {"error": "run_id is required"})
                return
            self._json_response(HTTPStatus.OK, self._job_payload(str(run_id), root))
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/settings/apikey":
            try:
                payload = self._read_json_body()
            except json.JSONDecodeError:
                self._json_response(HTTPStatus.BAD_REQUEST, {"error": "invalid json"})
                return
            clear = bool(payload.get("clear", False))
            if clear:
                os.environ.pop(self.deepseek_key_name, None)
                self._json_response(
                    HTTPStatus.OK,
                    {
                        "ok": True,
                        "message": "API key cleared from current dev_server process.",
                        "required_env_key": self.deepseek_key_name,
                        **self._settings_payload(),
                    },
                )
                return
            key_value = payload.get("deepseek_api_key")
            if not isinstance(key_value, str) or not key_value.strip():
                self._json_response(
                    HTTPStatus.BAD_REQUEST,
                    {
                        "error": "deepseek_api_key must be a non-empty string",
                        "required_env_key": self.deepseek_key_name,
                    },
                )
                return
            os.environ[self.deepseek_key_name] = key_value.strip()
            self._json_response(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "message": "API key updated in current dev_server process.",
                    "required_env_key": self.deepseek_key_name,
                    **self._settings_payload(),
                },
            )
            return
        if self.path == "/settings/config":
            try:
                payload = self._read_json_body()
            except json.JSONDecodeError:
                self._json_response(HTTPStatus.BAD_REQUEST, {"error": "invalid json"})
                return
            root = Path(self.directory or ".").resolve()
            config_path_raw = str(payload.get("config_path", "config.yaml"))
            content = payload.get("content")
            if not isinstance(content, str):
                self._json_response(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "content must be a string"},
                )
                return
            try:
                parsed_yaml = yaml.safe_load(content)
            except yaml.YAMLError as exc:
                self._json_response(
                    HTTPStatus.BAD_REQUEST,
                    {"error": f"invalid yaml: {exc}"},
                )
                return
            if parsed_yaml is not None and not isinstance(parsed_yaml, dict):
                self._json_response(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "config yaml root must be an object"},
                )
                return
            try:
                config_path = self._resolve_repo_path(root, config_path_raw)
            except ValueError as exc:
                self._json_response(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            config_path.parent.mkdir(parents=True, exist_ok=True)
            normalized_content = content if content.endswith("\n") else content + "\n"
            config_path.write_text(normalized_content, encoding="utf-8")
            self._json_response(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "message": "Config file saved.",
                    **self._config_editor_payload(root, config_path, config_path_raw),
                },
            )
            return

        if self.path == "/prepare/start":
            try:
                payload = self._read_json_body()
            except json.JSONDecodeError:
                self._json_response(HTTPStatus.BAD_REQUEST, {"error": "invalid json"})
                return
            root = Path(self.directory or ".").resolve()
            config_path_raw = str(payload.get("config_path", "config.yaml"))
            target_version_raw = payload.get("target_version")
            target_version = str(target_version_raw).strip() if target_version_raw else None
            skip_llm1 = bool(payload.get("skip_llm1", True))
            try:
                config_path = self._resolve_repo_path(root, config_path_raw)
            except ValueError as exc:
                self._json_response(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            if not config_path.exists():
                self._json_response(
                    HTTPStatus.BAD_REQUEST,
                    {"error": f"Config file not found: {config_path_raw}"},
                )
                return
            raw_config = self._load_yaml_dict(config_path)
            roles = ("llm2",) if skip_llm1 else ("llm1", "llm2")
            missing_keys = self._missing_api_keys_for_roles(raw_config, roles)
            if missing_keys:
                self._json_response(
                    HTTPStatus.BAD_REQUEST,
                    {
                        "error": "Missing API key(s) for prepare.",
                        "missing_api_keys": missing_keys,
                    },
                )
                return

            with self.jobs_lock:
                running_task = next(
                    (
                        task_id
                        for task_id, job in self.prepare_jobs.items()
                        if job.get("status") == "running"
                    ),
                    None,
                )
                if running_task is not None:
                    self._json_response(
                        HTTPStatus.CONFLICT,
                        {
                            "error": "A prepare task is already running.",
                            "running_task_id": running_task,
                        },
                    )
                    return

                task_id = f"prepare_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
                self.last_prepare_task_id = task_id
                self.prepare_jobs[task_id] = {
                    "status": "running",
                    "prepare_id": target_version,
                    "manifest_file": None,
                    "config_path": self._relative_to_root(root, config_path, config_path_raw),
                    "target_version": target_version,
                    "skip_llm1": skip_llm1,
                    "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "finished_at_utc": None,
                    "error": None,
                }

                thread = threading.Thread(
                    target=self._run_prepare_job,
                    kwargs={
                        "root": root,
                        "task_id": task_id,
                        "config_path": config_path,
                        "target_version": target_version,
                        "skip_llm1": skip_llm1,
                    },
                    daemon=True,
                )
                self.prepare_jobs[task_id]["thread"] = thread
                thread.start()

            self._json_response(
                HTTPStatus.ACCEPTED,
                {
                    "accepted": True,
                    "task_id": task_id,
                    "status": "running",
                    "target_version": target_version,
                    "skip_llm1": skip_llm1,
                },
            )
            return

        if self.path in {"/frozen/approve-prompts", "/frozen/approve-dialogues"}:
            try:
                payload = self._read_json_body()
            except json.JSONDecodeError:
                self._json_response(HTTPStatus.BAD_REQUEST, {"error": "invalid json"})
                return
            root = Path(self.directory or ".").resolve()
            kind = "prompts" if self.path.endswith("prompts") else "dialogues"
            candidate_raw = str(payload.get("candidate", "")).strip()
            version = str(payload.get("version", "")).strip()
            reviewer = str(payload.get("reviewer", "")).strip()
            note = payload.get("note")
            activate = bool(payload.get("activate", False))
            config_path_raw = str(payload.get("config_path", "config.yaml"))
            index_path_raw = payload.get("index_path")
            if not candidate_raw or not version or not reviewer:
                self._json_response(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "candidate, version, reviewer are required"},
                )
                return
            try:
                candidate_path = self._resolve_repo_path(root, candidate_raw)
                config_path, index_path, _, resolved_index_raw = self._resolve_config_and_index_paths(
                    root,
                    config_path_raw=config_path_raw,
                    index_path_raw=str(index_path_raw) if index_path_raw else None,
                )
            except ValueError as exc:
                self._json_response(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            try:
                entry = approve_candidate(
                    index_path=index_path,
                    kind=kind,  # type: ignore[arg-type]
                    candidate_path=candidate_path,
                    version=version,
                    reviewer=reviewer,
                    note=str(note) if note is not None else None,
                )
                if activate:
                    if kind == "prompts":
                        set_active_versions(index_path=index_path, prompts_version=version)
                    else:
                        set_active_versions(index_path=index_path, dialogues_version=version)
            except Exception as exc:  # noqa: BLE001
                self._json_response(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            payload_out = self._frozen_index_payload(
                root=root,
                config_path=config_path,
                index_path=index_path,
                index_path_raw=resolved_index_raw,
            )
            self._json_response(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "kind": kind,
                    "entry": entry,
                    "activate": activate,
                    **payload_out,
                },
            )
            return

        if self.path == "/frozen/use":
            try:
                payload = self._read_json_body()
            except json.JSONDecodeError:
                self._json_response(HTTPStatus.BAD_REQUEST, {"error": "invalid json"})
                return
            root = Path(self.directory or ".").resolve()
            config_path_raw = str(payload.get("config_path", "config.yaml"))
            index_path_raw = payload.get("index_path")
            prompts_version = str(payload.get("prompts_version", "")).strip()
            dialogues_version = str(payload.get("dialogues_version", "")).strip()
            if not prompts_version or not dialogues_version:
                self._json_response(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "prompts_version and dialogues_version are required"},
                )
                return
            try:
                config_path, index_path, _, resolved_index_raw = self._resolve_config_and_index_paths(
                    root,
                    config_path_raw=config_path_raw,
                    index_path_raw=str(index_path_raw) if index_path_raw else None,
                )
            except ValueError as exc:
                self._json_response(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            try:
                updated = apply_versions_to_config(
                    config_path=config_path,
                    index_path=index_path,
                    prompts_version=prompts_version,
                    dialogues_version=dialogues_version,
                )
            except Exception as exc:  # noqa: BLE001
                self._json_response(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            payload_out = self._frozen_index_payload(
                root=root,
                config_path=config_path,
                index_path=index_path,
                index_path_raw=resolved_index_raw,
            )
            self._json_response(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "updated": updated,
                    **payload_out,
                },
            )
            return

        if self.path == "/setup/draft":
            try:
                payload = self._read_json_body()
            except json.JSONDecodeError:
                self._json_response(HTTPStatus.BAD_REQUEST, {"error": "invalid json"})
                return
            root = Path(self.directory or ".").resolve()
            output_dir = root / "output" / "drafts"
            output_dir.mkdir(parents=True, exist_ok=True)
            ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
            file_name = f"draft_{ts}_{uuid.uuid4().hex[:6]}.json"
            draft_path = output_dir / file_name
            draft_payload = {
                "saved_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "config_path": payload.get("config_path", "config.yaml"),
                "dry_run": bool(payload.get("dry_run", False)),
                "overrides": payload.get("overrides", {}),
            }
            draft_path.write_text(
                json.dumps(draft_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self._json_response(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "draft_file": str(draft_path.relative_to(root)),
                },
            )
            return

        if self.path != "/run/start":
            self._json_response(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return

        try:
            payload = self._read_json_body()
        except json.JSONDecodeError:
            self._json_response(HTTPStatus.BAD_REQUEST, {"error": "invalid json"})
            return

        root = Path(self.directory or ".").resolve()
        config_path = str(payload.get("config_path", "config.yaml"))
        dry_run = bool(payload.get("dry_run", False))
        overrides = payload.get("overrides", {})
        requested_run_id = payload.get("run_id")
        if requested_run_id:
            run_id = str(requested_run_id)
        else:
            ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
            run_id = f"run_{ts}_{uuid.uuid4().hex[:6]}"

        try:
            config_file = self._resolve_repo_path(root, config_path)
        except ValueError as exc:
            self._json_response(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        if not config_file.exists():
            self._json_response(
                HTTPStatus.BAD_REQUEST,
                {"error": f"Config file not found: {config_path}"},
            )
            return

        base_config = self._load_yaml_dict(config_file)
        runtime_config_path = config_path
        resolved_config = base_config
        if isinstance(overrides, dict) and overrides:
            try:
                merged_config = self._apply_overrides(base_config, overrides)
            except (TypeError, ValueError) as exc:
                self._json_response(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            resolved_config = merged_config
            runtime_config_path = self._write_runtime_config(root, run_id, merged_config)

        missing_keys = self._missing_api_keys(resolved_config)
        if missing_keys:
            self._json_response(
                HTTPStatus.BAD_REQUEST,
                {
                    "error": "Missing API key(s). Configure in Settings first.",
                    "missing_api_keys": missing_keys,
                },
            )
            return

        cmd = [sys.executable, "control_agent.py", "--config", runtime_config_path, "--run_id", run_id]
        if dry_run:
            cmd.append("--dry_run")

        process = subprocess.Popen(  # noqa: S603
            cmd,
            cwd=str(root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.jobs[run_id] = {
            "process": process,
            "pid": process.pid,
            "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "dry_run": dry_run,
            "config_path": runtime_config_path,
        }
        self._json_response(
            HTTPStatus.ACCEPTED,
            {
                "accepted": True,
                "run_id": run_id,
                "pid": process.pid,
                "dry_run": dry_run,
                "config_path": runtime_config_path,
            },
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local dev server for frontend/output")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--root", default=".", help="Repository root to serve")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    handler = lambda *h_args, **h_kwargs: DevRequestHandler(  # noqa: E731
        *h_args,
        directory=str(root),
        **h_kwargs,
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving {root} at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
