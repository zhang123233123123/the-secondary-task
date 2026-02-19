from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import uuid
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import yaml

from backend.env_store import mask_secret, read_local_env, resolve_secret, upsert_local_env
from backend.runs_index import build_runs_index, write_runs_index


class DevRequestHandler(SimpleHTTPRequestHandler):
    jobs: dict[str, dict] = {}
    local_env_file = ".env.local"
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
        config_path.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
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
    def _file_status(path: Path) -> dict:
        if not path.exists():
            return {
                "exists": False,
                "size_bytes": 0,
                "modified_at_utc": None,
                "relative_path": str(path.name),
            }
        stat = path.stat()
        return {
            "exists": True,
            "size_bytes": stat.st_size,
            "modified_at_utc": dt.datetime.fromtimestamp(
                stat.st_mtime,
                tz=dt.timezone.utc,
            ).isoformat(),
            "relative_path": str(path.name),
        }

    @classmethod
    def _settings_payload(cls, root: Path) -> dict:
        env_path = root / cls.local_env_file
        process_value = os.environ.get(cls.deepseek_key_name)
        if process_value:
            return {
                "configured": True,
                "masked_key": mask_secret(process_value),
                "source": "process_env",
            }

        local_values = read_local_env(env_path)
        local_value = local_values.get(cls.deepseek_key_name)
        if local_value:
            return {
                "configured": True,
                "masked_key": mask_secret(local_value),
                "source": cls.local_env_file,
            }

        return {
            "configured": False,
            "masked_key": None,
            "source": None,
        }

    @classmethod
    def _missing_api_keys(cls, config_payload: dict, root: Path) -> list[str]:
        llm3 = config_payload.get("llm3", {}) if isinstance(config_payload.get("llm3"), dict) else {}
        llm4 = config_payload.get("llm4", {}) if isinstance(config_payload.get("llm4"), dict) else {}
        required = {
            str(llm3.get("api_key_env", cls.deepseek_key_name)),
            str(llm4.get("api_key_env", cls.deepseek_key_name)),
        }
        env_path = root / cls.local_env_file
        missing: list[str] = []
        for key in sorted(required):
            if not key:
                continue
            if not resolve_secret(key, env_path):
                missing.append(key)
        return missing

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
        if parsed.path == "/inputs/status":
            query = parse_qs(parsed.query)
            config_path_raw = str(query.get("config_path", ["config.yaml"])[0])
            try:
                config_path = self._resolve_repo_path(root, config_path_raw)
            except ValueError as exc:
                self._json_response(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return

            files = {
                "config": self._file_status(config_path),
                "dialogues": self._file_status(root / "dialogues.jsonl"),
                "prompts": self._file_status(root / "prompts.json"),
            }
            raw_config = self._load_yaml_dict(config_path)
            llm3 = raw_config.get("llm3", {}) if isinstance(raw_config.get("llm3"), dict) else {}
            llm4 = raw_config.get("llm4", {}) if isinstance(raw_config.get("llm4"), dict) else {}
            self._json_response(
                HTTPStatus.OK,
                {
                    "config_path": str(config_path.relative_to(root))
                    if config_path.exists()
                    else config_path_raw,
                    "files": files,
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
            self._json_response(HTTPStatus.OK, self._settings_payload(root))
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
            key = str(payload.get("deepseek_api_key", "")).strip()
            if len(key) < 10:
                self._json_response(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "deepseek_api_key is required and must be at least 10 chars"},
                )
                return
            root = Path(self.directory or ".").resolve()
            env_path = root / self.local_env_file
            upsert_local_env(env_path, self.deepseek_key_name, key)
            self._json_response(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "configured": True,
                    "masked_key": mask_secret(key),
                    "source": self.local_env_file,
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

        missing_keys = self._missing_api_keys(resolved_config, root)
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
