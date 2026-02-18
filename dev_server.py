from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import uuid
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from backend.runs_index import build_runs_index, write_runs_index


class DevRequestHandler(SimpleHTTPRequestHandler):
    jobs: dict[str, dict] = {}

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
        requested_run_id = payload.get("run_id")
        if requested_run_id:
            run_id = str(requested_run_id)
        else:
            ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
            run_id = f"run_{ts}_{uuid.uuid4().hex[:6]}"

        cmd = [sys.executable, "control_agent.py", "--config", config_path, "--run_id", run_id]
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
            "config_path": config_path,
        }
        self._json_response(
            HTTPStatus.ACCEPTED,
            {
                "accepted": True,
                "run_id": run_id,
                "pid": process.pid,
                "dry_run": dry_run,
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
