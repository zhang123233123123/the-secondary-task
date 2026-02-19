import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib import error, request

from dev_server import DevRequestHandler


def _start_server(root: Path):
    DevRequestHandler.jobs = {}
    handler = lambda *args, **kwargs: DevRequestHandler(  # noqa: E731
        *args,
        directory=str(root),
        **kwargs,
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return server, thread, f"http://{host}:{port}"


def _request_json(base_url: str, path: str, method: str = "GET", payload: dict | None = None):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if body is not None else {}
    req = request.Request(base_url + path, data=body, method=method, headers=headers)
    try:
        with request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def test_inputs_status_uses_paths_from_config(tmp_path):
    cfg_dir = tmp_path / "cfg"
    data_dir = tmp_path / "data"
    cfg_dir.mkdir()
    data_dir.mkdir()
    (data_dir / "dialogues.jsonl").write_text('{"dialogue_id":"x","turns":["hi"]}\n', encoding="utf-8")
    (data_dir / "prompts.json").write_text(
        json.dumps(
            {
                "conditions": {"default": "a", "evil": "b", "distant": "c"},
                "judge_system": "js",
                "judge_rubric": "jr",
                "judge_schema": {},
            }
        ),
        encoding="utf-8",
    )
    (cfg_dir / "config.yaml").write_text(
        "\n".join(
            [
                "dialogues_path: ../data/dialogues.jsonl",
                "prompts_path: ../data/prompts.json",
            ]
        ),
        encoding="utf-8",
    )

    server, thread, base_url = _start_server(tmp_path)
    try:
        status_code, payload = _request_json(base_url, "/inputs/status?config_path=cfg/config.yaml")
        assert status_code == 200
        assert payload["files"]["dialogues"]["exists"] is True
        assert payload["files"]["prompts"]["exists"] is True
        assert payload["files"]["dialogues"]["relative_path"] == "data/dialogues.jsonl"
        assert payload["files"]["prompts"]["relative_path"] == "data/prompts.json"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_inputs_status_marks_path_escape_error(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "\n".join(
            [
                "dialogues_path: ../../outside_dialogues.jsonl",
                "prompts_path: prompts.json",
            ]
        ),
        encoding="utf-8",
    )

    server, thread, base_url = _start_server(tmp_path)
    try:
        status_code, payload = _request_json(base_url, "/inputs/status?config_path=config.yaml")
        assert status_code == 200
        assert payload["files"]["dialogues"]["exists"] is False
        assert "dialogues" in payload["path_errors"]
        assert "path must stay inside repository root" in payload["path_errors"]["dialogues"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
