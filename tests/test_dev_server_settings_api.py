import json
import threading
import uuid
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


def test_settings_apikey_status_env_only(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    server, thread, base_url = _start_server(tmp_path)
    try:
        status_code, payload = _request_json(base_url, "/settings/apikey/status")
        assert status_code == 200
        assert payload["configured"] is False

        post_code, post_payload = _request_json(
            base_url,
            "/settings/apikey",
            method="POST",
            payload={"deepseek_api_key": "sk-test-1234567890"},
        )
        assert post_code == 410
        assert "disabled" in post_payload["error"]
        assert post_payload["required_env_key"] == "DEEPSEEK_API_KEY"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_settings_apikey_status_uses_process_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-1234567890")
    server, thread, base_url = _start_server(tmp_path)
    try:
        status_code, payload = _request_json(base_url, "/settings/apikey/status")
        assert status_code == 200
        assert payload["configured"] is True
        assert payload["source"] == "process_env"
        assert payload["masked_key"] is not None
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_run_start_rejects_missing_api_key(tmp_path):
    missing_key_name = f"MISSING_TEST_KEY_{uuid.uuid4().hex[:8]}"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "llm3:",
                f"  api_key_env: {missing_key_name}",
                "llm4:",
                f"  api_key_env: {missing_key_name}",
            ]
        ),
        encoding="utf-8",
    )

    server, thread, base_url = _start_server(tmp_path)
    try:
        status_code, payload = _request_json(
            base_url,
            "/run/start",
            method="POST",
            payload={"config_path": "config.yaml", "dry_run": True},
        )
        assert status_code == 400
        assert payload["error"].startswith("Missing API key")
        assert missing_key_name in payload["missing_api_keys"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
