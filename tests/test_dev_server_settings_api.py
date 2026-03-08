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
        assert post_code == 200
        assert post_payload["ok"] is True
        assert post_payload["configured"] is True
        assert post_payload["required_env_key"] == "DEEPSEEK_API_KEY"

        status_code, payload = _request_json(base_url, "/settings/apikey/status")
        assert status_code == 200
        assert payload["configured"] is True
        assert payload["source"] == "process_env"
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


def test_settings_apikey_clear(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-clear-123456")
    server, thread, base_url = _start_server(tmp_path)
    try:
        status_code, payload = _request_json(base_url, "/settings/apikey/status")
        assert status_code == 200
        assert payload["configured"] is True

        post_code, post_payload = _request_json(
            base_url,
            "/settings/apikey",
            method="POST",
            payload={"clear": True},
        )
        assert post_code == 200
        assert post_payload["ok"] is True
        assert post_payload["configured"] is False

        status_code, payload = _request_json(base_url, "/settings/apikey/status")
        assert status_code == 200
        assert payload["configured"] is False
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_settings_apikey_rejects_empty_value(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    server, thread, base_url = _start_server(tmp_path)
    try:
        status_code, payload = _request_json(
            base_url,
            "/settings/apikey",
            method="POST",
            payload={"deepseek_api_key": ""},
        )
        assert status_code == 400
        assert "non-empty string" in payload["error"]
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


def test_settings_config_get_and_save(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("max_turns: 10\nresume_strategy: reconstruct\n", encoding="utf-8")

    server, thread, base_url = _start_server(tmp_path)
    try:
        status_code, payload = _request_json(base_url, "/settings/config?config_path=config.yaml")
        assert status_code == 200
        assert payload["exists"] is True
        assert payload["config_path"] == "config.yaml"
        assert "max_turns: 10" in payload["content"]
        assert "max_turns" in payload["top_level_keys"]

        new_content = "max_turns: 5\nabort_on_error: false\n"
        post_code, post_payload = _request_json(
            base_url,
            "/settings/config",
            method="POST",
            payload={"config_path": "config.yaml", "content": new_content},
        )
        assert post_code == 200
        assert post_payload["ok"] is True
        assert post_payload["content"] == new_content
        assert (tmp_path / "config.yaml").read_text(encoding="utf-8") == new_content
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_settings_config_rejects_invalid_yaml(tmp_path):
    server, thread, base_url = _start_server(tmp_path)
    try:
        status_code, payload = _request_json(
            base_url,
            "/settings/config",
            method="POST",
            payload={"config_path": "config.yaml", "content": "max_turns: [1,\n"},
        )
        assert status_code == 400
        assert payload["error"].startswith("invalid yaml:")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_settings_config_rejects_path_escape(tmp_path):
    server, thread, base_url = _start_server(tmp_path)
    try:
        status_code, payload = _request_json(
            base_url,
            "/settings/config?config_path=../../etc/passwd",
        )
        assert status_code == 400
        assert "path must stay inside repository root" in payload["error"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
