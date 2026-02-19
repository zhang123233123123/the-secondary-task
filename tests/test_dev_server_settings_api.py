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


def test_settings_apikey_status_and_save(tmp_path):
    server, thread, base_url = _start_server(tmp_path)
    try:
        status_code, payload = _request_json(base_url, "/settings/apikey/status")
        assert status_code == 200
        assert payload["configured"] is False

        bad_code, bad_payload = _request_json(
            base_url,
            "/settings/apikey",
            method="POST",
            payload={"deepseek_api_key": "short"},
        )
        assert bad_code == 400
        assert "at least 10 chars" in bad_payload["error"]

        save_code, save_payload = _request_json(
            base_url,
            "/settings/apikey",
            method="POST",
            payload={"deepseek_api_key": "sk-test-1234567890"},
        )
        assert save_code == 200
        assert save_payload["ok"] is True
        assert save_payload["configured"] is True
        assert save_payload["source"] == ".env.local"

        env_path = tmp_path / ".env.local"
        assert env_path.exists()
        assert "DEEPSEEK_API_KEY=" in env_path.read_text(encoding="utf-8")

        status_code_2, payload_2 = _request_json(base_url, "/settings/apikey/status")
        assert status_code_2 == 200
        assert payload_2["configured"] is True
        assert payload_2["masked_key"] is not None
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
