import json
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib import error, request

from dev_server import DevRequestHandler


def _start_server(root: Path):
    DevRequestHandler.jobs = {}
    DevRequestHandler.prepare_jobs = {}
    DevRequestHandler.last_prepare_task_id = None
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


def _write_prepare_config(root: Path, key_name: str) -> Path:
    config_path = root / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "frozen_index_path: frozen_inputs/index.json",
                "llm1:",
                "  provider: deepseek",
                "  model: llm1-test",
                f"  api_key_env: {key_name}",
                "  base_url: https://api.deepseek.com/v1",
                "  temperature: 0.2",
                "  top_p: 1.0",
                "llm2:",
                "  provider: deepseek",
                "  model: llm2-test",
                f"  api_key_env: {key_name}",
                "  base_url: https://api.deepseek.com/v1",
                "  temperature: 0.9",
                "  top_p: 1.0",
                "llm3:",
                "  provider: deepseek",
                "  model: llm3-test",
                f"  api_key_env: {key_name}",
                "  base_url: https://api.deepseek.com/v1",
                "  temperature: 0.7",
                "  top_p: 1.0",
                "llm4:",
                "  provider: deepseek",
                "  model: llm4-test",
                f"  api_key_env: {key_name}",
                "  base_url: https://api.deepseek.com/v1",
                "  temperature: 0.0",
                "  top_p: 1.0",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def test_prepare_start_status_and_candidate_reads(tmp_path, monkeypatch):
    key_name = "TEST_PREP_KEY"
    monkeypatch.setenv(key_name, "sk-test-prepare-123")
    _write_prepare_config(tmp_path, key_name)

    def fake_prepare_inputs(*, config, config_path, target_version=None):  # noqa: ANN001
        del config
        config_file = Path(config_path)
        candidates_dir = config_file.parent / "frozen_inputs" / "candidates"
        candidates_dir.mkdir(parents=True, exist_ok=True)
        prepare_id = target_version or "prep_generated"
        prompts_candidate = candidates_dir / f"prompts_candidate_{prepare_id}.json"
        dialogues_candidate = candidates_dir / f"dialogues_candidate_{prepare_id}.jsonl"
        prompts_payload = {
            "conditions": {"default": "d", "evil": "e", "distant": "x"},
            "judge_system": "judge",
            "judge_rubric": "rubric",
            "judge_schema": {"type": "object"},
        }
        prompts_candidate.write_text(json.dumps(prompts_payload), encoding="utf-8")
        dialogues_candidate.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "dialogue_id": "D1",
                            "domain": "creative",
                            "turns": [{"role": "user", "text": "hello"}],
                        }
                    ),
                    json.dumps(
                        {
                            "dialogue_id": "D2",
                            "domain": "finance",
                            "turns": [{"role": "user", "text": "world"}],
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        manifest = {
            "prepare_id": prepare_id,
            "index_path": str(config_file.parent / "frozen_inputs" / "index.json"),
            "prompts_candidate": str(prompts_candidate),
            "dialogues_candidate": str(dialogues_candidate),
            "llm1_model": "llm1-test",
            "llm2_model": "llm2-test",
        }
        manifest_path = candidates_dir / f"prepare_manifest_{prepare_id}.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest

    monkeypatch.setattr("dev_server.prepare_inputs", fake_prepare_inputs)

    server, thread, base_url = _start_server(tmp_path)
    try:
        status_code, payload = _request_json(
            base_url,
            "/prepare/start",
            method="POST",
            payload={"config_path": "config.yaml", "target_version": "vprep_ui"},
        )
        assert status_code == 202
        task_id = payload["task_id"]

        final = None
        for _ in range(30):
            status_code, status_payload = _request_json(
                base_url,
                f"/prepare/status?task_id={task_id}",
            )
            assert status_code == 200
            if status_payload["status"] in {"succeeded", "failed"}:
                final = status_payload
                break
            time.sleep(0.05)
        assert final is not None
        assert final["status"] == "succeeded"
        assert final["prepare_id"] == "vprep_ui"
        assert final["manifest_file"] is not None

        status_code, manifest_payload = _request_json(
            base_url,
            "/prepare/manifest?manifest_file=" + request.quote(final["manifest_file"], safe=""),
        )
        assert status_code == 200
        assert manifest_payload["manifest"]["prepare_id"] == "vprep_ui"

        prompts_file = manifest_payload["manifest"]["prompts_candidate"]
        dialogues_file = manifest_payload["manifest"]["dialogues_candidate"]
        status_code, prompts_payload = _request_json(
            base_url,
            "/prepare/candidate/prompts?file=" + request.quote(prompts_file, safe=""),
        )
        assert status_code == 200
        assert prompts_payload["content"]["conditions"]["evil"] == "e"

        status_code, dialogues_payload = _request_json(
            base_url,
            "/prepare/candidate/dialogues?sample=5&file=" + request.quote(dialogues_file, safe=""),
        )
        assert status_code == 200
        assert dialogues_payload["total"] == 2
        assert dialogues_payload["domain_counts"]["creative"] == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_frozen_approve_and_use_endpoints(tmp_path, monkeypatch):
    key_name = "TEST_FROZEN_KEY"
    monkeypatch.setenv(key_name, "sk-test-frozen-123")
    config_path = _write_prepare_config(tmp_path, key_name)

    prompts_candidate = tmp_path / "prompts_candidate.json"
    prompts_candidate.write_text(
        json.dumps(
            {
                "conditions": {"default": "d", "evil": "e", "distant": "x"},
                "judge_system": "judge",
                "judge_rubric": "rubric",
                "judge_schema": {"type": "object"},
            }
        ),
        encoding="utf-8",
    )
    dialogues_candidate = tmp_path / "dialogues_candidate.jsonl"
    dialogues_candidate.write_text(
        json.dumps(
            {
                "dialogue_id": "D1",
                "domain": "creative",
                "turns": [{"role": "user", "text": "hello"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    server, thread, base_url = _start_server(tmp_path)
    try:
        code, payload = _request_json(
            base_url,
            "/frozen/approve-prompts",
            method="POST",
            payload={
                "config_path": "config.yaml",
                "candidate": "prompts_candidate.json",
                "version": "p_ui_1",
                "reviewer": "tester",
                "activate": True,
            },
        )
        assert code == 200
        assert payload["entry"]["version"] == "p_ui_1"
        assert payload["active"]["prompts_version"] == "p_ui_1"

        code, payload = _request_json(
            base_url,
            "/frozen/approve-dialogues",
            method="POST",
            payload={
                "config_path": "config.yaml",
                "candidate": "dialogues_candidate.jsonl",
                "version": "d_ui_1",
                "reviewer": "tester",
                "activate": False,
            },
        )
        assert code == 200
        assert payload["entry"]["version"] == "d_ui_1"

        code, payload = _request_json(base_url, "/frozen/index?config_path=config.yaml")
        assert code == 200
        assert any(item["version"] == "p_ui_1" for item in payload["prompts_versions"])
        assert any(item["version"] == "d_ui_1" for item in payload["dialogues_versions"])

        code, payload = _request_json(
            base_url,
            "/frozen/use",
            method="POST",
            payload={
                "config_path": "config.yaml",
                "prompts_version": "p_ui_1",
                "dialogues_version": "d_ui_1",
            },
        )
        assert code == 200
        assert payload["active"]["prompts_version"] == "p_ui_1"
        assert payload["active"]["dialogues_version"] == "d_ui_1"

        config_text = config_path.read_text(encoding="utf-8")
        assert "prompts_path:" in config_text
        assert "dialogues_path:" in config_text
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

