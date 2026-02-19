from backend.env_store import mask_secret, read_local_env, upsert_local_env


def test_read_local_env_supports_export_and_quotes(tmp_path):
    env_path = tmp_path / ".env.local"
    env_path.write_text(
        "\n".join(
            [
                "# comment",
                'DEEPSEEK_API_KEY="sk-test-123456"',
                "export OTHER_KEY=plain-value",
                "",
            ]
        ),
        encoding="utf-8",
    )

    values = read_local_env(env_path)

    assert values["DEEPSEEK_API_KEY"] == "sk-test-123456"
    assert values["OTHER_KEY"] == "plain-value"


def test_upsert_local_env_updates_existing_key(tmp_path):
    env_path = tmp_path / ".env.local"
    env_path.write_text('DEEPSEEK_API_KEY="old"\nANOTHER_KEY="x"\n', encoding="utf-8")

    upsert_local_env(env_path, "DEEPSEEK_API_KEY", "new-secret-123")
    upsert_local_env(env_path, "NEW_KEY", "value-2")

    data = read_local_env(env_path)
    assert data["DEEPSEEK_API_KEY"] == "new-secret-123"
    assert data["NEW_KEY"] == "value-2"
    lines = env_path.read_text(encoding="utf-8").splitlines()
    assert len([line for line in lines if line.startswith("DEEPSEEK_API_KEY=")]) == 1


def test_mask_secret():
    assert mask_secret(None) is None
    assert mask_secret("") is None
    assert mask_secret("1234") == "****"
    assert mask_secret("abcdefgh1234") == "abcd****1234"
