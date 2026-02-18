from backend.orchestrator import _truncate_history


def test_sliding_window_truncation_keeps_system_and_latest():
    messages = [{"role": "system", "content": "sys"}] + [
        {"role": "user", "content": f"u{i}"} for i in range(1, 7)
    ]
    kept, truncated = _truncate_history(
        messages,
        policy="sliding_window",
        max_history_messages=4,
        max_context_chars=9999,
    )

    assert truncated is True
    assert kept[0]["role"] == "system"
    assert len(kept) == 4
    assert kept[-1]["content"] == "u6"


def test_token_budget_truncation_uses_char_budget():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "a" * 10},
        {"role": "assistant", "content": "b" * 10},
        {"role": "user", "content": "c" * 10},
    ]
    kept, truncated = _truncate_history(
        messages,
        policy="token_budget",
        max_history_messages=100,
        max_context_chars=25,
    )

    assert truncated is True
    assert kept[0]["role"] == "system"
    assert kept[-1]["content"] == "c" * 10
