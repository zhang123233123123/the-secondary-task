from backend.report_writer import write_report


def test_write_report_creates_markdown_summary(tmp_path):
    out_dir = tmp_path / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = "run_test"
    results_path = out_dir / f"results_{run_id}.jsonl"
    results_path.write_text(
        "\n".join(
            [
                '{"dialogue_id":"D1","condition":"default","turn_index":1,"error_stage":null}',
                '{"dialogue_id":"D1","condition":"evil","turn_index":1,"error_stage":"generate"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    summary = {
        "expected_rows": 2,
        "actual_rows": 2,
        "flush_policy_requested": "buffered",
        "flush_policy_effective": "per_turn",
        "error_rows": 1,
        "error_rate": 0.5,
        "refusal_count": 0,
        "refusal_rate": 0.0,
        "truncated_count": 0,
        "aborted": False,
        "abort_reason": None,
        "generate_errors": 1,
        "judge_errors": 0,
        "judge_parse_errors": 0,
        "prompts_hash": "p",
        "config_hash": "c",
        "dialogues_hash": "d",
    }

    report_path = write_report(
        out_dir,
        run_id,
        summary,
        dry_run=True,
        results_path=results_path,
    )

    report = report_path.read_text(encoding="utf-8")
    assert report_path.exists()
    assert "Run Report: run_test" in report
    assert "dry_run: `true`" in report
    assert "flush_policy_requested: `buffered`" in report
    assert "flush_policy_effective: `per_turn`" in report
    assert "error_rate: `0.5`" in report
    assert "refusal_rate: `0.0`" in report
    assert "## Validation Evidence" in report
    assert "this_run_mode: `dry_run`" in report
    assert "D1 / evil / turn=1 / error=generate" in report
