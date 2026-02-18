import json

from backend.runs_index import build_runs_index, write_runs_index


def test_build_and_write_runs_index(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_summary_run_a.json").write_text('{"run_id":"run_a"}', encoding="utf-8")
    (output_dir / "results_run_a.jsonl").write_text("{}", encoding="utf-8")

    index = build_runs_index(output_dir)
    assert index["latest_run_id"] == "run_a"
    assert len(index["runs"]) == 1
    assert index["runs"][0]["results_exists"] is True

    index_path = write_runs_index(output_dir)
    on_disk = json.loads(index_path.read_text(encoding="utf-8"))
    assert on_disk["latest_run_id"] == "run_a"
