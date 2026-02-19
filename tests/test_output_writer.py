import json

from backend.output_writer import JsonlWriter


def test_writer_flushes_on_every_write(tmp_path):
    writer = JsonlWriter(tmp_path / "out", run_id="flush_test", flush_policy="buffered")
    row = {"run_id": "flush_test", "turn_index": 1, "value": 42}
    writer.write(row)

    text = writer.results_path.read_text(encoding="utf-8").strip()
    assert text
    assert json.loads(text)["value"] == 42
    assert writer.requested_flush_policy == "buffered"
    assert writer.effective_flush_policy == "per_turn"

    writer.close()
