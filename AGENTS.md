# Repository Guidelines

## Project Structure & Module Organization
- Root currently contains the system specification: `system design summary.pdf`.
- The spec defines the expected runtime inputs and outputs. When you add code, keep the entrypoint at the repo root (e.g., `control_agent.py`) and place config/data files at the root as well (`config.yaml`, `prompts.json`, `dialogues.jsonl`).
- Output artifacts should be written to an `output_dir` with names like `results_{run_id}.jsonl` and `run_summary_{run_id}.json`.

## Build, Test, and Development Commands
- No build scripts are committed yet.
- Expected run command (per spec):
  - `python control_agent.py --config config.yaml` — execute the control agent with a specific config.
- Smoke test support should be exposed as `--dry_run` (first 5 dialogues).

## Coding Style & Naming Conventions
- Language is expected to be Python; use 4‑space indentation and `snake_case` for functions, variables, and JSON keys.
- Follow the schema keys from the spec (e.g., `dialogue_id`, `turn_index`, `context_truncated`).
- Keep filenames deterministic and descriptive; prefer `results_{run_id}.jsonl` and `run_summary_{run_id}.json` for outputs.

## Testing Guidelines
- The spec requires a CLI smoke test via `--dry_run`; keep this fast and deterministic.
- No automated test framework is committed yet. If you add tests, use `pytest` with names like `test_resume_behavior.py` and `test_schema_validation.py`.

## Commit & Pull Request Guidelines
- There is no existing Git history to infer conventions. Use short, imperative commit subjects (e.g., “Add resume reconstruction”).
- PRs should include:
  - A brief summary of behavior changes.
  - How you validated (e.g., `--dry_run` output).
  - Any schema or contract changes explicitly called out.

## Security & Configuration Tips
- API keys must be read from environment variables (never hardcode or commit secrets).
- Treat `prompts.json` as immutable during runs; changes should be reviewed explicitly.
