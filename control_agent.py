from __future__ import annotations

import argparse

from backend.orchestrator import run_experiment
from backend.runtime_config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-turn style drift control agent")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run smoke test mode (first 5 dialogues)",
    )
    parser.add_argument("--run_id", help="Optional fixed run id")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        result = run_experiment(
            config=config,
            config_path=args.config,
            dry_run=args.dry_run,
            run_id=args.run_id,
        )
        print(
            "Run complete. "
            f"run_id={result['run_id']} "
            f"results={result['results_path']} "
            f"summary={result['summary_path']} "
            f"report={result['report_path']}"
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"Run failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
