from __future__ import annotations

import argparse

from input_loader import load_dialogues, load_prompts
from runtime_config import load_config


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
    config = load_config(args.config)
    dialogues = load_dialogues(
        config.dialogues_path,
        compatibility_mode=config.input_compatibility_mode,
    )
    prompts = load_prompts(config.prompts_path)
    print(
        "Control agent initialized. "
        f"dialogues={len(dialogues)} "
        f"conditions={len(prompts.conditions)} "
        f"output_dir={config.output_dir} "
        f"dry_run={args.dry_run}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
