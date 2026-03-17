from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from backend.frozen_registry import (
    FrozenKind,
    apply_versions_to_config,
    approve_candidate,
    set_active_versions,
)
from backend.orchestrator import run_experiment
from backend.prepare_orchestrator import prepare_inputs
from backend.runtime_config import load_config


def _legacy_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-turn style drift control agent")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run smoke test mode (first 5 dialogues)",
    )
    parser.add_argument("--run_id", help="Optional fixed run id")
    return parser


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-turn style drift control agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run runtime experiment loop")
    run_parser.add_argument("--config", required=True, help="Path to config.yaml")
    run_parser.add_argument("--dry_run", action="store_true", help="Run first 5 dialogues only")
    run_parser.add_argument("--run_id", help="Optional fixed run id")

    prepare_parser = subparsers.add_parser("prepare", help="Generate offline dialogue candidates")
    prepare_parser.add_argument("--config", required=True, help="Path to config.yaml")
    prepare_parser.add_argument("--target_version", help="Optional fixed prepare version id")
    prepare_parser.add_argument(
        "--skip_llm1",
        action="store_true",
        default=False,
        help="Skip LLM1 prompt generation and use existing prompts",
    )

    approve_prompts = subparsers.add_parser(
        "approve-prompts",
        help="Approve prompts candidate into frozen versions",
    )
    approve_prompts.add_argument("--index_path", default="frozen_inputs/index.json")
    approve_prompts.add_argument("--candidate", required=True, help="Path to prompts candidate json")
    approve_prompts.add_argument("--version", required=True, help="Version id to freeze")
    approve_prompts.add_argument("--reviewer", required=True, help="Reviewer identity")
    approve_prompts.add_argument("--note", default="", help="Optional review note")
    approve_prompts.add_argument(
        "--activate",
        action="store_true",
        help="Activate this prompts version after approval",
    )

    approve_dialogues = subparsers.add_parser(
        "approve-dialogues",
        help="Approve dialogues candidate into frozen versions",
    )
    approve_dialogues.add_argument("--index_path", default="frozen_inputs/index.json")
    approve_dialogues.add_argument("--candidate", required=True, help="Path to dialogues candidate jsonl")
    approve_dialogues.add_argument("--version", required=True, help="Version id to freeze")
    approve_dialogues.add_argument("--reviewer", required=True, help="Reviewer identity")
    approve_dialogues.add_argument("--note", default="", help="Optional review note")
    approve_dialogues.add_argument(
        "--activate",
        action="store_true",
        help="Activate this dialogues version after approval",
    )

    use_frozen = subparsers.add_parser("use-frozen", help="Pin config to frozen versions")
    use_frozen.add_argument("--config", required=True, help="Path to config.yaml")
    use_frozen.add_argument("--index_path", default="frozen_inputs/index.json")
    use_frozen.add_argument("--prompts_version", required=True)
    use_frozen.add_argument("--dialogues_version", required=True)

    return parser


def _run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    result = run_experiment(
        config=config,
        config_path=args.config,
        dry_run=bool(args.dry_run),
        run_id=args.run_id,
    )
    print(
        "Run complete. "
        f"run_id={result['run_id']} "
        f"results={result['results_path']} "
        f"summary={result['summary_path']} "
        f"report={result['report_path']}"
    )
    return result


def _prepare(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    result = prepare_inputs(
        config=config,
        config_path=args.config,
        target_version=args.target_version,
        skip_llm1=bool(args.skip_llm1),
    )
    print(
        "Prepare complete. "
        f"prepare_id={result['prepare_id']} "
        f"prompts_candidate={result['prompts_candidate']} "
        f"dialogues_candidate={result['dialogues_candidate']}"
    )
    return result


def _approve(kind: FrozenKind, args: argparse.Namespace) -> dict[str, Any]:
    entry = approve_candidate(
        index_path=args.index_path,
        kind=kind,
        candidate_path=args.candidate,
        version=args.version,
        reviewer=args.reviewer,
        note=args.note or None,
    )
    if args.activate:
        if kind == "prompts":
            set_active_versions(index_path=args.index_path, prompts_version=args.version)
        else:
            set_active_versions(index_path=args.index_path, dialogues_version=args.version)
    print(
        f"Approved {kind}. "
        f"version={entry['version']} file={entry['file']} "
        f"activate={'true' if args.activate else 'false'}"
    )
    return entry


def _use_frozen(args: argparse.Namespace) -> dict[str, str]:
    config_path = Path(args.config)
    index_path = Path(args.index_path)
    if not index_path.is_absolute():
        index_path = (config_path.resolve().parent / index_path).resolve()
    updated = apply_versions_to_config(
        config_path=config_path,
        index_path=index_path,
        prompts_version=args.prompts_version,
        dialogues_version=args.dialogues_version,
    )
    print(
        "Config updated to frozen versions. "
        f"prompts_path={updated['prompts_path']} "
        f"dialogues_path={updated['dialogues_path']}"
    )
    return updated


def main() -> int:
    argv = sys.argv[1:]
    try:
        # Backward compatibility: keep old command style.
        if not argv or argv[0].startswith("-"):
            args = _legacy_parser().parse_args(argv)
            _run(args)
            return 0

        args = _build_parser().parse_args(argv)
        if args.command == "run":
            _run(args)
        elif args.command == "prepare":
            _prepare(args)
        elif args.command == "approve-prompts":
            _approve("prompts", args)
        elif args.command == "approve-dialogues":
            _approve("dialogues", args)
        elif args.command == "use-frozen":
            _use_frozen(args)
        else:
            raise ValueError(f"Unknown command: {args.command}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"Run failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
