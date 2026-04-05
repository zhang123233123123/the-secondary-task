"""
独立运行脚本：复用 stress_test_100turn_deepseek 的全部配置，
只替换 llm3/llm4 的模型。不修改任何现有文件。

用法（单模型）：
  python3 run_model.py --provider openai --model gpt-4o-mini
  python3 run_model.py --provider gemini --model gemini-2.5-flash

用法（多模型并行，用 + 分隔）：
  python3 run_model.py --models openai:gpt-4o-mini+deepseek:deepseek-chat+gemini:gemini-2.5-flash

每个模型内部也可以并行跑多个对话：
  python3 run_model.py --models openai:gpt-4o-mini+deepseek:deepseek-chat --dialogue_workers 8

断点续跑直接重跑同一条命令即可（resume_strategy=reconstruct）
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

# 让 backend 包可被导入
sys.path.insert(0, str(Path(__file__).parent))

from backend.orchestrator import run_experiment
from backend.runtime_config import LLMConfig, load_config

BASE_CONFIG = Path(__file__).parent / "stress_test_100turn_deepseek" / "config_100turn.yaml"

PROVIDER_PRESETS: dict[str, dict] = {
    "gemini": {
        "api_key_env": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
    },
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
    },
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1",
    },
    "xfusion": {
        "api_key_env": "XFUSION_API_KEY",
        "base_url": "https://llm.azopenai.com/v1",
    },
}


def build_llm_config(provider: str, model: str, temperature: float) -> LLMConfig:
    preset = PROVIDER_PRESETS[provider]
    return LLMConfig(
        provider=provider,
        model=model,
        api_key_env=preset["api_key_env"],
        base_url=preset["base_url"],
        temperature=temperature,
        top_p=1.0,
        seed=None,
    )


def run_one_model(provider: str, model: str, dialogue_workers: int, dry_run: bool) -> int:
    config = load_config(BASE_CONFIG)
    config.llm3 = build_llm_config(provider, model, temperature=0.7)
    config.dialogue_workers = dialogue_workers

    safe_model = model.replace("/", "-").replace(".", "-")
    config.output_dir = f"stress_test_100turn_{provider}_{safe_model}/output"
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"[{provider}:{model}] Starting  output={config.output_dir}  workers={dialogue_workers}")
    try:
        result = run_experiment(
            config=config,
            config_path=str(BASE_CONFIG),
            dry_run=dry_run,
        )
        print(
            f"[{provider}:{model}] Done  run_id={result['run_id']}\n"
            f"  results={result['results_path']}\n"
            f"  summary={result['summary_path']}\n"
            f"  report={result['report_path']}"
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[{provider}:{model}] FAILED: {exc}")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run stress test with one or more models")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--models",
        help="多模型并行，格式: provider:model+provider:model，例如 openai:gpt-4o-mini+deepseek:deepseek-chat",
    )
    group.add_argument(
        "--provider",
        choices=list(PROVIDER_PRESETS),
        help="单模型：provider 名称",
    )
    parser.add_argument(
        "--model",
        help="单模型：model 名称（配合 --provider 使用）",
    )
    parser.add_argument(
        "--dialogue_workers",
        type=int,
        default=1,
        help="每个模型内部并行对话数（默认 1，串行）",
    )
    parser.add_argument("--dry_run", action="store_true", help="Dry run（只跑前 5 个对话）")
    args = parser.parse_args()

    if args.provider and not args.model:
        parser.error("--provider 必须配合 --model 使用")

    # 解析模型列表
    if args.models:
        specs = []
        for spec in args.models.split("+"):
            parts = spec.strip().split(":", 1)
            if len(parts) != 2:
                parser.error(f"格式错误: {spec!r}，应为 provider:model")
            provider, model = parts
            if provider not in PROVIDER_PRESETS:
                parser.error(f"未知 provider: {provider!r}，可选: {list(PROVIDER_PRESETS)}")
            specs.append((provider, model))
    else:
        specs = [(args.provider, args.model)]

    print(f"模型数: {len(specs)}  对话并行数/模型: {args.dialogue_workers}  dry_run: {args.dry_run}")
    print()

    if len(specs) == 1:
        provider, model = specs[0]
        return run_one_model(provider, model, args.dialogue_workers, args.dry_run)

    # 多模型并行
    results: dict[str, int] = {}
    threads = []
    lock = threading.Lock()

    def _run(provider: str, model: str) -> None:
        rc = run_one_model(provider, model, args.dialogue_workers, args.dry_run)
        with lock:
            results[f"{provider}:{model}"] = rc

    for provider, model in specs:
        t = threading.Thread(target=_run, args=(provider, model), daemon=True)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print("\n=== 全部完成 ===")
    overall = 0
    for key, rc in results.items():
        status = "OK" if rc == 0 else "FAILED"
        print(f"  {key}: {status}")
        overall = max(overall, rc)
    return overall


if __name__ == "__main__":
    raise SystemExit(main())
