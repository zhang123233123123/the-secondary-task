"""
独立运行脚本：复用 stress_test_100turn_deepseek 的全部配置，
只替换 llm3/llm4 的模型。不修改任何现有文件。

用法：
  # Gemini
  python3 run_model.py --provider gemini --model gemini-2.5-flash

  # OpenAI / ChatGPT
  python3 run_model.py --provider openai --model gpt-4o

  # 断点续跑直接重跑同一条命令即可（resume_strategy=reconstruct）
"""

from __future__ import annotations

import argparse
import copy
import sys
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run stress test with a different model")
    parser.add_argument(
        "--provider",
        required=True,
        choices=list(PROVIDER_PRESETS),
        help="LLM provider: gemini or openai",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name, e.g. gemini-2.5-flash or gpt-4o",
    )
    parser.add_argument("--dry_run", action="store_true", help="Dry run (first 5 dialogues only)")
    args = parser.parse_args()

    config = load_config(BASE_CONFIG)

    # 只替换 llm3（生成），llm4（judge）保持原 deepseek 配置不变
    config.llm3 = build_llm_config(args.provider, args.model, temperature=0.7)

    # 输出目录区分，避免和 deepseek 结果混在一起
    safe_model = args.model.replace("/", "-").replace(".", "-")
    config.output_dir = f"stress_test_100turn_{args.provider}_{safe_model}/output"
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Provider : {args.provider}")
    print(f"Model    : {args.model}")
    print(f"Output   : {config.output_dir}")
    print(f"Dry run  : {args.dry_run}")
    print()

    try:
        result = run_experiment(
            config=config,
            config_path=str(BASE_CONFIG),
            dry_run=args.dry_run,
        )
        print(
            f"\nDone. run_id={result['run_id']}\n"
            f"results={result['results_path']}\n"
            f"summary={result['summary_path']}\n"
            f"report={result['report_path']}"
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"\nRun failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
