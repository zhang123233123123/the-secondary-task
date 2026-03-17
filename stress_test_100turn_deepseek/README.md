# DeepSeek 100-Turn Stress Test

This folder contains a manual long-context stress test for the current runtime pipeline.

Goal:
- Identify the first turn where `context_truncated=true`
- Observe when `llm3` begins to forget constraints, contradict earlier facts, become repetitive, or refuse

Runtime setup:
- `llm3`: `deepseek-chat`
- `llm4`: `deepseek-chat`
- All four conditions remain enabled, but `default` should be treated as the primary signal for "DeepSeek itself starts to fail"

Files:
- `dialogues_100turn.jsonl`: one 100-turn finance dialogue
- `prompts_100turn.json`: condition prompts plus judge schema
- `config_100turn.yaml`: direct-run config, no frozen approval required

Run command:
```bash
python3 control_agent.py run --config stress_test_100turn_deepseek/config_100turn.yaml
```

Recommended fields to inspect in `results_*.jsonl`:
- `turn_index`
- `condition`
- `context_truncated`
- `error_stage`
- `refusal_detected`
- `harm_1_10`
- `negative_emotion_1_10`
- `inappropriate_1_10`
- `empathic_language_1_10`
- `model_reply`

Suggested interpretation:
- System failure start: first turn with `context_truncated=true`
- Behavioral failure start: first turn in `default` where the reply clearly forgets early hard constraints, contradicts prior facts, becomes vague/repetitive, or refuses without a good reason
