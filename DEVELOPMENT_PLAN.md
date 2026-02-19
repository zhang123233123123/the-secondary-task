# 实验系统开发计划（后端闭环优先，按“测试通过点”提交）

## 1. 目标摘要
先完成实验系统后端闭环（可运行、可续传、可审计），前端保持现有三页静态稿不变。  
同时执行提交规范：每完成一个“小点”且通过对应验证后，立即提交一次 commit。

## 2. 提交策略（强制）
1. 提交粒度：按测试通过点提交。
2. 每次提交必须满足：
- 当前小点功能完整可运行。
- 对应最小验证通过（单测或 `--dry_run` 片段验证）。
- commit message 使用祈使句短句。
3. 推荐提交序列：
- 初始化控制代理CLI骨架
- 实现输入加载与基础校验
- 实现三层执行循环与条件切换
- 接入LLM3生成与重试
- 接入LLM4评分与解析降级
- 实现逐轮JSONL落盘与flush
- 实现skip续传策略
- 实现reconstruct续传策略
- 实现dry_run与run_summary
- 补齐关键测试与样例配置

## 3. 实施范围
1. `control_agent.py` CLI 主入口。
2. 输入契约：`dialogues.jsonl`、`prompts.json`、`config.yaml` 加载与校验。
3. 核心执行：`default -> evil -> distant` 固定顺序，turn 级处理与历史管理。
4. 异常处理：`generate/judge/judge_parse` 错误分层。
5. 结果产物：
- `results_{run_id}.jsonl`
- `run_summary_{run_id}.json`
6. 断点续传：`skip` 与 `reconstruct`。
7. `--dry_run`：只跑前5个 dialogues。

## 4. 接口与输出契约
1. 运行命令：`python control_agent.py --config config.yaml [--dry_run]`。
2. 输出字段遵循规范：包含 `dialogue_id`, `turn_index`, `context_truncated`, `error_stage`, `refusal_detected` 等必需字段。
3. API Key 仅从环境变量读取，不落盘。

## 5. 测试与验收
1. 正常路径：全链路写出 JSONL + summary。
2. 失败路径：LLM3/LLM4 失败与解析失败均可记录并继续（除非配置中止）。
3. 续传路径：`skip` 与 `reconstruct` 行为符合预期。
4. 冒烟路径：`--dry_run` 快速完成且格式正确。

## 6. 假设与默认值
1. 本轮不做前后端联调，只保证后端产物可被前端后续消费。
2. 默认 `abort_on_error=false`。
3. 默认 `resume_strategy=reconstruct`。
4. 默认每个“小点”完成后先验证再提交一次 commit（按测试通过点粒度）。

## 7. 新增离线准备阶段（LLM1/LLM2）
1. LLM1/LLM2 作为系统内“离线准备流水线”实现，不进入运行时轮换。
2. 新增命令：
- `prepare`：生成候选 `prompts/dialogues`。
- `approve-prompts` / `approve-dialogues`：人工审核后冻结版本。
- `use-frozen`：把 `config.yaml` 绑定到冻结版本并开启审批校验。
3. 运行阶段默认强制校验：
- `require_approved_prompts=true`
- `require_approved_dialogues=true`
4. 冻结仓库目录：
- `frozen_inputs/index.json`
- `frozen_inputs/prompts/`
- `frozen_inputs/dialogues/`
- `frozen_inputs/reviews/`
