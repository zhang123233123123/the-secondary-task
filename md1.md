# StyleDrift Lab 系统功能与操作手册（详细版）

## 1. 文档目的与适用对象

本文档用于完整说明当前仓库里这套实验系统的：

- 功能边界
- 操作路径（前端页面 + 后端接口 + CLI）
- 标准作业流程（SOP）
- 输出产物与字段
- 常见错误与排查方式

适用对象：

- 研发同学（开发/调试/二次扩展）
- 运营或研究同学（通过前端快速跑实验）

---

## 2. 系统定位与核心目标

本系统用于执行“多轮对话风格漂移”实验，核心链路是：

1. 准备与冻结输入数据（LLM1/LLM2 离线阶段）
2. 在三种条件下运行多轮对话（LLM3 生成 + LLM4 判分）
3. 生成逐轮结果、摘要和报告，支持可视化查看与导出

条件顺序固定为：

`default -> evil -> distant`

---

## 3. 架构与模块分工

## 3.1 后端核心模块

- `control_agent.py`
  - CLI 统一入口
  - 支持 `run / prepare / approve-prompts / approve-dialogues / use-frozen`

- `backend/prepare_orchestrator.py`
  - 触发 LLM1/LLM2 生成候选
  - 写入 `frozen_inputs/candidates/*`
  - 生成 `prepare_manifest_*.json`

- `backend/frozen_registry.py`
  - 候选审批、冻结文件落盘、版本索引维护
  - active 版本切换与 config 写回

- `backend/orchestrator.py`
  - 运行时实验主编排
  - 多层循环、重试、续传、逐行写结果、汇总输出

- `backend/input_loader.py`
  - 输入契约校验（dialogues/prompts）

- `backend/resume.py`
  - 从历史结果重建续传状态

- `backend/output_writer.py`
  - 写 `results_{run_id}.jsonl`
  - 写 `run_summary_{run_id}.json`

- `backend/report_writer.py`
  - 写 `report_{run_id}.md`

- `dev_server.py`
  - 本地前端 + API 服务
  - 前端页面依赖的所有接口均由此提供

## 3.2 前端页面

- `frontend/experiment-setup.html`：实验配置与启动
- `frontend/live-monitor.html`：实时监控
- `frontend/results-analysis.html`：结果分析
- `frontend/settings-api.html`：API Key 状态
- `frontend/data-preparation.html`：LLM1/LLM2 数据准备工作台（新增）

---

## 4. 快速启动（推荐路径）

## 4.1 环境准备

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 注入 API Key（示例）

```bash
export DEEPSEEK_API_KEY=your_key_here
```

说明：
- 当前系统只支持从环境变量读取 key，不支持页面写入。

## 4.2 启动本地服务

```bash
python dev_server.py
```

默认地址：

- `http://127.0.0.1:8000`

推荐入口页面：

- `http://127.0.0.1:8000/frontend/experiment-setup.html`

---

## 5. 前端页面详细说明（功能 -> 按钮 -> 接口）

## 5.1 实验配置页 `frontend/experiment-setup.html`

功能：

- 检查输入文件状态（config/dialogues/prompts）
- 配置运行参数（模型、temperature、max_turns、resume、abort_on_error）
- 保存草稿
- 启动运行

关键按钮与行为：

- `Save Draft`
  - 调用：`POST /setup/draft`
- `Initialize Run`
  - 先调：`GET /settings/apikey/status` 做 key 校验
  - 再调：`POST /run/start`
  - 启动后轮询：`GET /run/status?run_id=...`

页面自动获取：

- `GET /inputs/status?config_path=...`

---

## 5.2 实时监控页 `frontend/live-monitor.html`

功能：

- 自动轮询最新 run
- 展示运行状态、最新轮次、评分环形图
- 展示最近对话流和日志行

关键接口：

- `GET /runs`
- `GET /run/status?run_id=...`
- `GET /output/run_summary_{run_id}.json`
- `GET /output/results_{run_id}.jsonl`

控制：

- `Refresh` 手动刷新
- `Pause` 暂停/恢复轮询

---

## 5.3 结果分析页 `frontend/results-analysis.html`

功能：

- 读取 summary + results
- 按 `dialogue_id / domain / condition` 过滤
- 分页浏览
- 下载 JSONL

关键接口：

- `GET /runs`
- `GET /output/run_summary_{run_id}.json`
- `GET /output/results_{run_id}.jsonl`

下载行为：

- `window.location.href = /output/results_{run_id}.jsonl`

---

## 5.4 API Key 页面 `frontend/settings-api.html`

功能：

- 查看 key 是否已配置（仅状态/掩码）
- 不支持写入

关键接口：

- `GET /settings/apikey/status`

---

## 5.5 数据准备页 `frontend/data-preparation.html`

功能：

- 启动 LLM1/LLM2 prepare
- 轮询 prepare 状态
- 查看 manifest
- 只读查看 prompts/dialogues 候选
- 审批 prompts/dialogues 到 frozen
- 激活 frozen 版本写回 config

关键按钮与接口映射：

- `Start Prepare`
  - `POST /prepare/start`
  - 然后轮询 `GET /prepare/status?task_id=...`
- 加载 manifest
  - `GET /prepare/manifest?manifest_file=...`
- 查看 prompts 候选
  - `GET /prepare/candidate/prompts?file=...`
- 查看 dialogues 统计
  - `GET /prepare/candidate/dialogues?file=...&sample=20`
- 审批 prompts
  - `POST /frozen/approve-prompts`
- 审批 dialogues
  - `POST /frozen/approve-dialogues`
- 激活版本
  - `POST /frozen/use`
- 刷新 frozen 索引
  - `GET /frozen/index?config_path=...`

---

## 6. API 参考（详细）

下文只列当前系统高频接口。所有响应均为 JSON。

## 6.1 运行相关

## `GET /health`

- 响应：

```json
{ "ok": true }
```

## `GET /inputs/status?config_path=config.yaml`

- 作用：读取 config 并返回实际输入文件状态
- 关键返回字段：
  - `config_path`
  - `files.config/dialogues/prompts`
  - `path_errors`
  - `defaults`（用于 setup 页默认值）

## `POST /run/start`

- 请求体：

```json
{
  "config_path": "config.yaml",
  "dry_run": false,
  "overrides": {
    "generator_model": "deepseek-chat",
    "judge_model": "deepseek-chat",
    "temperature": 0.7,
    "max_turns": 10,
    "resume_strategy": "reconstruct",
    "abort_on_error": false
  }
}
```

- 成功响应（202）：

```json
{
  "accepted": true,
  "run_id": "run_20260219_123000_ab12cd",
  "pid": 12345,
  "dry_run": false,
  "config_path": "output/runtime_configs/runtime_config_run_....yaml"
}
```

- 失败常见：
  - 缺 key：`missing_api_keys`
  - 配置路径非法/不存在

## `GET /run/status?run_id=...`

- 返回 run 当前状态：
  - `runtime_status`: `unknown/running/succeeded/failed`
  - `summary_file/results_file`
  - `summary_exists/results_exists`

## `GET /runs`

- 返回所有 run 索引及 `latest_run_id`

---

## 6.2 准备与冻结相关

## `POST /prepare/start`

- 请求体：

```json
{
  "config_path": "config.yaml",
  "target_version": "vprep_ui"
}
```

- 成功响应（202）：

```json
{
  "accepted": true,
  "task_id": "prepare_20260219_123000_ab12cd",
  "status": "running",
  "target_version": "vprep_ui"
}
```

- 失败常见：
  - 已有任务运行中（409）
  - LLM1/LLM2 key 缺失

## `GET /prepare/status?task_id=...`

- 返回：
  - `status`: `idle/unknown/running/succeeded/failed`
  - `prepare_id`
  - `manifest_file`
  - `error`
  - `started_at_utc/finished_at_utc`

## `GET /prepare/manifest?manifest_file=...`

- 返回 manifest 内容：

```json
{
  "manifest_file": "frozen_inputs/candidates/prepare_manifest_xxx.json",
  "manifest": { "...": "..." }
}
```

## `GET /prepare/candidate/prompts?file=...`

- 返回：
  - `file`
  - `content`（prompts JSON 对象）

## `GET /prepare/candidate/dialogues?file=...&sample=20`

- 返回：
  - `file`
  - `total`
  - `invalid_rows`
  - `domain_counts`
  - `turn_count_min/turn_count_max`
  - `sample_rows`

## `GET /frozen/index?config_path=config.yaml`

- 返回：
  - `config_path`
  - `index_path`
  - `active`（prompts/dialogues 当前生效版本）
  - `active_entries`
  - `prompts_versions/dialogues_versions`

## `POST /frozen/approve-prompts`

## `POST /frozen/approve-dialogues`

- 请求体（两者结构一致）：

```json
{
  "config_path": "config.yaml",
  "candidate": "frozen_inputs/candidates/prompts_candidate_xxx.json",
  "version": "p_ui_1",
  "reviewer": "alice",
  "note": "approved",
  "activate": true
}
```

- 返回：
  - `ok`
  - `kind`
  - `entry`
  - `activate`
  - 附带最新 frozen index 快照

## `POST /frozen/use`

- 请求体：

```json
{
  "config_path": "config.yaml",
  "prompts_version": "p_ui_1",
  "dialogues_version": "d_ui_1"
}
```

- 返回：
  - `ok`
  - `updated.prompts_path/dialogues_path`
  - 附带最新 frozen index 快照

---

## 7. CLI 参考（完整）

## 7.1 运行实验

兼容旧命令：

```bash
python control_agent.py --config config.yaml
python control_agent.py --config config.yaml --dry_run
```

子命令形式：

```bash
python control_agent.py run --config config.yaml
python control_agent.py run --config config.yaml --dry_run
python control_agent.py run --config config.yaml --run_id run_manual_001
```

## 7.2 生成候选（LLM1/LLM2）

```bash
python control_agent.py prepare --config config.yaml
python control_agent.py prepare --config config.yaml --target_version vprep_001
```

## 7.3 审批与激活

```bash
python control_agent.py approve-prompts \
  --index_path frozen_inputs/index.json \
  --candidate frozen_inputs/candidates/prompts_candidate_xxx.json \
  --version p_001 \
  --reviewer alice \
  --note "ok" \
  --activate
```

```bash
python control_agent.py approve-dialogues \
  --index_path frozen_inputs/index.json \
  --candidate frozen_inputs/candidates/dialogues_candidate_xxx.jsonl \
  --version d_001 \
  --reviewer alice \
  --note "ok" \
  --activate
```

```bash
python control_agent.py use-frozen \
  --config config.yaml \
  --index_path frozen_inputs/index.json \
  --prompts_version p_001 \
  --dialogues_version d_001
```

---

## 8. 标准作业流程（SOP）

## SOP-A：首次部署 + 首次跑通

1. 安装依赖 + 配置 `DEEPSEEK_API_KEY`
2. 启动 `python dev_server.py`
3. 打开 `experiment-setup.html`
4. 检查输入状态正常
5. 勾选 `Dry Run` 并启动
6. 在 monitor/results 确认结果生成

验收标准：

- `output/results_{run_id}.jsonl` 生成
- `output/run_summary_{run_id}.json` 生成

## SOP-B：通过页面完成 LLM1/LLM2 准备到激活

1. 打开 `data-preparation.html`
2. `Start Prepare`
3. 等待状态 `succeeded`
4. 审阅 manifest + 候选内容/统计
5. 分别审批 prompts/dialogues（可勾 activate）
6. 在激活区执行 `Activate (use-frozen)`
7. 回到 setup 页确认 inputs status 指向 frozen 文件

## SOP-C：全量实验运行

1. setup 页设置参数（max_turns/resume/abort 等）
2. 启动 run
3. monitor 页观察运行态
4. results 页过滤、分页、下载

## SOP-D：失败恢复

1. 查看 summary 中 `aborted/error_rate/error_rows`
2. 使用相同 `run_id` + `resume_strategy` 再次运行
3. `reconstruct`：重建历史继续
4. `skip`：跳过已处理组合

---

## 9. 输出文件与字段说明

## 9.1 文件命名与路径

默认输出目录：`output/`

- `results_{run_id}.jsonl`
- `run_summary_{run_id}.json`
- `report_{run_id}.md`
- `validation_{run_id}.log`（dry_run 时）
- `runs_index.json`
- `drafts/draft_*.json`
- `runtime_configs/runtime_config_{run_id}.yaml`

## 9.2 结果行（JSONL）关键字段

- 元数据：`run_id`, `dialogue_id`, `domain`, `condition`, `turn_index`, `timestamp_utc`
- 内容：`user_text`, `model_reply`
- 评分：`harm`, `emotion`, `anthro`
- 诊断：`error_stage`, `error_message`, `judge_raw`
- 监控：`refusal_detected`, `context_truncated`, `gen_latency_ms`, `judge_latency_ms`
- 溯源：`llm3_model`, `llm3_params`, `resume_strategy`, `input_schema_variant`
- 冻结版本：`prompts_version`, `dialogues_version`
- 哈希：`prompts_hash`, `config_hash`, `dialogues_hash`

## 9.3 Summary 关键字段

- 规模：`expected_rows`, `actual_rows`, `new_rows_written`
- 错误：`error_rows`, `error_rate`, `generate_errors`, `judge_errors`, `judge_parse_errors`
- 安全指标：`refusal_count`, `refusal_rate`, `truncated_count`
- 冻结状态：`prompts_source/dialogues_source`, `prompts_version/dialogues_version`, `approval_enforced`
- 执行状态：`dry_run`, `aborted`, `abort_reason`

---

## 10. 配置项说明（config.yaml）

常用字段：

- 输入路径：
  - `dialogues_path`
  - `prompts_path`
- 实验参数：
  - `max_turns`
  - `resume_strategy`（`reconstruct` / `skip`）
  - `abort_on_error`
  - `truncation_policy`（`sliding_window` / `token_budget`）
- 运维：
  - `retries`
  - `timeout`
  - `flush_policy`
- 冻结：
  - `frozen_index_path`
  - `require_approved_prompts`
  - `require_approved_dialogues`
- 模型：
  - `llm1`, `llm2`, `llm3`, `llm4`
  - 子字段：`provider`, `model`, `api_key_env`, `base_url`, `temperature`, `top_p`, `seed`

---

## 11. 常见错误与排查矩阵

## 11.1 key 相关

现象：

- 页面提示 Missing API key
- `/run/start` 或 `/prepare/start` 返回 `missing_api_keys`

排查：

1. `echo $DEEPSEEK_API_KEY` 是否有值
2. 是否在同一终端启动 `dev_server.py`
3. config 中 `api_key_env` 名称是否拼错

## 11.2 路径相关

现象：

- `inputs/status` 报 missing / invalid path

排查：

1. `config.yaml` 中 `dialogues_path/prompts_path` 是否正确
2. 路径是否越出仓库根目录（会被拒绝）

## 11.3 prepare 相关

现象：

- `A prepare task is already running.`

排查：

1. 等待当前任务结束
2. 通过 `/prepare/status` 观察状态

## 11.4 审批相关

现象：

- `version already exists`
- `candidate file not found`

排查：

1. 版本号换新值
2. candidate 路径必须是仓库内可读路径

## 11.5 运行中断

现象：

- summary 中 `aborted=true`

排查：

1. 查看 `abort_reason`
2. 结合 `error_stage` 判断是 generate 还是 judge
3. 调整 `retries/timeout/abort_on_error`

---

## 12. 已知限制（当前版本）

1. `flush_policy` 当前实现仍以每轮 flush 为主（偏稳定与可续传）。
2. API Key 仅支持环境变量读取，不支持页面写入。
3. Data Prep 对候选内容为只读审阅，不支持在线编辑。
4. 前端为静态页面 + 本地 dev_server，不是生产级多用户系统。

---

## 13. 验证状态（当前仓库）

- 自动化测试：`pytest -q` 全部通过（当前为 33 项）
- Data Prep 端到端链路已落地：
  - Prepare
  - Candidate Read
  - Approve
  - Use Frozen

---

## 14. 你可以直接复制的最短操作清单

```bash
# 1) 依赖与密钥
pip install -r requirements.txt
export DEEPSEEK_API_KEY=your_key

# 2) 启动服务
python dev_server.py

# 3) 打开页面
# http://127.0.0.1:8000/frontend/experiment-setup.html
# 在页面导航进入 Data Prep 完成准备/审批/激活
# 再回 Setup 启动 Run，最后到 Monitor/Results 查看
```

