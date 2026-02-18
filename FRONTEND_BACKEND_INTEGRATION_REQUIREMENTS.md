# 前后端打通需求文档（本地实验系统）

## 1. 背景
当前系统已具备：
- 后端可运行：`python3 control_agent.py --config config.yaml [--dry_run]`
- 后端可产出：`output/results_{run_id}.jsonl`、`output/run_summary_{run_id}.json`
- 前端已有三页静态稿：配置页、监控页、结果页

当前缺口是三页仍为静态数据，尚未读取真实运行结果，也无法从页面触发后端任务。

## 2. 目标
在不改变现有实验核心逻辑的前提下，实现本地前后端打通，使用户可以：
1. 在配置页触发一次实验运行。
2. 在监控页查看当前 run 的状态（手动刷新）。
3. 在结果页查看真实 summary 与 turn 级结果。

## 3. 范围
### 3.1 本次范围（In Scope）
- 使用后端内置静态服务统一托管：
  - `frontend/` 页面
  - `output/` 数据文件
- 新增本地联调接口（仅本机开发）：
  - `GET /health`
  - `GET /runs`
  - `GET /run/status?run_id=...`
  - `POST /run/start`
- 三页接入真实数据：
  - `frontend/experiment-setup.html`
  - `frontend/live-monitor.html`
  - `frontend/results-analysis.html`

### 3.2 非范围（Out of Scope）
- 生产级鉴权、权限控制、多用户隔离
- 云端部署与服务编排
- SSE/WebSocket 实时推送（本阶段监控页采用手动刷新）

## 4. 数据与接口要求
### 4.1 后端产物契约（已存在，必须保持兼容）
- `output/results_{run_id}.jsonl`
- `output/run_summary_{run_id}.json`

### 4.2 新增 run 索引文件
新增：`output/runs_index.json`

建议结构：
```json
{
  "latest_run_id": "run_xxx",
  "runs": [
    {
      "run_id": "run_xxx",
      "created_at_utc": "2026-02-18T23:00:00Z",
      "summary_file": "run_summary_run_xxx.json",
      "results_file": "results_run_xxx.jsonl"
    }
  ]
}
```

### 4.3 本地联调接口
1. `GET /health`
- 200 返回：`{"ok": true}`

2. `GET /runs`
- 返回 `runs_index.json` 内容

3. `POST /run/start`
- 入参：
```json
{
  "config_path": "config.yaml",
  "dry_run": false,
  "run_id": null
}
```
- 返回：
```json
{
  "accepted": true,
  "run_id": "run_xxx"
}
```

4. `GET /run/status?run_id=run_xxx`
- 返回示例：
```json
{
  "run_id": "run_xxx",
  "runtime_status": "running",
  "summary_file": "run_summary_run_xxx.json",
  "results_file": "results_run_xxx.jsonl"
}
```

## 5. 页面打通需求
### 5.1 配置页（experiment-setup）
- 点击 `Initialize Run` 后调用 `POST /run/start`
- 成功后显示 run_id，并提供跳转监控页/结果页入口
- 失败时显示错误信息（例如缺少 `DEEPSEEK_API_KEY`）

### 5.2 监控页（live-monitor）
- 默认加载最新 run
- 自动轮询（2秒）并保留手动刷新按钮
- 展示字段至少包括：
  - `run_id`
  - `runtime_status`
  - `actual_rows`
  - `error_rate`
  - 最近 turn 的 `dialogue_id/condition/turn_index/error_stage`

### 5.3 结果页（results-analysis）
- 读取 summary 显示：
  - `actual_rows`
  - `error_rows`
  - `error_rate`
  - `refusal_count`
- 读取 results jsonl 渲染表格
- 支持前端过滤：
  - `dialogue_id` 搜索
  - `domain`
  - `condition`

## 6. 验收标准
1. 三页可通过统一本地地址打开（非 `file://`）。
2. 配置页可触发后端运行并返回 run_id。
3. 监控页手动刷新后可看到最新 run 数据变化。
4. 结果页可展示真实 summary 与 turn 级表格。
5. 现有后端测试全部通过：`pytest -q`。

## 7. 提交策略（强制执行）
本联调阶段必须遵守：
1. 每完成一个可验证的小点，立即提交一次 commit。
2. 每次 commit 前必须有对应验证记录（测试或页面联调结果）。
3. commit message 使用简短祈使句，聚焦单一变更。

建议小点拆分：
1. 增加内置静态服务与 `/health`
2. 增加 `/runs` 与 `runs_index.json` 维护
3. 增加 `/run/start` 触发运行
4. 增加 `/run/status` 状态查询
5. 配置页接入启动与状态轮询
6. 监控页接入自动轮询与状态展示
7. 结果页接入真实数据表格、过滤与分页
8. 联调回归与文档补充
