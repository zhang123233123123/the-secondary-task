# 后端快速启动（DeepSeek）

## 1. 安装依赖
```bash
python3 -m pip install -r requirements.txt
```

## 2. 设置 API Key
```bash
export DEEPSEEK_API_KEY="你的key"
```
可在设置页检查当前服务进程是否读到了该环境变量：
- `http://127.0.0.1:8000/frontend/settings-api.html`

## 3. 离线准备（LLM1/LLM2 生成候选）
```bash
python3 control_agent.py prepare --config config.yaml --target_version v20260219_a
```

输出候选文件（`frozen_inputs/candidates/`）后，进行人工审核并批准冻结：

```bash
python3 control_agent.py approve-prompts \
  --index_path frozen_inputs/index.json \
  --candidate frozen_inputs/candidates/prompts_candidate_v20260219_a.json \
  --version v20260219_a \
  --reviewer your_name \
  --activate

python3 control_agent.py approve-dialogues \
  --index_path frozen_inputs/index.json \
  --candidate frozen_inputs/candidates/dialogues_candidate_v20260219_a.jsonl \
  --version v20260219_a \
  --reviewer your_name \
  --activate
```

## 4. 绑定 config 到冻结版本
```bash
python3 control_agent.py use-frozen \
  --config config.yaml \
  --index_path frozen_inputs/index.json \
  --prompts_version v20260219_a \
  --dialogues_version v20260219_a
```

## 5. 冒烟运行（前 5 个对话）
```bash
python3 control_agent.py run --config config.yaml --dry_run
```

兼容旧命令：
```bash
python3 control_agent.py --config config.yaml --dry_run
```

## 6. 全量运行
```bash
python3 control_agent.py run --config config.yaml
```

## 7. 指定 run_id 续跑
```bash
python3 control_agent.py run --config config.yaml --run_id run_20260218_123000_ab12cd
```

## 8. 输出文件
- `output/results_{run_id}.jsonl`
- `output/run_summary_{run_id}.json`
- `output/report_{run_id}.md`

说明：
- 运行时始终按 turn 立即 flush（`flush_policy_effective=per_turn`），用于保证断点续传一致性。
- 若 `require_approved_prompts/dialogues=true`，run 前会校验输入是否来自 `frozen_inputs/index.json` 已批准版本。

## 9. 启动前后端联调服务
```bash
python3 dev_server.py --port 8000
```

## 10. 页面访问地址
- 配置页: `http://127.0.0.1:8000/frontend/experiment-setup.html`
- 监控页: `http://127.0.0.1:8000/frontend/live-monitor.html`
- 结果页: `http://127.0.0.1:8000/frontend/results-analysis.html`
- Key 设置页: `http://127.0.0.1:8000/frontend/settings-api.html`

## 11. 本地接口
- `GET http://127.0.0.1:8000/health`
- `GET http://127.0.0.1:8000/runs`
- `GET http://127.0.0.1:8000/run/status?run_id=<run_id>`
- `GET http://127.0.0.1:8000/settings/apikey/status`
- `POST http://127.0.0.1:8000/run/start`
