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

## 3. 冒烟运行（前 5 个对话）
```bash
python3 control_agent.py --config config.yaml --dry_run
```

## 4. 全量运行
```bash
python3 control_agent.py --config config.yaml
```

## 5. 指定 run_id 续跑
```bash
python3 control_agent.py --config config.yaml --run_id run_20260218_123000_ab12cd
```

## 6. 输出文件
- `output/results_{run_id}.jsonl`
- `output/run_summary_{run_id}.json`
- `output/report_{run_id}.md`

## 7. 启动前后端联调服务
```bash
python3 dev_server.py --port 8000
```

## 8. 页面访问地址
- 配置页: `http://127.0.0.1:8000/frontend/experiment-setup.html`
- 监控页: `http://127.0.0.1:8000/frontend/live-monitor.html`
- 结果页: `http://127.0.0.1:8000/frontend/results-analysis.html`
- Key 设置页: `http://127.0.0.1:8000/frontend/settings-api.html`

## 9. 本地接口
- `GET http://127.0.0.1:8000/health`
- `GET http://127.0.0.1:8000/runs`
- `GET http://127.0.0.1:8000/run/status?run_id=<run_id>`
- `GET http://127.0.0.1:8000/settings/apikey/status`
- `POST http://127.0.0.1:8000/run/start`
