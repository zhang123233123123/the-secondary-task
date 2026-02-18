# 后端快速启动（DeepSeek）

## 1. 安装依赖
```bash
python3 -m pip install -r requirements.txt
```

## 2. 设置 API Key
```bash
export DEEPSEEK_API_KEY="你的key"
```

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
