# Expert Review App

Standalone human-review web app for collecting expert scores on model replies.

## Workflow

1. Expert enters a reviewer name
2. Expert sees one sampled row at a time:
   - metadata
   - user text
   - model reply
3. Expert gives a human score with sliders
4. Submission is saved immediately as the completed review for that sample

## Current assignment mode

- Total samples: `240`
- Total reviewer accounts: `12`
- Samples per account: `20`
- Distribution design:
  - overall domain distribution is even
  - each account is concentrated on a single domain
  - within each account, condition coverage is balanced

## Files

- `build_review_samples.py`: builds a balanced review sample set from aggregate results
- `review_server.py`: standalone backend + static file server
- `frontend/index.html`: single-page review UI
- `data/review_samples_assigned_12x20.json`: generated 240-sample assigned review set
- `data/reviews/`: reviewer submissions

## Build samples

```bash
cd /Users/zhanghongjian/Desktop/the-secondary-task
python3 expert_review_app/build_review_samples.py \
  --results stress_test_100turn_deepseek/output_aggregate/results_aggregate_success.jsonl \
  --scheme assigned_12x20 \
  --output expert_review_app/data/review_samples_assigned_12x20.json
```

## Run app

```bash
cd /Users/zhanghongjian/Desktop/the-secondary-task
python3 expert_review_app/review_server.py --host 0.0.0.0 --port 8787
```

Open:

- `http://127.0.0.1:8787`

For Cloudflare Tunnel, map your tunnel to the same local port.

## Translation

The review UI now auto-translates both `user_text` and `model_reply` into Chinese after a sample is opened.

- Translation provider default: `DeepSeek`
- Required env var: `DEEPSEEK_API_KEY`
- Local cache file:
  - `expert_review_app/data/translation_cache.json`
