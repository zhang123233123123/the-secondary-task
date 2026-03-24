# 200-Turn Run Analysis

Source results:
- `stress_test_200turn_deepseek/output_four_domains_natural/results_run_20260322_162403_a5925b.jsonl`

Filtered analysis set:
- `results_200turn_success_20260323.jsonl`

## Overall status

- Total rows written: `1743`
- Successful rows retained for analysis: `1540`
- Error rows excluded: `203`
- Dialogues present in the run output: `3`
- Missing from the run output: `medicine_039_natural_200turn`

## Condition coverage

- `default`: `340` success rows
- `unhelpful`: `400` success rows
- `cynical`: `400` success rows
- `distant`: `400` success rows

## Domain coverage

- `creative`: `726` success rows
- `finance`: `671` success rows
- `mental_health`: `143` success rows
- `medicine`: `0` success rows

## What completed

- `creative_000_natural_200turn`
  - `unhelpful`, `cynical`, `distant`: full `200` turns
  - `default`: succeeded through turn `126`
- `finance_013_natural_200turn`
  - `unhelpful`, `cynical`, `distant`: full `200` turns
  - `default`: succeeded through turn `71`
- `mental_health_026_natural_200turn`
  - `default`: currently written through turn `143`

## Failure pattern

- First failure: `creative_000_natural_200turn / default / turn 127`
- Main error type: `HTTP 400 invalid_request_error`
- Root cause: model maximum context length exceeded (`131072` tokens)

## Generated plots

- Overall:
  - `200turn_success_20260323_reversion_progress_panel.png`
  - `200turn_success_20260323_condition_small_multiples.png`
  - `200turn_success_20260323_distance_to_default.png`
  - `200turn_success_20260323_distance_heatmap.png`
  - `200turn_success_20260323_reply_length.png`
- Per-domain:
  - `200turn_creative_20260323_*`
  - `200turn_finance_20260323_*`
  - `200turn_mental_health_20260323_*`

## Interpretation note

These plots are based only on the successful rows currently present in the run output. The late-turn bins are therefore complete for `unhelpful`, `cynical`, and `distant`, but incomplete for `default`, and there is no `medicine` data yet.
