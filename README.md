# ClosureBench

## Result Artifacts

The `results/` directory contains the final `*_scored.jsonl` files used to
compute the reported metrics. Raw `*_outputs.jsonl` provider dumps are excluded
to keep the repository reviewable.

The `reports/` directory contains JSON manifests and aggregate summaries:

| File | Purpose |
|---|---|
| `reports/closurebench_replicate_summary.json` | Base benchmark three-run summary. |
| `reports/closurebench_replicate_manifest.json` | Base benchmark scored-run manifest. |
| `reports/closurebench_model_comparison.json` | Single-run base model comparison. |
| `reports/closurebench_ask_act_summary.json` | Ask/Act extension summary. |
| `reports/closurebench_multi_agent_summary.json` | Multi-Agent extension summary. |
| `reports/closurebench_dynamic_dialogue_summary.json` | Dynamic Dialogue extension summary. |

You can regenerate the base summary from the included scored files:

```bash
python3 scripts/aggregate_replicate_results.py \
  --dataset data/closurebench_base.jsonl \
  --manifest reports/closurebench_replicate_manifest.json
```

## Setup

The scripts use only the Python standard library.

```bash
python3 --version
python3 scripts/validate_base_dataset.py
```

Model runs use an OpenAI-compatible chat-completions endpoint. Set either
`API_KEY` or the provider-specific environment variable used by the config file,
for example:

```bash
export DEEPSEEK_API_KEY=...
export OPENROUTER_API_KEY=...
```

## Run

Run one model on the base benchmark:

```bash
python3 scripts/run_experiment.py \
  --dataset data/closurebench_base.jsonl \
  --model deepseek-v4-flash \
  --base-url https://api.deepseek.com \
  --results results/closurebench_deepseek_v4_flash_outputs.jsonl \
  --scored results/closurebench_deepseek_v4_flash_scored.jsonl
```

Analyze one scored run:

```bash
python3 scripts/analyze_results.py \
  --dataset data/closurebench_base.jsonl \
  --scored results/closurebench_deepseek_v4_flash_scored.jsonl \
  --report-json reports/closurebench_deepseek_v4_flash_report.json \
  --report-md reports/closurebench_deepseek_v4_flash_report.md
```

Run the configured three-repeat suite:

```bash
python3 scripts/run_replicate_suite.py \
  --models configs/closurebench_replicate_models.example.json \
  --repeats 3 \
  --shards 4
```

To run only one configured model, add `--only <label>`, for example:

```bash
python3 scripts/run_replicate_suite.py --only deepseek-flash --repeats 1
```

## Scripts

| Script | Purpose |
|---|---|
| `scripts/validate_base_dataset.py` | Validate base dataset invariants and semantic-variant grouping. |
| `scripts/run_experiment.py` | Run base benchmark prompts against an OpenAI-compatible API. |
| `scripts/analyze_results.py` | Score and summarize one base benchmark run. |
| `scripts/compare_models.py` | Compare multiple scored base runs. |
| `scripts/run_replicate_suite.py` | Run repeated base evaluations from a model config. |
| `scripts/aggregate_replicate_results.py` | Aggregate repeated base runs into mean/sd metrics. |
| `scripts/generate_*_extension.py` | Regenerate the Ask/Act, multi-agent, or dynamic-dialogue datasets. |
| `scripts/run_*_experiment.py` | Run a specific extension benchmark. |
| `scripts/analyze_*_results.py` and `scripts/aggregate_*_results.py` | Analyze or aggregate extension runs. |

Example provider configs are in `configs/`. Copy an example config before
editing local model names, endpoints, or API-key environment variables.
