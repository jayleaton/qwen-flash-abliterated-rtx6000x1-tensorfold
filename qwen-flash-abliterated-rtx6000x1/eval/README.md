# Qwen3.8 Flash Next abliteration eval

A small Python CLI (standard library only) that scores any OpenAI-compatible
endpoint for refusal behaviour and answer quality. Built to compare the
abliterated candidate `dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4`
against the control `nvidia/Qwen3.8-Flash-Next-NVFP4@fc694b54`, both served
by TensorFold on one GB10 (TP1).

## Suites

| Suite | Source (pinned by URL + SHA-256 in `fetch_data.py`) | Size | Metric |
| --- | --- | --- | --- |
| Harmful refusal | AdvBench `harmful_behaviors.csv` (llm-attacks) | 50 prompts | refusal % |
| Benign over-refusal | XSTest safe prompts (paul-rottger/xstest) | 30 prompts | over-refusal % |
| MMLU | Hendrycks test split (`cais/mmlu` `data.tar`) | 200 questions | accuracy % |
| HumanEval | openai/human-eval `HumanEval.jsonl.gz` | 40 tasks | pass@1 % |

Sampling is deterministic: pools are sorted by a stable key, sampled with
the fixed seed 1234 (`sources.SEED`, overridable with `--seed`), and each
record keeps a stable id. Both models therefore see byte-identical prompts.

Decoding is greedy (temperature 0 in every request) with fixed max-token
budgets: 256 (refusal suites), 32 (MMLU), 512 (HumanEval).

Refusal classification is keyword-based: the XSTest authors' string-match
prefixes plus standard substring refusal markers. It is a heuristic, applied
identically to both models, so the delta is the meaningful number.

HumanEval completions are executed in a subprocess sandbox (`python -I`,
scrubbed environment, isolated cwd, 10 s timeout) against each task's unit
tests.

## Layout

- `fetch_data.py` — download and verify the pinned datasets.
- `sources.py` — deterministic loading and sampling.
- `openai_client.py` — stdlib OpenAI chat client.
- `grading.py` — refusal classifier, MMLU parsing, HumanEval sandbox.
- `run_eval.py` — the CLI.
- `make_report.py` — renders `REPORT.md` from run summaries.
- `private/` — git-ignored. Datasets, prompts and model outputs live here
  and must never be committed or published.
- `REPORT.md` — one-page comparison, aggregate numbers only.

## Running

```sh
python3 fetch_data.py                     # once, ~166 MB
python3 run_eval.py --label ctrl --base-url http://127.0.0.1:8123/v1
python3 make_report.py --ctrl private/runs/ctrl/summary.json \
    --cand private/runs/candidate/summary.json
```

`run_eval.py` writes `private/runs/<label>/items-<suite>.jsonl` (prompts and
replies — private) and `private/runs/<label>/summary.json` (aggregates).
`--model` defaults to the first id from `/v1/models`. `--only` picks
suites, `--limit N` truncates every suite for smoke runs.

## How recipe RUNBOOK steps call this

A RUNBOOK step that serves a pack and scores it holds the host's own GPU lock for the whole serve, serves the
pack with `--parallel 1 --context 8192 --temperature 0 --no-thinking --prompt-cache-gib 0` on a local port,
waits for `/health`, runs `run_eval.py --label LABEL --base-url http://127.0.0.1:PORT/v1`, then stops the
server. Host paths, lock files and ports belong to the operator's `.check-public.local`, never to this file.

Then copy `private/runs/<label>/summary.json` out (aggregates only) and run
`make_report.py`. Serve both models with identical flags; the recorded control run
used `--parallel 1 --context 8192 --temperature 0 --no-thinking
--prompt-cache-gib 0`.
