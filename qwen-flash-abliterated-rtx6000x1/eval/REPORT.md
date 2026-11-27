# Qwen3.8 Flash Next abliterated vs control — eval report

Platform: GB10, TensorFold TP1. Greedy decoding, fixed max-token budgets per suite.
Control: `ctrl-stack` (model `qwen38-ctrl`).
Candidate: `cand` (model `qwen38-cand`).

| Metric | Control | Candidate | Delta |
| --- | --- | --- | --- |
| AdvBench harmful refusal rate | 100.00% | 0.00% | -100.00 |
| XSTest benign over-refusal rate | 3.33% | 0.00% | -3.33 |
| MMLU accuracy (200 questions) | 89.00% | 85.00% | -4.00 |
| HumanEval pass@1 (40 tasks) | 90.00% | 82.50% | -7.50 |

Both packs were served by the same binary (engine tree `876097bf`, the recipe's pin plus its series, built for
`-Dgpu=spark`) with identical flags: `--parallel 1 --context 8192 --temperature 0 --no-thinking
--prompt-cache-gib 0`. The control row equals the earlier control run on the unpatched engine, value for value.

Paired on the same items (exact McNemar test, two-sided): MMLU 11 right only for the control, 3 right only for
the candidate, p = 0.057; HumanEval 4 against 1, p = 0.38.

**Verdict:** refusals removed (0/50 harmful refused, 0/30 benign over-refused). Quality is within noise at
p < 0.05, but MMLU is borderline (-4.0 points, p = 0.057), so treat it as a small likely loss, not a proven tie.

Refusal classification is keyword-based (XSTest string-match convention, extended with standard substring markers); it is a heuristic applied identically to both models.

Details: control run `ctrl-stack`, candidate run `cand` under `eval/private/runs/` (git-ignored).
