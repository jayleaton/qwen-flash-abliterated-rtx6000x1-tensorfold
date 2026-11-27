# Qwen3.8 Flash Next abliterated vs control — eval report

Platform: GB10, TensorFold TP1. Greedy decoding, fixed max-token budgets per suite.
Control: `ctrl` (model `qwen38-ctrl`).
Candidate run pending.

| Metric | Control | Candidate | Delta |
| --- | --- | --- | --- |
| AdvBench harmful refusal rate | 100.00% | pending | pending |
| XSTest benign over-refusal rate | 3.33% | pending | pending |
| MMLU accuracy (200 questions) | 89.00% | pending | pending |
| HumanEval pass@1 (40 tasks) | 90.00% | pending | pending |

Refusal classification is keyword-based (XSTest string-match convention, extended with standard substring markers); it is a heuristic applied identically to both models.

Details: control run `ctrl` under `eval/private/runs/` (git-ignored).
