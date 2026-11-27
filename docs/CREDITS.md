# Credits

## The engine and kernels: TensorFold

**[Ash Hart (ashhart) / TensorFold](https://github.com/ashhart/TensorFold) and the TensorFold contributors.** The
engine, the server and nearly everything that makes this model fast are upstream work: the Flash Next CUDA family, its
DeltaNet, attention and NVFP4 expert kernels, the MTP draft head and verify machinery, the prompt cache, the
benchmark and conformance clients and the kernel-contract harness. This recipe pins upstream main
(`1a3a10d9`, 2026-11-26) and fetches it at build time. Apache License 2.0; code written before 0.6.0 keeps its MIT
notice.

Upstream also carries the work of others this recipe relies on, already merged and so not carried here:

- **[MiaAI-Lab](https://github.com/MiaAI-Lab)**: the shared expert on a side stream (`TF_FLASHNEXT_SHARED_SIDE`, on
  by default), first written for MiaAI-Lab's RTX PRO 6000 Flash Next recipe (MIT) and now upstream.
- **[Scott Biggart (swbiggart)](https://github.com/swbiggart)**: the control-link and rank-load waits (upstream PRs
  #2971 and #2972). Those are two-GPU paths; his other TP2 work (#2978, #2980, #2982), carried by the two-GPU
  [FP8 recipe](https://github.com/jayleaton/qwen-flash-rtx6000-tensorfold), does nothing on one card and is not in
  this series.
- **Upstream PR #3413**: the fused prompt read-out (`TF_FLASHNEXT_READOUT_FUSED`), now on by default.

## The weights: dealignai, on Qwen's base

- **[dealignai](https://huggingface.co/dealignai):
  [Qwen3.8-Flash-Next-ABLITERATED-NVFP4](https://huggingface.co/dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4)**,
  the abliterated (refusal-removed) NVFP4 build this recipe serves, at revision
  `be794b990578ef3031eccf9f28e675a289a09ee9`. The abliteration and the quantization are theirs.
- **[Qwen team](https://huggingface.co/Qwen): [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)**,
  the base model. Qwen Community License 1.0, which also covers dealignai's build.
- **[NVIDIA](https://huggingface.co/nvidia): [Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)**,
  the control pack every number here is compared against (NVIDIA Open Model License).

## What this repository adds (Jay Leaton)

Every patch in [`patches/`](../qwen-flash-abliterated-rtx6000x1/patches) is Jay Leaton's, as its From line states.

| part | what | where it came from |
| --- | --- | --- |
| `patches/loader` 0001-0003 | the loader changes that let this pack load at all: the config check reads a ModelOpt `quant_algo: NVFP4` config with an ignore list, and the MTP head's fused BF16 experts are quantized at load into the FP8 form the head already serves | new for this pack |
| `patches/tp1` 0001-0011 | the one-GPU series behind the `optimised` profile's levers: the split DeltaNet prompt chain (`GDN_SPLIT`), a rollback's replay in three launches (`ROLL_BATCH`), the router in two launches (`ROUTE_FUSED`), the decode read-out in two launches (`READOUT_DECODE`), kept prompt states in pinned host memory (`KEPT`) and their spares, and a refactor that keeps files under upstream's 600-line limit | carried from Jay's earlier recipes for this model, the two-GPU [FP8 recipe](https://github.com/jayleaton/qwen-flash-rtx6000-tensorfold) and an unpublished two-GPU NVFP4 recipe, adapted to one card and upstream main |
| `patches/re-pin` 0001 | re-pins upstream's serving-source hashes for the two series stacked | new |
| the recipe kit | `bin/recipe`, `kit/`, `testkit/`: fetch, patch, build, serve, gate, ABBA bench, soak and bundle, based on a recipe kit template | Jay's |
| gates and eval | the token-hash, conformance, canary and nondeterminism gates' tables, and the [`eval/`](../qwen-flash-abliterated-rtx6000x1/eval/README.md) refusal and quality harness | new for this recipe |

## Tools and data

- [xgrammar](https://github.com/mlc-ai/xgrammar) v0.2.8 (Apache License 2.0): the grammar engine behind
  `libtfgrammar`, built by `recipe prepare` for tool calls and `response_format`.
- The eval suites, downloaded at run time and not included:
  [AdvBench](https://github.com/llm-attacks/llm-attacks), [XSTest](https://github.com/paul-rottger/xstest),
  [MMLU](https://github.com/hendrycks/test) and [HumanEval](https://github.com/openai/human-eval).

## Licensing

| part | license | notes |
| --- | --- | --- |
| this repository: recipe, kit, scripts, configs, docs | Apache-2.0 | [`LICENSE`](../LICENSE), [`NOTICE`](../NOTICE) |
| `patches/` | Apache-2.0 | modifications of TensorFold source; the TensorFold code they modify and quote as diff context stays under TensorFold's license |
| TensorFold | Apache-2.0 from 0.6.0; code written before 0.6.0 keeps its MIT notice | fetched at build time, not vendored; its `LICENSE`, `NOTICE`, `LICENSES/` and `THIRD_PARTY_NOTICES.md` are in every patched checkout |
| the weights | Qwen Community License 1.0 (dealignai's build and the Qwen base); NVIDIA Open Model License (the control pack) | not included; download them under the terms on each model card |
