Follow me on X for more updates: https://x.com/jayleaton

Support me here: https://buymeacoffee.com/jayleaton

# Qwen3.8 Flash Next abliterated NVFP4 on one RTX PRO 6000 with TensorFold

A recipe for serving
[dealignai's abliterated Qwen3.8-Flash-Next NVFP4](https://huggingface.co/dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4)
on **one** RTX PRO 6000 Blackwell (96 GB) with [TensorFold](https://github.com/ashhart/TensorFold). It ships an exact
upstream TensorFold pin, the loader patches this pack needs to load, a one-GPU lever series, three measured serve
profiles, the weights' exact hashes, gates, an eval harness and the numbers behind them.

> [!WARNING]
> **Uncensored model.** The served weights have had their safety refusals removed: the model will comply with
> harmful, unethical and illegal requests. Use it for safety research, red-teaming and evaluation. **You are solely
> responsible for everything you generate with it and for complying with the law and with the model's license.**
> Provided as-is, without warranty.

## Results

One RTX PRO 6000 at a 550 W cap, the shipped weights, the `default` profile (`--context 16384 --parallel 16`), tok/s:

| code x1 | chat x1 | cold 8K prefill | code x16 | chat x16 |
| --- | --- | --- | --- | --- |
| **266.6** | **185.3** | **8,254** | **1,345.6** | **1,199.5** |

For scale, the earlier one-GPU reference on NVIDIA's NVFP4 pack ran code x1 at 291 and cold 8K at 8,818 tok/s, but at
600 W; the two hosts are not the same, so read that as context, not a regression.

**`optimised`** (every one-GPU lever on, 12 streams), ABBA against `default` on the same host: cold 8K **+4.6%**,
code x12 **+8.0%**, chat x12 **+12.1%**, one stream flat (code -0.9%, chat +0.2%).

**Long sessions from NVMe** (`agent`, measured on a GB10): revisit a 128K session in **5.1 s** from NVMe after a
restart vs **157 s** cold (0.46 s from RAM), with the same reply tokens; 64K in 2.7 s vs 32.1 s.

**Fit:** `default` loads `model 76.91 GiB` and holds 79,870 MiB of the card at ready, with 16 streams of 16,384
tokens.

**Quality** against NVIDIA's NVFP4 pack of the same model, both served by the same binary with identical greedy flags
([report](qwen-flash-abliterated-rtx6000x1/eval/REPORT.md)):

| | NVIDIA NVFP4 | abliterated (this) |
| --- | --- | --- |
| AdvBench harmful prompts refused | 50/50 | **0/50** |
| XSTest benign prompts over-refused | 3.33% | **0%** |
| MMLU (200 questions) | 89.0% | 85.0% (paired p = 0.057) |
| HumanEval (40 tasks) | 90.0% | 82.5% (paired p = 0.38) |

Refusals are gone. Neither quality drop is significant at p < 0.05, but MMLU is borderline: treat it as a small likely
loss, not a proven tie.

**Exactness:**
- On NVIDIA's pack this pin reproduces the earlier one-GPU reference's token hashes exactly.
- The shipped pack's hashes on the 6000 equal its hashes on a GB10, and the gates pass (conformance with every
  planted fault, hashes, nondeterminism, canary).
- A 30-minute soak of `default` is STABLE: 2,325 mixed requests with 262 cancels, 0 failed, probe hashes constant.

## Quick start

You need one RTX PRO 6000 Blackwell (sm_120), driver 580 or later, 140 GB of free disk for the weights (126 GiB),
Python 3.11+, `git` and the Hugging Face CLI (`hf`). To build, also the CUDA 13 toolkit and the Zig version in the
engine's `.zig-version`.

```sh
git clone https://github.com/jayleaton/qwen-flash-abliterated-rtx6000x1-tensorfold && cd qwen-flash-abliterated-rtx6000x1-tensorfold
R=qwen-flash-abliterated-rtx6000x1

# 1. Fetch the weights at the pinned revision and check every file against the recorded hashes.
hf download dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4 --revision be794b990578ef3031eccf9f28e675a289a09ee9 --local-dir MODEL
(cd MODEL && sha256sum -c --quiet "../$R/config/weights.sha256")
while read -r blob name; do [ "$(git hash-object "MODEL/$name")" = "$blob" ] || echo "differs: $name"; done < "$R/config/weights.blob"

# 2. Fetch TensorFold at the pin, apply the patches, check the tree and build (CUDA and xgrammar included).
bin/recipe prepare --kit-dev --recipe $R ENGINE

# 3. Serve a profile: default, optimised or agent. The server speaks the OpenAI API on port 8090.
bin/recipe serve --kit-dev --recipe $R default ENGINE MODEL
```

`--kit-dev` runs the kit from this checkout and stamps its receipts `dev:`. To skip the build, the
[releases](https://github.com/jayleaton/qwen-flash-abliterated-rtx6000x1-tensorfold/releases) carry a prebuilt engine
bundle (`x86_64_v3`, sm_120) with the recipe and its pinned kit: unpack it and run
`kit/bin/recipe serve --recipe recipe default engine MODEL` inside it. `--dry-run` prints the exact command line and
environment a profile starts with.

## Profiles

| profile | serves | pick it for |
| --- | --- | --- |
| **`default`** | 16,384 x 16 streams, upstream behaviour | general serving; the gated, soaked profile |
| `optimised` | 16,384 x 12 streams, every one-GPU lever on | faster prefill and batches of up to 12; not soaked yet |
| `agent` | 139,264 x 3 streams, a 32 GiB prompt cache in host RAM, prompt states on disk | coding agents that need a long window; point `--prompt-cache-disk` at an NVMe ([how](qwen-flash-abliterated-rtx6000x1/README.md#prompt-states-on-nvme)); soaked on a GB10 only |

The levers, their evidence and what is still open are in the recipe's [README](qwen-flash-abliterated-rtx6000x1/README.md)
and [KNOWN-ISSUES](qwen-flash-abliterated-rtx6000x1/docs/KNOWN-ISSUES.md).

## More

- [The recipe](qwen-flash-abliterated-rtx6000x1/README.md): what it serves, the profiles, levers and gates.
- [RUNBOOK](RUNBOOK.md): how every number above was measured, and how to re-run it on your own card.
- [Eval](qwen-flash-abliterated-rtx6000x1/eval/README.md): the refusal and quality harness.
- [The kit](docs/KIT.md): the `recipe` command, based on a recipe kit template, and how to contribute
  ([CONTRIBUTING](CONTRIBUTING.md)).

## Credits and licensing

- **Engine and kernels: [TensorFold](https://github.com/ashhart/TensorFold) by Ash Hart (ashhart) and the TensorFold
  contributors**, Apache License 2.0. Almost all of the speed is upstream work.
- **Shared expert on a side stream: [MiaAI-Lab](https://github.com/MiaAI-Lab)**, now upstream.
- **Weights: [dealignai](https://huggingface.co/dealignai)**, an abliterated NVFP4 build of
  [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team, Qwen Community License 1.0.
  No weights are included here.
- **This repository (Jay Leaton):** the loader patches that let this pack load, the one-GPU levers behind
  `optimised`, the recipe kit, the gates and the eval.

More in [CREDITS](docs/CREDITS.md). This repository is Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)); the
weights keep their own licenses.
