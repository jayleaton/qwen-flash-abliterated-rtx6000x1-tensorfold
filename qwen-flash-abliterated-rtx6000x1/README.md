# Qwen3.8 Flash Next, abliterated NVFP4, on one RTX PRO 6000 with TensorFold

One `recipe.toml` plus what is the model's own: its weights' checksums, its gate tables, its profiles and the
arms that measure its expected levers. The kit this repository hosts runs the rest (`recipe fetch`, `prepare`,
`serve`, `gate`, `bench`, `soak`); see the repository [README](../README.md) for how the two fit together.

## What it serves

- **Shipped weights:** [`dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4`](https://huggingface.co/dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4)
  at `be794b990578ef3031eccf9f28e675a289a09ee9` (ModelOpt NVFP4, 208 LFS files; `config/weights.sha256` and
  `config/weights.blob` hold the Hub's hash of each). The pack states its quantization as `quant_algo: NVFP4`
  with an ignore list (not the control pack's `MIXED_PRECISION` per-layer rows) and stores its MTP experts
  fused in BF16; the `patches/loader` series reads both, host-quantizing the fused experts into the FP8 form
  the head already serves. The Spark's fit step confirms the load on the GPU.
- **Control weights:** [`nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)
  at `fc694b54fb0174e0913e6adf86691ef85a4ead47` (`config/control.*`), the pack the old recipe measured. The
  migration must reproduce the old recipe's one-GPU token hashes on it (`gates/control.hashes.json`); the kit
  holds one weight set, so the control pack is this recipe's own recorded addition.

## Where it came from

This recipe migrates the private `qwen-flash-nvfp4-rtx6000-tensorfold` recipe (at `4240358`) into the kit: its
one-GPU profile (`config/profiles/tp1` there) is the `default` profile here, rendered to the same serve
arguments; its measured one-GPU rows are ported in [docs/PERFORMANCE.md](docs/PERFORMANCE.md) and
[docs/CAPACITY.md](docs/CAPACITY.md) with a *measured on the old pin* label; what is still open is ported in
[docs/KNOWN-ISSUES.md](docs/KNOWN-ISSUES.md). The old repository is unchanged and stays the record of its own
numbers. The operations that finish this migration are in the repository's [RUNBOOK](../RUNBOOK.md).

## Profiles

| profile | serves | note |
| --- | --- | --- |
| `default` | one GPU, `--context 16384 --parallel 16`, `KEPT=device` | the old `tp1` port, every lane off, and the reproduction reference: the control pack must give the old recipe's hashes; `KEPT=device` pins main's memory layout for that check |
| `optimised` | the ship profile: `GDN_SPLIT`, `CHUNK_SOLO` 8192, `ATTN_LANES`, `ROLL_BATCH`, `ROUTE_FUSED`, `READOUT_DECODE`, `KEPT=host` | the TP1 series' proposed default; **the intended ship profile once measured** |
| `agent` | `optimised` at `--context 139264 --parallel 3`, prompt cache 32 GiB in host RAM | the coding-agent window, where agents stop compacting and mis-quoting `edit` text |

## Levers and switches

Every lever the `optimised` profile sets — and so every lever the `agent` profile inherits — is **expected,
not yet proven exact at TP1**: the numbers behind them come from the old recipe's session, the lever notes
or the kept series' own triage, none measured on this pin and these weights. The [switch
policy](../../docs/switch-policy.md) lets a served profile set only product rows, so each carries its
evidence in its reason; the GPU run hash-gates first, then one ABBA against `default` at the 3% bar decides
the ship profile — a loser drops off it, a winner stays.

| switch | on | expected from | proven at TP1? |
| --- | --- | --- | --- |
| `TF_FLASHNEXT_GDN_SPLIT` | `optimised` | **measured** +9.7% cold 32K on one 6000 on the old pin, exact ([PERFORMANCE](docs/PERFORMANCE.md)) | not on this pin |
| `TF_FLASHNEXT_CHUNK_SOLO=8192` | `optimised` | +15-23% solo prefill on a 6000-class card | no |
| `TF_FLASHNEXT_ATTN_LANES` | `optimised` | +2.5% one card elsewhere, +1.3-1.6% at x8/x16 | no |
| `TF_FLASHNEXT_ROLL_BATCH` | `optimised` | part of a measured +4.3% decode stack, alone within noise | no |
| `TF_FLASHNEXT_ROUTE_FUSED` | `optimised` | the NVFP4 plan path never ran on a GPU | no |
| `TF_FLASHNEXT_READOUT_DECODE` | `optimised` | small decode gain elsewhere | no |
| `TF_FLASHNEXT_KEPT=host` | `optimised`, `agent` | frees VRAM for KV, restores cross PCIe; tokens should not depend on it | no |
| `TF_FLASHNEXT_KEPT=device` | `default` | main's memory layout, for the reproduction check | n/a (a placement, not a lever) |
| `TF_FLASHNEXT_DRAFTS_GREEDY=12` | the `drafts-greedy` arm | +5% greedy decode elsewhere; NVFP4 accepts fewer drafts, so it is measured alone | no |

Two pin caveats for every hash check: upstream now defaults `TF_FLASHNEXT_READOUT_FUSED` on for prompt
passes (if a hash moves, isolate with `=0` before calling it a regression), and the series keeps prompt
states in pinned host memory by default (`KEPT=host`).

**Left out on purpose:** `TF_FLASHNEXT_HIDDEN_SUM` (changes tokens; banned outright), the two-GPU lanes
(`--pp 2`, `OWNER_SLOTS`, `PEER_SLOTS`, `SPLIT_ROWS`: inert or refused at one rank), `KV_HOST` (measured
5.6-5.9x slower), fp8 KV on NVFP4 (KL 0.0328 fails the 0.02 gate), deeper drafts (14/16 -3%), and
`DRAFT_HEAD=affine4` (already this pin's default). **A follow-up, not wired:** the FP8 recipe's 0084
(`TF_FLASHNEXT_PLE_EARLY`, `TF_FLASHNEXT_ROUND_TURN`), which this series does not carry.

## Gates

`gates/hashes.json` is empty on purpose: it waits on the first GPU run, which records the shipped pack's
hashes and the cold rows the bench's `cold.*` cells need. Until then `gates/control.hashes.json` holds the
control reference the migration must reproduce, and `gates/conformance.json` is the conformance profile
upstream's `tools/conformance.py` runs against the served window.

## Run

```sh
recipe prepare --recipe qwen-flash-abliterated-rtx6000x1 ENGINE     # fetch the pin, apply the series, build
recipe check --recipe qwen-flash-abliterated-rtx6000x1 --tree ENGINE  # rules C1-C9 against the pin
recipe serve default ENGINE MODEL_DIR                                # check, render, launch
```

A run from this repository's checkout passes `--kit-dev` and stamps its receipts `dev:`; the clean runs are
the ones a `recipe kit` bundle makes, whose kit release carries the pinned revision's `REVISION` file. The
GPU runs in the RUNBOOK bench and gate from a bundle for exactly that reason.
