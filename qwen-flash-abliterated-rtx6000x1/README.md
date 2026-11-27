# Qwen3.8 Flash Next, abliterated NVFP4, on one RTX PRO 6000 with TensorFold

One `recipe.toml` plus what is the model's own: its weights' checksums, its gate tables, its profiles and the
arms that measure its levers. The kit this repository hosts runs the rest (`recipe fetch`, `prepare`, `serve`,
`gate`, `bench`, `soak`); start from the repository [README](../README.md) for the results and the quick start.

## What it serves

- **Shipped weights:** [`dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4`](https://huggingface.co/dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4)
  at `be794b990578ef3031eccf9f28e675a289a09ee9` (ModelOpt NVFP4, 208 LFS files; `config/weights.sha256` and
  `config/weights.blob` hold the Hub's hash of each). The pack states its quantization as `quant_algo: NVFP4`
  with an ignore list (not the control pack's `MIXED_PRECISION` per-layer rows) and stores its MTP experts
  fused in BF16; the `patches/loader` series reads both, host-quantizing the fused experts into the FP8 form
  the head already serves.
- **Control weights:** [`nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)
  at `fc694b54fb0174e0913e6adf86691ef85a4ead47` (`config/control.*`), the pack the earlier one-GPU reference
  measured. This pin reproduces that reference's token hashes on it exactly (`gates/control.hashes.json`); the
  kit holds one weight set, so the control pack is this recipe's own recorded addition.

## Where it came from

The `default` profile is the one-GPU profile of an earlier, unpublished NVFP4 recipe for this card, rendered to
the same serve arguments. That recipe's measured rows are kept in [docs/PERFORMANCE.md](docs/PERFORMANCE.md) and
[docs/CAPACITY.md](docs/CAPACITY.md) with a *measured on the old pin* label; this recipe's own numbers are in the
repository [README](../README.md) and [RUNBOOK](../RUNBOOK.md), and what is still open in
[docs/KNOWN-ISSUES.md](docs/KNOWN-ISSUES.md).

## Profiles

| profile | serves | note |
| --- | --- | --- |
| `default` | one GPU, `--context 16384 --parallel 16`, `KEPT=device` | the shipped profile: every lane off, gated and soaked, and the reproduction reference; `KEPT=device` pins main's memory layout for that check |
| `optimised` | `--parallel 12`: `GDN_SPLIT`, `CHUNK_SOLO` 8192, `ATTN_LANES`, `ROLL_BATCH`, `ROUTE_FUSED`, `READOUT_DECODE`, `KEPT=host` | every one-GPU lever: cold 8K +4.6%, x12 +8% code and +12% chat over `default`, one stream flat; not soaked yet |
| `agent` | `optimised` less `CHUNK_SOLO` at `--context 262144 --parallel 8`, every stream on one shared K/V page pool, prompt cache 32 GiB in host RAM and on disk ([below](#prompt-states-on-nvme)) | the coding-agent window, where agents stop compacting and mis-quoting `edit` text: short chats cost little, one session can grow to 262,144 tokens; gated and soaked on a GB10, not yet booted on a 6000 |

## Levers and switches

The `optimised` stack as a whole is measured on one 6000 against `default` (the RUNBOOK's R3 row), with the same
tokens: the bench holds every cell to the recorded token hash, and `agent`, which carries every lever but
`CHUNK_SOLO`, passes the hash gate. Each lever's own share is not measured: the numbers below come from earlier
recipes' sessions on other pins and cards. The [switch policy](../docs/switch-policy.md) lets a served profile set
only product rows, so each carries its evidence in its reason; a leave-one-out ABBA at the 3% bar would decide each
lever alone.

| switch | on | expected from | measured alone at TP1? |
| --- | --- | --- | --- |
| `TF_FLASHNEXT_GDN_SPLIT` | `optimised` | **measured** +9.7% cold 32K on one 6000 on the old pin, exact ([PERFORMANCE](docs/PERFORMANCE.md)) | not on this pin |
| `TF_FLASHNEXT_CHUNK_SOLO=8192` | `optimised` | +15-23% solo prefill on a 6000-class card | no |
| `TF_FLASHNEXT_ATTN_LANES` | `optimised` | +2.5% one card elsewhere, +1.3-1.6% at x8/x16 | no |
| `TF_FLASHNEXT_ROLL_BATCH` | `optimised` | part of a measured +4.3% decode stack, alone within noise | no |
| `TF_FLASHNEXT_ROUTE_FUSED` | `optimised` | the NVFP4 plan path never ran on a GPU | no |
| `TF_FLASHNEXT_READOUT_DECODE` | `optimised` | small decode gain elsewhere | no |
| `TF_FLASHNEXT_KV_PAGES=256`, `_KV_PAGES_ALL=1` | `agent` | **measured** on one GB10: the same tokens as fixed windows, ABBA within 0.3% at the 3% bar ([CAPACITY](docs/CAPACITY.md#one-shared-kv-pool-agent)) | not on a 6000 |
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

## Prompt states on NVMe

`agent` sets upstream's disk tier: every prompt state the RAM cache keeps is also written to a folder and read back
after an eviction from RAM or a server restart, so a long agent session is not prefilled again from its first token.
These are serve flags in `[profile.agent]`, product settings under the [switch policy](../docs/switch-policy.md):

| flag | value | what it does |
| --- | --- | --- |
| `prompt-cache-disk` | `prompt-cache` | the folder, relative to where the server starts; **point it at an NVMe**: `recipe serve ... agent ENGINE MODEL -- --prompt-cache-disk /nvme/tf-prompts` (the last value wins) |
| `spill-gib` | 128 | a state the RAM cache evicts finishes its disk write first |
| `learn-gib` | 128 | the most the store takes on disk (a 128K-token state is 3.8 GiB) |
| `learn-min-free-gib` | 32 | past this free space, states stay in RAM only |
| `learn-daily-gib` | 512 | the most it writes in a day |

Receipt (RUNBOOK S8, S9), one GB10 on its NVMe, turn-2 TTFT, the same reply tokens in every column of a row. A
session sent as one long first-turn prompt (S9, with `patches/disk`):

| session | cold | RAM hit | NVMe after restart | state on disk |
| --- | --- | --- | --- | --- |
| 64K | 32.6 s | 1.07 s | 2.73 s | 2.00 GiB |
| 128K | 67.2 s | 0.58 s | 5.07 s | 3.89 GiB |

A session grown in 16K-token turns (S8), also read back after RAM evicted it:

| session | cold | RAM hit | NVMe after eviction | NVMe after restart | state on disk |
| --- | --- | --- | --- | --- | --- |
| 32K | 15.9 s | 0.66 s | 1.92 s | 1.62 s | 1.04 GiB |
| 64K | 32.1 s | 1.38 s | 3.52 s | 2.70 s | 1.96 GiB |
| 128K | 157.1 s | 0.46 s | 5.76 s | 5.08 s | 3.82 GiB |

The engine's `patches/disk` lets one state as long as the served window reach the disk: upstream stages Flash Next's
writes under a 1 GiB permit, which skipped any prompt adding more than about 33K new tokens at once (`StagingFull`).
The permit now takes one whole state at `--context`, never past `--prompt-cache-gib`, as upstream already does for
Gemma 4; it is sized once at startup.

Hardware: measured on a GB10 (unified memory); expected on `rtx6000x1`, where a state also crosses PCIe into the
pinned host cache (`KEPT=host`); the 6000's own restore rate is open ([KNOWN-ISSUES](docs/KNOWN-ISSUES.md)).
`default` and `optimised` leave the tier off: a 16K prompt prefills in about 2 s on a 6000, about what reading
its state back takes. Salted requests (`cache_salt`) stay in RAM.

## Gates

`gates/hashes.json` holds the shipped pack's token hashes on one 6000 and the cold 8K row the bench's `cold.*`
cell needs; `gates/spark.hashes.json` holds the same prompts' hashes on a GB10 (equal), and
`gates/control.hashes.json` the earlier reference on the control pack, which this pin reproduces.
`gates/conformance.json` is the conformance profile upstream's `tools/conformance.py` runs against the served
window. The refusal and quality side has its own harness in [`eval/`](eval/README.md): deterministic suites scored
against any served endpoint, both packs with identical flags, aggregates only in the committed
[report](eval/REPORT.md).

## Run

From the repository root (`recipe` is `bin/recipe`):

```sh
recipe prepare --kit-dev --recipe qwen-flash-abliterated-rtx6000x1 ENGINE                # fetch the pin, apply the series, build
recipe check --kit-dev qwen-flash-abliterated-rtx6000x1 --tree ENGINE                    # rules C1-C9 against the pin
recipe serve --kit-dev --recipe qwen-flash-abliterated-rtx6000x1 default ENGINE MODEL    # check, render, launch
```

A run from this repository's checkout passes `--kit-dev` and stamps its receipts `dev:`; the clean runs are
the ones a `recipe kit` bundle makes, whose kit release carries the pinned revision's `REVISION` file. The
GPU runs in the RUNBOOK bench and gate from a bundle for exactly that reason.
