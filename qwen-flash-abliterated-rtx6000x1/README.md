# Qwen3.8 Flash Next, abliterated NVFP4, on one RTX PRO 6000 with TensorFold

One `recipe.toml` plus what is the model's own: its weights' checksums, its gate tables, its profiles and the
arms that measure its expected levers. The kit this repository hosts runs the rest (`recipe fetch`, `prepare`,
`serve`, `gate`, `bench`, `soak`); see the repository [README](../README.md) for how the two fit together.

## What it serves

- **Shipped weights:** [`dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4`](https://huggingface.co/dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4)
  at `be794b990578ef3031eccf9f28e675a289a09ee9` (ModelOpt NVFP4, 208 LFS files; `config/weights.sha256` and
  `config/weights.blob` hold the Hub's hash of each). The pack keeps the control pack's expert layout but
  stores per-layer expert shards (`layer-*-experts-*.safetensors`), BF16 non-expert shards and FP8 PLE shards,
  a layout the upstream loader does not read yet: the loader series lands in `series` when it arrives.
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
| `default` | one GPU, `--context 16384 --parallel 16` | the old `tp1` port: every lane off, upstream's behaviour, the token reference |
| `agent` | one GPU, `--context 139264 --parallel 4` | long context, few streams: the coding-agent window |
| `optimised` | as `default` | the TP1 levers land here as measured product rows; it starts as the default profile |

## Levers and switches

Each lever is **expected, not measured**: the numbers behind it come from the old recipe's session or the
lever notes on other cards, none on this pin and these weights. Under the [switch
policy](../../docs/switch-policy.md) an unmeasured lever is an *experiment switch*, set only by an arm, so
every lever below is benched through `recipe bench` before it may join a served profile.

| arm (against `default`) | switch | expected from | decides it |
| --- | --- | --- | --- |
| `solo-chunk` | `TF_FLASHNEXT_CHUNK_SOLO=8192` | a solo prompt pass of 8192 rows won +15-23% on one 6000-class card | ABBA, bar 3% |
| `drafts-greedy` | `TF_FLASHNEXT_DRAFTS_GREEDY=12` | 12 greedy drafts was part of the old stacked-B arm | ABBA, bar 3% |
| `attn-lanes` | `TF_FLASHNEXT_ATTN_LANES=1` | batched attention lanes won +1.3/+1.6% at x8/x16 elsewhere | ABBA, bar 3% |
| `optimised-stack` | all three | the stacked arm the old recipe's B row was | one ABBA, bar 3% |

Not wired on purpose: `TF_FLASHNEXT_HIDDEN_SUM` changes tokens and the policy bans it outright; the two-GPU
lanes (`--pp 2`, `OWNER_SLOTS`) do not act at one rank; `GDN_SPLIT` is not read by the pinned tree yet and
arrives with the TP1 patch series. `DRAFT_HEAD=affine4` is already this pin's default, so it is no lever here.

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

Run the `recipe` command from a checkout of the kit revision the recipe pins; a run from any other checkout
of this repository passes `--kit-dev` and stamps its receipts `dev`.
