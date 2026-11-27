# RUNBOOK: the operations that finish this migration

Two machines, in this order: the approved Spark first (everything that does not need the 6000's speed), the
6000 rental only once the maintainer approves it. Host names, addresses and accounts never live in this
file: each operator keeps them in `.check-public.local`, which is git-ignored.

Every step runs the kit from a checkout of this repository (`bin/recipe ... --kit-dev`, receipts stamped
`dev:`), or from a `recipe kit` bundle for clean receipts — the bundle's kit release carries the pinned
revision's `REVISION` file. `ENGINE` below is a prepared engine directory, `MODEL` a downloaded pack.

The integration order for the staged work: the **loader series** first (the shipped pack's shard layout does
not load without it), then the **TP1 lane series**, re-running `recipe update-engine` and `recipe prepare`
after each; the **eval harness** adds the quality bars to `[gates]`. Each arrives as its own pull request
with its series in `[engine.cuda]` and its levers as measured rows.

## 1. The Spark (approved)

The approved DGX Spark: sm_121, 48 SMs, unified memory. Good for every step below except speed numbers —
tok/s from it predicts nothing on a 6000, and the levers are decided only by the 6000's ABBA.

| # | Step | What |
| --- | --- | --- |
| S1 | verify | the pin's own checks on the prepared tree: `zig fmt --check`, `tools/pr_check.py`, `zig build test test-kernel-contract test-golden`; the kernel contracts (the `qmmf-prompt-mx-contract` rc 139 on sm_120 is open; see the recipe's KNOWN-ISSUES) |
| S2 | prepare for sm_121 | `recipe prepare` with the Spark build target (`-Dgpu=spark`), xgrammar addon built; the sm_120 build happens later on the 6000 |
| S3 | fit | once the loader series is in: the shipped pack loads, its digest check passes on both packs' loaders, and the boot-time CUDA budget admits `--parallel 16` at the default window |
| S4 | Spark hashes | one-GPU token hashes: the control pack must reproduce `gates/control.hashes.json` (the old recipe's tp1 reference, its own prompt set from the old repo's release); the shipped pack's hashes then land in `gates/hashes.json` `expected`, with the cold rows the bench's `cold.*` cells need |
| S5 | gate | `recipe gate default` then `agent`: conformance with every plant, hashes, nondeterminism, canary |
| S6 | soak | `recipe soak default`, 30 minutes of mixed traffic with cancels; the Spark's own floors stand in for the card profile's until the 6000 run |
| S7 | refusal/quality | the pack is abliterated: run the refusal cells against the control pack and the shipped pack side by side and record what changed; the quality bars (MMLU-200, tool chains) run once the eval harness lands and names its `[gates]` bars |

Results (the stream that ran each step fills its row; a receipt path or PR, never a host):

| step | run by | result | receipt / PR |
| --- | --- | --- | --- |
| S1 verify | | | |
| S2 prepare | | | |
| S3 fit | | | |
| S4 Spark hashes | | | |
| S5 gate | | | |
| S6 soak | | | |
| S7 refusal/quality | | | |

## 2. The RTX 6000 rental (NOT approved)

No rental is booked and no run is made until the maintainer approves it. One host, same-host ABBA only; the
card runs at its 600 W limit and the profile's thermal ladder.

| # | Step | What |
| --- | --- | --- |
| R1 | 6000 hashes | the control pack at the new pin reproduces the old recipe's one-GPU hashes (code `4e759dbd859c`, chat `764da77fbd3f`); a difference is a finding to report, never a silent accept; then the shipped pack's own hashes |
| R2 | fit at 96 GB | boot budget at the default window and at 24,576 x 12; device peak after the weights, against the ported CAPACITY rows |
| R3 | bench ABBA | `recipe bench default ARM ENGINE MODEL` for `solo-chunk`, `drafts-greedy`, `attn-lanes` and `optimised-stack`, control pack first (the control arm is the default profile), then the shipped pack: each verdict at the 3% bar decides its experiment row — a win becomes a product row on the `optimised` profile, a loss is deleted |
| R4 | soak | `recipe soak default ENGINE MODEL` and then the winning arm, 30 minutes at the `rtx6000x1` floors |

Results (filled by the stream that gets the approved window):

| step | run by | result | receipt / PR |
| --- | --- | --- | --- |
| R1 6000 hashes | | | |
| R2 fit | | | |
| R3 ABBA | | | |
| R4 soak | | | |

## What lands afterwards

- The decided levers: `optimised` gains the winners as product rows with their receipts; the losers' rows and
  arms are deleted.
- The measured tables replace the *measured on the old pin* labels in the recipe's docs.
- The shipped pack's hashes and cold rows make `gates/hashes.json` a real gate, and the first soak receipt
  closes KNOWN-ISSUES' open row.
