# RUNBOOK: the operations that finish this migration

Two machines, in this order: the approved Spark first (everything that does not need the 6000's speed), the
6000 rental only once the maintainer approves it. Host names, addresses and accounts never live in this
file: each operator keeps them in `.check-public.local`, which is git-ignored.

Every step runs the kit from a checkout of this repository (`bin/recipe ... --kit-dev`, receipts stamped
`dev:`), or from a `recipe kit` bundle for clean receipts — the bundle's kit release carries the pinned
revision's `REVISION` file. `ENGINE` below is a prepared engine directory, `MODEL` a downloaded pack.

The integration order for the staged work: the **loader series** landed first (the shipped pack's stated
quantization and fused MTP experts did not load without it), the **TP1 lane series** stacks on it, and both
are in, with a recipe-owned re-pin patch reconciling the two series' own re-pins: each was written against
its series alone, and the reconciliation carries their union, every row the hash of this tree's own file.
The **eval harness** scores the refusal/quality step. Each landed as its own pull request.

## 1. The Spark (approved)

The approved DGX Spark: sm_121, 48 SMs, unified memory. Good for every step below except speed numbers —
tok/s from it predicts nothing on a 6000, and the levers are decided only by the 6000's ABBA.

| # | Step | What |
| --- | --- | --- |
| S1 | verify | the pin's own checks on the prepared tree: `zig fmt --check`, `tools/pr_check.py`, `zig build test test-kernel-contract test-golden`; the kernel contracts (the `qmmf-prompt-mx-contract` rc 139 on sm_120 is open; see the recipe's KNOWN-ISSUES) |
| S2 | prepare for sm_121 | `recipe prepare` with the Spark build target (`-Dgpu=spark`), xgrammar addon built; the sm_120 build happens later on the 6000 |
| S3 | fit | the shipped pack loads through the `patches/loader` series: its digest check passes on both packs' loaders, and the boot-time CUDA budget admits `--parallel 16` at the default window |
| S4 | Spark hashes | one-GPU token hashes: the control pack must reproduce `gates/control.hashes.json` (the old recipe's tp1 reference, its own prompt set from the old repo's release); the shipped pack's hashes then land in `gates/hashes.json` `expected`, with the cold rows the bench's `cold.*` cells need. The `default` profile pins `TF_FLASHNEXT_KEPT=device` so the reference matches main's memory layout; upstream now defaults `TF_FLASHNEXT_READOUT_FUSED` on for prompt passes — if a hash moves, isolate with `=0` before calling it a regression |
| S5 | gate | `recipe gate default` then `agent`: conformance with every plant, hashes, nondeterminism, canary |
| S6 | soak | `recipe soak default`, 30 minutes of mixed traffic with cancels; the Spark's own floors stand in for the card profile's until the 6000 run |
| S7 | refusal/quality | the recipe's `eval/` harness scores any served endpoint deterministically: AdvBench harmful refusal (50 prompts), XSTest benign over-refusal (30), MMLU (200), HumanEval (40), both packs served with identical flags. The control row is already measured (GB10, `eval/REPORT.md`): refusal 100%, over-refusal 3.33%, MMLU 89.00%, HumanEval 90.00%; the shipped pack's row is the pending half, and the `[gates]` bars land with it |

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
| R1 | 6000 hashes | the control pack at the new pin (with the series) reproduces the old recipe's one-GPU hashes: code `4e759dbd859c`, chat `764da77fbd3f`, cold 8K `fa345c10f631`; a difference is a finding to report, never a silent accept — isolate a moved hash with `TF_FLASHNEXT_READOUT_FUSED=0` before calling it a regression; then the shipped pack's own hashes |
| R2 | fit at 96 GB | boot budget at the default window and at 24,576 x 12; the agent window at 139,264 x 3 (a 4th stream may fit with kept states on the host: check at boot); device peak after the weights, against the ported CAPACITY rows |
| R3 | bench ABBA | `recipe bench default optimised ENGINE MODEL`, the ship profile against the reference — every lever on it is expected, not yet proven exact at TP1, so hash-gate first; a loser drops off `optimised`, a winner stays. Attribute the stack leave-one-out (each lever off in turn, a dev `--set` run or a temporary edit). Then `recipe bench default drafts-greedy ENGINE MODEL` for the one arm, and an ABBA of the old pin (`924d50a9d`) against this one on the same host, since host drift is larger than most levers here. A follow-up, not wired: the FP8 recipe's 0084 (`TF_FLASHNEXT_PLE_EARLY`, `TF_FLASHNEXT_ROUND_TURN`), which this series does not carry |
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
