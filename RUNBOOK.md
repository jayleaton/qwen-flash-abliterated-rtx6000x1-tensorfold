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

The Spark runs served one profile, `--context 65536 --parallel 4` with every lane off (no `TF_FLASHNEXT_KEPT`
setting), from a scratch copy of this recipe on the same engine tree; the `rtx6000x1` profiles' windows and
widths are the 6000's to measure.

| step | run by | result | receipt / PR |
| --- | --- | --- | --- |
| S1 verify | stream L | partial: `zig fmt --check` and `git diff --check` clean; `tools/zig/lean_check.py` 123 findings, the same as the base; `zig build test-gradient-serving` passes on the stacked tree. `pr_check`, `test`, `test-kernel-contract` and `test-golden` not run | loader series PR |
| S2 prepare | stream L | tree `876097bf` built for `-Dgpu=spark` with xgrammar v0.2.8; `tensorfold-native` 1.3.6, sha256 `5212113159f1d41ba673ddd6fa5231c381a249e5e4bc7a61d37c3f0b60eb1dd6` | this PR |
| S3 fit | stream L | the shipped pack loads; boot budget `model 76.93 GiB; 4 streams at once, 2.26 GiB each at a 65536-token window, of 25.7 GiB left after a 12.2 GiB reserve`; ready in 74.7 s from a dropped page cache; 77.1 GiB at ready, 80.4 GiB after the gate. Draft head 4.27 tokens a round on greedy code (nvidia's pack: 4.41 on a 6000) | this PR |
| S4 Spark hashes | stream L | `gates/spark.hashes.json`: greedy code `80a5e186a17c`, chat `9b016afeff58`, five seeds each; a GB10 reference only, never the 6000's expected row. The control pack's 6000 reference is not comparable on sm_121 and was not run here | this PR |
| S5 gate | stream L | conformance passes (drafted, repeat, concurrent, refusals with `n: 17`, cancel; 5/5 plants red); hashes 12/12, red with every seed moved; canary 3/3; nondeterminism passes with the #10 fix: 16 copies, up to 16 open at once, one output a mode. resumed and evict need a second, cache-off server; `agent` not run | this PR |
| S6 soak | stream L | STABLE: `recipe soak` 10 minutes (shortened from 30 for the shared Spark), 4 workers, 133 requests of mixed kinds with 20 cancels and disconnects, 3 probes with constant hashes, 0 failed, free memory never below 18.05 GiB | this PR |
| S7 refusal/quality | stream L | refusal 100% to 0%, over-refusal 3.33% to 0%, MMLU 89.0% to 85.0% (paired p = 0.057), HumanEval 90.0% to 82.5% (p = 0.38): refusals removed, quality within noise at p < 0.05 with MMLU borderline (`eval/REPORT.md`) | this PR |

## 2. The RTX 6000 (approved: one Colab G4 runtime)

The maintainer approved a Google Colab G4 runtime: 1x RTX PRO 6000 Blackwell 96 GB, 176 GB RAM, Zen 5 EPYC,
236 GB of disk. Colab VMs are ephemeral and their driver and power cap can change between runtimes, so
`rental.sh check` records both; a cap below 600 W makes speed numbers incomparable with the old tp1 reference,
which ran at 600 W. One host, same-host ABBA only. Everything below runs unattended from a fresh host in one
command, with no compile: the engine comes prebuilt in a `recipe kit` bundle. Stop the runtime when the run
ends or would sit idle for more than about 30 minutes: it burns compute units.

### Host

| need | minimum | why |
| --- | --- | --- |
| GPU | 1x RTX PRO 6000 Blackwell, 96 GB (sm_120) | the recipe's hardware profile; the reference numbers are at 600 W |
| driver | 580 or later | the engine is built with nvcc 13.0.2 (CUDA 13.0) |
| CPU, OS | x86_64 with AVX2 (x86-64-v3), glibc 2.35 or later (Ubuntu 22.04+) | the bundle's binary is built for `-Dcpu=x86_64_v3` in `nvidia/cuda:13.0.2-devel-ubuntu22.04` |
| RAM | 128 GB (192 GB better) | the 47.7 GiB FP8 PLE table is page cache, never uploaded; a smaller host re-reads it from disk at each boot |
| disk | 140 GB free for weights | one pack at a time: the control pack (124 GiB) is removed before the shipped pack (126 GiB) comes |
| software | python 3.11+, git, curl, pip, network to Hugging Face | the kit, the weight fetch |

### The bundle

`qwen-flash-abliterated-rtx6000x1-876097bfb837` (engine tree `876097bf`, the pin plus `patches/loader`, `patches/tp1:0011` and
`patches/re-pin`, built for `GPU=sm_120 CPU=x86_64_v3`), attached to this repository's release `engine-876097bfb837-sm120-r2`:

- bundle `.tar.gz` sha256 `3b39392ab498e9fd12b43e26033de25d57d88e7ededd6ab9929d785ecc818139`
- `engine/zig-out/native/bin/tensorfold-native` sha256 `4ee7b682d62f1f9610249d5bdebbfde729b1e3c6f656c5a01b37efa61c490495`
- the bundle's kit is this repository at the recipe's `[kit] rev`, so its receipts are clean, not `dev:`

### Run it

```sh
W=$HOME/rental; mkdir -p "$W" && cd "$W"
gh release download engine-876097bfb837-sm120-r2 -R jayleaton/qwen-flash-abliterated-rtx6000x1-tensorfold
echo "3b39392ab498e9fd12b43e26033de25d57d88e7ededd6ab9929d785ecc818139  qwen-flash-abliterated-rtx6000x1-876097bfb837.tar.gz" | sha256sum -c && tar xzf qwen-flash-abliterated-rtx6000x1-876097bfb837.tar.gz
gh repo clone jayleaton/qwen-flash-abliterated-rtx6000x1-tensorfold repo
nohup bash repo/qwen-flash-abliterated-rtx6000x1/tools/rental.sh all > rental.log 2>&1 &
```

`tools/rental.sh all` runs the steps below in order, stops at the first one that fails and logs one line a step
to `$W/out/SUMMARY.txt`; each step also runs alone (`rental.sh hashes`). A pack already on disk (`MODEL=`,
`CONTROL=`) is checked and not downloaded again. Afterwards, commit the recorded `gates/hashes.json` from
`repo/` and the receipts' results into the table below.

| # | step | what | expected |
| --- | --- | --- | --- |
| R0 | `check` | the host against the table above, its power limit and CPU; `recipe kit --verify` on the bundle | every row passes |
| R0 | `fetch_control` | `hf download` of the nvidia pack at `fc694b54`, then every file checked: `sha256sum -c config/control.sha256` for the large ones, `git hash-object` against `config/control.blob` for the rest | 13 sha256 and 12 blobs match |
| R1, R3 | `control` | the nvidia pack's `default` hashes and unsalted cold 8K against the old recipe's one-GPU reference, then `recipe bench default optimised` on it (ABBA inside the kit's control arm, 2 rounds, 6 cells, bar 3%) | code `4e759dbd859c`, chat `764da77fbd3f`, cold 8K `fa345c10f631`; `default` code x1 about 291 tok/s, cold 8K about 8,818 tok/s at 600 W |
| | `drop_control` | removes the nvidia pack | |
| R0 | `fetch` | the shipped pack at `be794b99`, checked against `config/weights.sha256` and `config/weights.blob` | 208 sha256 and 214 blobs match |
| R2 | `fit` | boots `default` (16,384 x 16) and `agent` (139,264 x 3) on the shipped pack, logging the boot budget line and device memory | both boot; device memory matches the control pack's |
| R1 | `hashes` | the shipped pack's hashes and cold 8K row, written into `gates/hashes.json` (the 16,384-token profiles cannot serve the 32K prompt) | |
| R1 | `gate` | `recipe gate default`, then `agent`: conformance with its plants, hashes, nondeterminism, canary | all pass; resumed and evict need a second server and do not fit beside the first on one card |
| R3 | `bench` | `recipe bench default optimised` on the shipped pack | `optimised` decides by the bar |
| R4 | `soak` | `recipe soak default`, 30 minutes of mixed traffic with cancels, at the `rtx6000x1` floors | passes |

If the control's hashes differ from the reference, that is a finding to report, never a silent accept: isolate
a moved hash with `TF_FLASHNEXT_READOUT_FUSED=0` before calling it a regression.

**Fit, expected.** After load-time quantization the shipped pack's device bytes equal the nvidia pack's: the
two packs share every tensor except the MTP experts, which load into the same FP8 form. The nvidia pack on one
6000 uploaded 73.3 GiB and peaked at 81.2 GiB at `--parallel 4` on the old pin. On one GB10 the shipped pack
reported `model 76.93 GiB` and held 77.1 GiB at ready and 80.4 GiB after the gate, at 65,536 x 4. So it fits in
96 GB, with the ported [CAPACITY](qwen-flash-abliterated-rtx6000x1/docs/CAPACITY.md) rows' headroom.

**Time.** About 4.5 hours with `all`: check and bundle 10 min; two fetches 15-25 min each (124-126 GiB at
about 1 Gb/s or better, then the checks, about 2 min a pack); control hashes 10 min; two bench ABBAs about
1.5 h (12 boots of about 75 s plus their cells, a bench); fit 5 min; hashes 10 min; gate 20 min; soak 35 min.
Budget 5 hours, about 50 compute units at 10 an hour. The follow-ups below add about 2 hours.

**Afterwards, by hand** (not in `all`): attribute the `optimised` stack leave-one-out (each lever off in turn,
a dev `--set` run or a temporary edit); `recipe bench default drafts-greedy`; an ABBA of the old pin
(`924d50a9d`) against this one on the same host, since host drift is larger than most levers here; then
`recipe soak` of the winning arm. A follow-up, not wired: the FP8 recipe's 0084 (`TF_FLASHNEXT_PLE_EARLY`,
`TF_FLASHNEXT_ROUND_TURN`), which this series does not carry.

Results (filled by the stream that gets the approved window):

| step | run by | result | receipt / PR |
| --- | --- | --- | --- |
| R1 6000 hashes | stream L, stream V | the control pack at this pin reproduces the old recipe's one-GPU reference exactly, decode code `4e759dbd859c`, chat `764da77fbd3f` and cold 8K `fa345c10f631` (all 12 decode hashes); the shipped pack's `gates/hashes.json`: code `80a5e186a17c`, chat `9b016afeff58`, cold 8K `16307318fc93`, all 12 decode hashes equal to the GB10's `gates/spark.hashes.json`. Gate `default` and `agent`: 4 passed, 0 failed, 2 skipped (resumed and evict) | this PR |
| R2 fit | stream L, stream V | `default` boots on both packs with `model 76.91 GiB; 16 streams at once, 0.85 GiB each at a 16384-token window, of 14.0 GiB left`, 79,870 MiB on the card, ready in 92-104 s. `optimised` now serves `--parallel 12`: `model 79.55 GiB; 12 streams at once, 0.95 GiB each ... of 11.4 GiB left`, 82,564 MiB, no headroom. `agent` needed 13.41 GiB of 11.34 GiB with `TF_FLASHNEXT_CHUNK_SOLO=8192`; without it, `3 streams at once, 4.37 GiB each at a 139264-token window, of 14.0 GiB left`, 79,900 MiB (KNOWN-ISSUES) | this PR |
| R3 ABBA | stream V | shipped pack, 550 W cap (old tp1 reference: 600 W), 2 rounds, bar 3%, control drift up to 2.4%. `default`: code x1 266.6, chat x1 185.3, cold 8K 8,254, code x16 1,345.6, chat x16 1,199.5 tok/s (old tp1 on the control pack at 600 W: code x1 291, cold 8K 8,818). `optimised` (x12) against `default`: code x1 -0.9%, chat x1 +0.2%, cold 8K +4.6%, code x12 +8.0%, chat x12 +12.1%; the kit's verdict reads `worse` only from the x16 cells, which 12 streams serve queued (-9.6%, -20.8%). By the first cell `optimised` does not win at one stream | this PR |
| R4 soak | stream V | STABLE: `recipe soak default` on the shipped pack, 30 minutes, 4 workers, 2,325 requests with 262 cancels, 0 failed, 7 probes with constant hashes, free memory never below 14.78 GiB, peak 92 C at a 550 W cap. Two earlier 30-minute soaks read UNSTABLE only from their cold 32K probes, which a 16,384-token window refuses (KNOWN-ISSUES); each also had 0 failed requests | this PR |

## What lands afterwards

- The decided levers: `optimised` gains the winners as product rows with their receipts; the losers' rows and
  arms are deleted.
- The measured tables replace the *measured on the old pin* labels in the recipe's docs.
- The shipped pack's hashes and cold rows make `gates/hashes.json` a real gate, and the first soak receipt
  closes KNOWN-ISSUES' open row.
