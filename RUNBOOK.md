# RUNBOOK: how the numbers were measured, and how to re-run them

Two machines, in this order: a DGX Spark (GB10) for every check that does not need a 6000's speed, then one RTX
PRO 6000 for the speed, fit, gate and soak runs. Any card of the same kind works for either; keep host names,
addresses and accounts out of the repository (put them in `.check-public.local`, which is git-ignored, so
`recipe publish-check` catches them).

Every step runs the kit from a checkout of this repository (`bin/recipe ... --kit-dev`, receipts stamped `dev:`),
or from a `recipe kit` bundle for clean receipts: the bundle's kit release carries the pinned revision's
`REVISION` file. `ENGINE` below is a prepared engine directory, `MODEL` a downloaded pack.

The engine is upstream TensorFold main plus four series, applied in order: the **loader series** (the shipped
pack's stated quantization and fused MTP experts do not load without it), the **one-GPU lever series**
(`patches/tp1`), a **re-pin patch** that reconciles the two series' re-pins of upstream's serving-source
hashes, and the **disk permit** (`patches/disk`), which lets one state as long as the window reach the disk tier. The **eval harness** in the recipe's `eval/` scores refusals and quality.

## 1. A GB10 (DGX Spark)

sm_121, 48 SMs, unified memory. Good for every step below except speed: tok/s on a GB10 predicts nothing on a 6000,
and levers are decided only by a 6000's ABBA.

| # | step | what |
| --- | --- | --- |
| S1 | verify | the pin's own checks on the prepared tree: `zig fmt --check`, `tools/pr_check.py`, `zig build test test-kernel-contract test-golden` (the `qmmf-prompt-mx-contract` rc 139 on sm_120 is open; see the recipe's KNOWN-ISSUES) |
| S2 | prepare for sm_121 | `GPU=spark recipe prepare`, xgrammar addon built |
| S3 | fit | the shipped pack loads through `patches/loader`, and the boot-time CUDA budget admits the profile |
| S4 | hashes | one-GPU token hashes of the shipped pack, kept in `gates/spark.hashes.json` as a GB10 reference |
| S5 | gate | `recipe gate default`: conformance with every plant, hashes, nondeterminism, canary |
| S6 | soak | `recipe soak default`, mixed traffic with cancels |
| S8 | prompt states on NVMe | `agent`'s flags on the GB10's NVMe: turn-2 TTFT of sessions grown in 16K-token turns to 32K, 64K and 128K, cold (a fresh `cache_salt`), from RAM, from disk after RAM evicted the state and from disk after a restart, each reply's token hash; then a 10-minute `recipe soak agent` with the store's size sampled every 30 s |
| S9 | one long first prompt | with `patches/disk`: a 64K and a 128K session sent as one first-turn prompt; its state on disk, cold, from RAM and from disk after a restart, each reply's token hash, and the server log's `StagingFull` count |
| S7 | refusal/quality | the recipe's `eval/` harness against both packs served with identical flags: AdvBench harmful refusal (50 prompts), XSTest benign over-refusal (30), MMLU (200), HumanEval (40) |

The GB10 runs served `--context 65536 --parallel 4` with every lever off; the `rtx6000x1` profiles' windows and
widths are the 6000's to measure.

| step | result |
| --- | --- |
| S1 verify | partial: `zig fmt --check` and `git diff --check` clean; `tools/zig/lean_check.py` 123 findings, the same as the base; `zig build test-gradient-serving` passes on the stacked tree. `pr_check`, `test`, `test-kernel-contract` and `test-golden` not run |
| S2 prepare | tree `876097bf` built for `-Dgpu=spark` with xgrammar v0.2.8; `tensorfold-native` 1.3.6, sha256 `5212113159f1d41ba673ddd6fa5231c381a249e5e4bc7a61d37c3f0b60eb1dd6` |
| S3 fit | the shipped pack loads; boot budget `model 76.93 GiB; 4 streams at once, 2.26 GiB each at a 65536-token window, of 25.7 GiB left after a 12.2 GiB reserve`; ready in 74.7 s from a dropped page cache; 77.1 GiB at ready, 80.4 GiB after the gate. Draft head 4.27 tokens a round on greedy code (NVIDIA's pack: 4.41 on a 6000) |
| S4 hashes | `gates/spark.hashes.json`: greedy code `80a5e186a17c`, chat `9b016afeff58`, five seeds each |
| S5 gate | conformance passes (drafted, repeat, concurrent, refusals with `n: 17`, cancel; 5/5 plants red); hashes 12/12, red with every seed moved; canary 3/3; nondeterminism passes: 16 copies, up to 16 open at once, one output a mode. resumed and evict need a second, cache-off server |
| S6 soak | STABLE: 10 minutes, 4 workers, 133 requests of mixed kinds with 20 cancels and disconnects, 3 probes with constant hashes, 0 failed, free memory never below 18.05 GiB |
| S7 refusal/quality | refusal 100% to 0%, over-refusal 3.33% to 0%, MMLU 89.0% to 85.0% (paired p = 0.057), HumanEval 90.0% to 82.5% (p = 0.38) ([`eval/REPORT.md`](qwen-flash-abliterated-rtx6000x1/eval/REPORT.md)) |
| S8 prompt states | `agent`'s args at `--context 139264 --parallel 3` (the GB10 sizes the RAM cache to 15.9 GiB). Turn-2 TTFT cold / RAM / NVMe after eviction / NVMe after restart: 32K 15.9 / 0.66 / 1.92 / 1.62 s, 64K 32.1 / 1.38 / 3.52 / 2.70 s, 128K 157.1 / 0.46 / 5.76 / 5.08 s; reply tokens equal in all four at each size (`7989cc9ec180`, `9ccea81c1c81`, `0b75826f55a0`). States on disk: 1.04, 1.96 and 3.82 GiB. Before `patches/disk`, a 32K prompt sent in one piece was written too but a 64K one was skipped (`StagingFull`). Soak STABLE: 10 minutes, 126 requests, 3 probes with constant hashes, 0 failed; the store grew from 0 to 4.5 GiB (the kit refused only to write the receipt, whose args held the run's absolute folder) |
| S9 one long prompt | tree `88d5f918` (`patches/disk`), RAM cache 18.6 GiB. 64K: cold 32.6 s, RAM 1.07 s, NVMe after restart 2.73 s, tokens `d2a261efbf82` in all three, 2.00 GiB on disk. 128K: cold 67.2 s, RAM 0.58 s, NVMe after restart 5.07 s, tokens `4169cbb640d8`, 3.89 GiB. 0 `StagingFull` lines. S8's grown 128K session cold-prefilled in 157.1 s against this one prompt's 67.2 s; S8 ran with a 15.9 GiB RAM cache and many short messages, and the difference is not isolated |

## 2. One RTX PRO 6000

The 6000 runs below used a rented cloud host. Cloud hosts' drivers and power caps change between machines, so
`rental.sh check` records both; a cap below 600 W makes speed numbers incomparable with the earlier one-GPU reference,
which ran at 600 W. One host, same-host ABBA only. Everything runs unattended from a fresh host in one command, with no
compile: the engine comes prebuilt in a `recipe kit` bundle. On a rented host, stop it when the run ends.

### Host

| need | minimum | why |
| --- | --- | --- |
| GPU | 1x RTX PRO 6000 Blackwell, 96 GB (sm_120) | the recipe's hardware profile |
| driver | 580 or later | the engine is built with nvcc 13.0.2 (CUDA 13.0) |
| CPU, OS | x86_64 with AVX2 (x86-64-v3), glibc 2.35 or later (Ubuntu 22.04+) | the bundle's binary is built for `-Dcpu=x86_64_v3` in `nvidia/cuda:13.0.2-devel-ubuntu22.04` |
| RAM | 128 GB (192 GB better) | the 47.7 GiB FP8 PLE table is page cache, never uploaded; a smaller host re-reads it from disk at each boot |
| disk | 140 GB free for weights | one pack at a time: the control pack (124 GiB) is removed before the shipped pack (126 GiB) comes |
| software | python 3.11+, git, curl, pip, `gh`, network to Hugging Face and GitHub | the kit, the weight fetch, the bundle |

### The bundle

`qwen-flash-abliterated-rtx6000x1-876097bfb837` (engine tree `876097bf`, the pin plus `patches/loader`,
`patches/tp1:0011` and `patches/re-pin`, built for `GPU=sm_120 CPU=x86_64_v3`), attached to this repository's
release `engine-876097bfb837-sm120-r2`:

- bundle `.tar.gz` sha256 `3b39392ab498e9fd12b43e26033de25d57d88e7ededd6ab9929d785ecc818139`
- `engine/zig-out/native/bin/tensorfold-native` sha256 `4ee7b682d62f1f9610249d5bdebbfde729b1e3c6f656c5a01b37efa61c490495`
- the bundle's kit is this repository at the recipe's `[kit] rev`, so its receipts are clean, not `dev:`

### Run it

```sh
W=$HOME/rental; mkdir -p "$W" && cd "$W"
gh release download engine-876097bfb837-sm120-r2 -R jayleaton/qwen-flash-abliterated-rtx6000x1-tensorfold
echo "3b39392ab498e9fd12b43e26033de25d57d88e7ededd6ab9929d785ecc818139  qwen-flash-abliterated-rtx6000x1-876097bfb837.tar.gz" | sha256sum -c && tar xzf qwen-flash-abliterated-rtx6000x1-876097bfb837.tar.gz
git clone https://github.com/jayleaton/qwen-flash-abliterated-rtx6000x1-tensorfold repo
nohup bash repo/qwen-flash-abliterated-rtx6000x1/tools/rental.sh all > rental.log 2>&1 &
```

`tools/rental.sh all` runs the steps below in order, stops at the first one that fails and logs one line a step
to `$W/out/SUMMARY.txt`; each step also runs alone (`rental.sh hashes`). A pack already on disk (`MODEL=`,
`CONTROL=`) is checked and not downloaded again. Afterwards, `repo/` holds the recorded `gates/hashes.json`.

| # | step | what | expected |
| --- | --- | --- | --- |
| R0 | `check` | the host against the table above, its power limit and CPU; `recipe kit --verify` on the bundle | every row passes |
| R0 | `fetch_control` | `hf download` of NVIDIA's pack at `fc694b54`, then every file checked: `sha256sum -c config/control.sha256` for the large ones, `git hash-object` against `config/control.blob` for the rest | 13 sha256 and 12 blobs match |
| R1, R3 | `control` | NVIDIA's pack's `default` hashes and unsalted cold 8K against the earlier one-GPU reference, then `recipe bench default optimised` on it (ABBA inside the kit's control arm, 2 rounds, 6 cells, bar 3%) | code `4e759dbd859c`, chat `764da77fbd3f`, cold 8K `fa345c10f631`; `default` code x1 about 291 tok/s, cold 8K about 8,818 tok/s at 600 W |
| | `drop_control` | removes NVIDIA's pack | |
| R0 | `fetch` | the shipped pack at `be794b99`, checked against `config/weights.sha256` and `config/weights.blob` | 208 sha256 and 214 blobs match |
| R2 | `fit` | boots `default` (16,384 x 16) and `agent` (139,264 x 3) on the shipped pack, logging the boot budget line and device memory | both boot; device memory matches the control pack's |
| R1 | `hashes` | the shipped pack's hashes and cold 8K row, written into `gates/hashes.json` (the 16,384-token profiles cannot serve the 32K prompt) | |
| R1 | `gate` | `recipe gate default`, then `agent`: conformance with its plants, hashes, nondeterminism, canary | all pass; resumed and evict need a second server and do not fit beside the first on one card |
| R3 | `bench` | `recipe bench default optimised` on the shipped pack | `optimised` decides by the bar |
| R4 | `soak` | `recipe soak default`, 30 minutes of mixed traffic with cancels, at the `rtx6000x1` floors | passes |

If the control's hashes differ from the reference, that is a finding, never a silent accept: isolate a moved hash
with `TF_FLASHNEXT_READOUT_FUSED=0` before calling it a regression.

**Time.** About 4.5 hours with `all`: check and bundle 10 min; two fetches 15-25 min each (124-126 GiB at about
1 Gb/s or better, then the checks, about 2 min a pack); control hashes 10 min; two bench ABBAs about 1.5 h (12 boots
of about 75 s plus their cells, a bench); fit 5 min; hashes 10 min; gate 20 min; soak 35 min. Budget 5 hours.

**Afterwards, by hand** (not in `all`): attribute the `optimised` stack leave-one-out (each lever off in turn, a dev
`--set` run or a temporary edit); `recipe bench default drafts-greedy`; then `recipe soak` of the winning arm.

### Results

| step | result |
| --- | --- |
| R1 hashes | NVIDIA's pack at this pin reproduces the earlier one-GPU reference exactly, decode code `4e759dbd859c`, chat `764da77fbd3f` and cold 8K `fa345c10f631` (all 12 decode hashes); the shipped pack's `gates/hashes.json`: code `80a5e186a17c`, chat `9b016afeff58`, cold 8K `16307318fc93`, all 12 decode hashes equal to the GB10's `gates/spark.hashes.json`. Gate `default` and `agent`: 4 passed, 0 failed, 2 skipped (resumed and evict) |
| R2 fit | `default` boots on both packs with `model 76.91 GiB; 16 streams at once, 0.85 GiB each at a 16384-token window, of 14.0 GiB left`, 79,870 MiB on the card, ready in 92-104 s. `optimised` serves `--parallel 12`: `model 79.55 GiB; 12 streams at once, 0.95 GiB each ... of 11.4 GiB left`, 82,564 MiB, no headroom. `agent` needed 13.41 GiB of 11.34 GiB with `TF_FLASHNEXT_CHUNK_SOLO=8192`; without it, `3 streams at once, 4.37 GiB each at a 139264-token window, of 14.0 GiB left`, 79,900 MiB (KNOWN-ISSUES) |
| R3 ABBA | shipped pack, 550 W cap (earlier reference: 600 W), 2 rounds, bar 3%, control drift up to 2.4%. `default`: code x1 266.6, chat x1 185.3, cold 8K 8,254, code x16 1,345.6, chat x16 1,199.5 tok/s (the earlier reference on NVIDIA's pack at 600 W: code x1 291, cold 8K 8,818). `optimised` (x12) against `default`: code x1 -0.9%, chat x1 +0.2%, cold 8K +4.6%, code x12 +8.0%, chat x12 +12.1%; the kit's verdict reads `worse` only from the x16 cells, which 12 streams serve queued (-9.6%, -20.8%). By the first cell `optimised` does not win at one stream |
| R4 soak | STABLE: `recipe soak default` on the shipped pack, 30 minutes, 4 workers, 2,325 requests with 262 cancels, 0 failed, 7 probes with constant hashes, free memory never below 14.78 GiB, peak 92 C at a 550 W cap. Two earlier 30-minute soaks read UNSTABLE only from their cold 32K probes, which a 16,384-token window refuses (KNOWN-ISSUES); each also had 0 failed requests |

## Still open

- `optimised` and `agent` have not run a soak, and the `optimised` levers have not been attributed one by one.
- The `drafts-greedy` arm has not been benched.
- A follow-up, not wired: the FP8 recipe's 0084 (`TF_FLASHNEXT_PLE_EARLY`, `TF_FLASHNEXT_ROUND_TURN`), which this
  series does not carry.
