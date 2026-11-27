# RUNBOOK: how the numbers were measured, and how to re-run them

Two machines, in this order: a DGX Spark (GB10) for every check that does not need a 6000's speed, then one RTX
PRO 6000 for the speed, fit, gate and soak runs. Any card of the same kind works for either; keep host names,
addresses and accounts out of the repository (put them in `.check-public.local`, which is git-ignored, so
`recipe publish-check` catches them).

Every step runs the kit from a checkout of this repository (`bin/recipe ... --kit-dev`, receipts stamped `dev:`),
or from a `recipe kit` bundle for clean receipts: the bundle's kit release carries the pinned revision's
`REVISION` file. `ENGINE` below is a prepared engine directory, `MODEL` a downloaded pack.

The engine is upstream TensorFold main plus six series, applied in order: the **loader series** (the shipped
pack's stated quantization and fused MTP experts do not load without it), the **one-GPU lever series**
(`patches/tp1`), a **re-pin patch** that reconciles the two series' re-pins of upstream's serving-source
hashes, the **disk permit** (`patches/disk`), which lets one state as long as the window reach the disk tier, the
**dynamic K/V series** (`patches/dynkv`), which puts the engine's own stream on upstream's page pool so that
`agent`'s streams share one pool, and the **precision series** (`patches/precision`: `--precision checkpoint` for
the opt-in `w4a4` profile, with its own re-pin). The **eval harness** in the recipe's `eval/` scores refusals and
quality.

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

### The shared K/V pool on the GB10

Tree `dcc6f37a` (the stack above without `patches/disk`, plus `patches/dynkv`) built for `-Dgpu=spark`, with dev-only GB10 profiles
beside the recipe's: fixed windows at 65,536 x 4, the same with `TF_FLASHNEXT_KV_PAGES=256` and
`TF_FLASHNEXT_KV_PAGES_ALL=1`, and the pages at 262,144 x 8. Without the levers and with them (`agent`'s set
and a host cache), each one's numbers are in [CAPACITY](qwen-flash-abliterated-rtx6000x1/docs/CAPACITY.md#one-shared-kv-pool-agent).

| step | result |
| --- | --- |
| D1 host tests | `zig build test-cuda-host`: 9,852 of 9,898 pass, the one failure (`start CLI: Ctrl-C during a deferred download`) the same on the base tree (9,851 of 9,897); `lean_check` 123 findings, as the base; `pr_check` 0 errors |
| D2 hashes | 12/12 equal to `gates/spark.hashes.json` for fixed windows, every stream on pages, 262,144 x 8, and 262,144 x 8 with `agent`'s levers |
| D3 ABBA | fixed against paged, 2 rounds, bar 3%: code x1 -0.1%, chat x1 -0.1%, code x4 +1.5%, chat x4 +0.3%, cold 8K +0.3%; no difference |
| D4 mixed | 262,144 x 8: short chats beside one session grown to 152,125 tokens, 0 of 123 failed; with a host cache each turn restored the last |
| D5 exhaustion | 12 requests reserving about 263K tokens each of a 787,200-token pool ran two at a time, the rest waiting; all 12 returned 200 |
| D6 soak | 10 minutes at 262,144 x 8: STABLE, 140 requests, 22 cancels, 0 failed, probe hashes constant |
| D7 with `patches/disk` | the recipe's tree `0272db7d`, `agent` at the GB10's sizes (16 GiB RAM cache, the disk tier on): hashes 12/12 equal; mixed traffic 0 of 26 failed, the session grown to 152,125 tokens restoring each turn, 8.2 GiB of states on disk; 10-minute soak STABLE, 136 requests, 0 failed, probe hashes constant |

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

The release bundle is tree `776a5050` (the pin plus all six series), built for `GPU=sm_120 CPU=x86_64_v3` in
`nvidia/cuda:13.0.2-devel-ubuntu22.04` and attached to this repository's release `engine-776a50508dca-sm120-r3`:

- `qwen-flash-abliterated-rtx6000x1-776a50508dca.tar.gz` sha256 `511339d41fcff534918682aba53589edc37b261a1231002fe3dfddf7f17bbe55`
- `engine/zig-out/native/bin/tensorfold-native` sha256 `2123ceca50b25a57c89dcc237372aa51e5964fef5544e954cdc51c665e779898`
- the bundle's kit is this repository at the recipe's `[kit] rev`, so its receipts are clean, not `dev:`

R1 to R4 below ran on the earlier bundle `engine-876097bfb837-sm120-r2` (tree `876097bf`, before `patches/disk`,
`patches/dynkv` and `patches/precision`; tar.gz sha256 `3b39392ab498e9fd12b43e26033de25d57d88e7ededd6ab9929d785ecc818139`,
binary `4ee7b682d62f1f9610249d5bdebbfde729b1e3c6f656c5a01b37efa61c490495`); its `default` and `optimised` give the same
tokens as r3's, so it stays published for those cells' comparability.

### Run it

```sh
W=$HOME/rental; mkdir -p "$W" && cd "$W"
gh release download engine-776a50508dca-sm120-r3 -R jayleaton/qwen-flash-abliterated-rtx6000x1-tensorfold
echo "511339d41fcff534918682aba53589edc37b261a1231002fe3dfddf7f17bbe55  qwen-flash-abliterated-rtx6000x1-776a50508dca.tar.gz" | sha256sum -c && tar xzf qwen-flash-abliterated-rtx6000x1-776a50508dca.tar.gz
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
| R2 | `fit` | boots `default` (16,384 x 16) and `agent` (262,144 x 8 on one page pool) on the shipped pack, logging the boot budget line and device memory | both boot; device memory matches the control pack's |
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
| R2 fit | `default` boots on both packs with `model 76.91 GiB; 16 streams at once, 0.85 GiB each at a 16384-token window, of 14.0 GiB left`, 79,870 MiB on the card, ready in 92-104 s. `optimised` serves `--parallel 12`: `model 79.55 GiB; 12 streams at once, 0.95 GiB each ... of 11.4 GiB left`, 82,564 MiB, no headroom. `agent`, at its earlier 139,264 x 3, needed 13.41 GiB of 11.34 GiB with `TF_FLASHNEXT_CHUNK_SOLO=8192`; without it, `3 streams at once, 4.37 GiB each at a 139264-token window, of 14.0 GiB left`, 79,900 MiB (KNOWN-ISSUES) |
| R3 ABBA | shipped pack, 550 W cap (earlier reference: 600 W), 2 rounds, bar 3%, control drift up to 2.4%. `default`: code x1 266.6, chat x1 185.3, cold 8K 8,254, code x16 1,345.6, chat x16 1,199.5 tok/s (the earlier reference on NVIDIA's pack at 600 W: code x1 291, cold 8K 8,818). `optimised` (x12) against `default`: code x1 -0.9%, chat x1 +0.2%, cold 8K +4.6%, code x12 +8.0%, chat x12 +12.1%; the kit's verdict reads `worse` only from the x16 cells, which 12 streams serve queued (-9.6%, -20.8%). By the first cell `optimised` does not win at one stream |
| R4 soak | STABLE: `recipe soak default` on the shipped pack, 30 minutes, 4 workers, 2,325 requests with 262 cancels, 0 failed, 7 probes with constant hashes, free memory never below 14.78 GiB, peak 92 C at a 550 W cap. Two earlier 30-minute soaks read UNSTABLE only from their cold 32K probes, which a 16,384-token window refuses (KNOWN-ISSUES); each also had 0 failed requests |

### A second host at 600 W

One RTX PRO 6000 Blackwell **Server** Edition on another cloud host: driver 580.82.07, power limit 600 W (its default
and maximum), PCIe gen5 x16, 48 x86 cores, 176 GiB RAM, NVMe scratch. Under one decode stream the SM held 2,265-2,280
MHz at up to 300 W with no throttle reason. The `default` cells ran on bundle r2; `agent` on tree `0272db7d`, which
differs from `776a5050` only by `patches/precision` (opt-in, `w4a4` only); `w4a4` on bundle r3. Its numbers compare
with each other, not with the first host's (another card and cap).

| step | result |
| --- | --- |
| G1 ABBA at 600 W | r2, `default` against `optimised` (x12), 2 rounds, bar 3%, control drift up to 1.0%: `default` code x1 242.6, chat x1 167.4, cold 8K 7,558, code x16 1,189.5, chat x16 1,044.2 tok/s; `optimised` code x1 -0.2%, chat x1 +1.1%, cold 8K **+15.8%** (8,753), code x16 -6.4%, chat x16 -16.9% (12 streams serve x16 queued, so the kit's `worse` reads those cells). An earlier pass was void: its first cell's cold 8K ran at 1,384 tok/s (KNOWN-ISSUES) |
| G2 `agent` fit and gate | `model 76.97 GiB; 8 streams at once ... 1900 pages of 256 rows (13.95 GiB) as they need them (enough for 1 at a 262144-token window) ... of 14.0 GiB left`; `/metrics` `kv_capacity_mode` paged, pool 486,400 tokens; 83,711 MiB on the card. `recipe gate agent`: 4 passed, 0 failed, 2 skipped (resumed, evict) |
| G3 `agent` ABBA | against `default`, x1 cells: code -0.2%, chat +0.7%; no difference |
| G4 fixed against paged | dev profiles at 36,864 x 8 (fixed, and the same with `TF_FLASHNEXT_KV_PAGES=256`, `_ALL=1`), same binary: code x1 +0.7%, chat x1 +0.3%, code x8 +0.6%, chat x8 +0.4%, cold 8K +0.9%, cold 32K 8,030 against 8,010 (fixed's first 32K cell, 1,393, left out); same tokens. The kit reads it `unmeasured`: its `default` control cannot serve the 32K cell |
| G5 mixed traffic | `agent`, NVMe tier on: short chats beside one session grown to 247,320 tokens; 55 of 56 requests 200, the 56th that session's next turn (267,476 tokens), refused `context_length_exceeded`. Pool never below 144,128 tokens free of 486,400; up to 7 running and 6 waiting; 14 GB on disk |
| G6 turn-2 TTFT | S8's method on `agent`: cold / RAM / NVMe after eviction / NVMe after restart: 32K 6.18 / 1.23 / 4.37 / 1.61 s, 64K 10.14 / 1.40 / 2.59 / 2.11 s, 128K 17.17 / 1.89 / 5.63 / 3.09 s; tokens equal in all four at each size and equal to the GB10's (`7989cc9ec180`, `9ccea81c1c81`, `0b75826f55a0`); 0 `StagingFull` |
| G7 one long prompt | S9's method: 64K cold 10.27 / RAM 1.40 / NVMe after restart 2.08 s, 128K 18.87 / 1.82 / 3.03 s; tokens `d2a261efbf82`, `4169cbb640d8` (the GB10's); 0 `StagingFull` |
| G8 restore rate | the server's disk reads: 2.0 GiB in 0.66 s and 3.9 GiB in 1.25-1.55 s (2.5-3.1 GiB/s); the whole restore after a restart (disk read and the copy over PCIe, the request's `wait_seconds`) 1.04 GiB in 0.56 s, 1.96 GiB in 1.06 s, 3.82 GiB in 2.06 s: **1.85 GiB/s** at every size |
| G9 `agent` soak | STABLE: 30 minutes, 4 workers, 1,298 requests with 169 cancels, 0 failed, 7 probes with constant hashes, free memory never below 11.04 GiB, peak 67 C |
| W1 `w4a4` gate | r3: 4 passed, 0 failed, 2 skipped; hashes 12/12 equal to the GB10-recorded `w4a4` rows, so they stand unchanged; cold 8K `f95c702eb707` and 32K `72b1fa31238e`, the GB10's |
| W2 `w4a4` ABBA | against `optimised` (control `default`), 2 rounds: cold 8K **-12.7%** (7,553 against 8,647), code x1 +5.6%, chat x1 +9.4%, code x12 -4.9%, chat x12 -3.4%. The lever loses on this card's prompts |
| W3 small passes | the flag on `default`'s 3,072-row passes, against `default`: cold 8K **-38.4%** (4,492 against 7,290), code x16 +0.9% |
| W4 nsys | one cold prompt under nsys at `--context 65536 --parallel 4`, window ms 8K / 32K: `default` 1,007 / 4,067, `optimised` 834 / 3,421, `w4a4` 964 / 3,957. Routed experts at 8K: 237, 195 and 326 ms; the FP4 path is the slower one here. `default`'s largest family is the DeltaNet chain (25.8%; 8.2% on the GB10), `optimised`'s split takes it from 253 to 154 ms. GPU busy 95-97% |
| W5 kernel contract | `tf-cuda-test kernel-contract "ck swiglu" "ck experts"` built for sm_120 from tree `776a5050`: 46 PASS, 0 FAIL, 0 N/A |
| W6 `w4a4` eval | the eval's flags: refusal 0/50, over-refusal 0/30, MMLU 83.0%, HumanEval 82.5%, the GB10's row |
| W7 `w4a4` fit | `model 79.55 GiB; 12 streams at once, 0.95 GiB each at a 16384-token window, of 11.4 GiB left`, 82,563 MiB: as `optimised` |

## Still open

- `optimised` has not run a soak, and its levers have not been attributed one by one.
- `w4a4` loses on a 6000's prompts (W2-W4): its FP4 expert path is slower than the BF16-row one there. It stays
  opt-in and unshipped; a soak of it is not owed until a lever makes it win.
- The `drafts-greedy` arm has not been benched.
- A follow-up, not wired: the FP8 recipe's 0084 (`TF_FLASHNEXT_PLE_EARLY`, `TF_FLASHNEXT_ROUND_TURN`), which this
  series does not carry.
