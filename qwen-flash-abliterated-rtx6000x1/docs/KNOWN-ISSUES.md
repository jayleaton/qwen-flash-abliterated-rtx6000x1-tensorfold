# Known issues

Rows marked *old pin* are ported from an earlier one-GPU recipe and were measured on its pin; the rest are this
recipe's own.

## The shipped weights load through the `patches/loader` series

The abliterated pack keeps the control pack's quantization values (ModelOpt NVFP4) but states them as
`quant_algo: NVFP4` with an ignore list, and stores its MTP experts fused in BF16; upstream main refuses both.
The two loader patches teach the config check the stated layout and host-quantize the fused experts into the
FP8 form the head already serves (nvidia's own layout, at FP8 rounding noise). Served tokens do not depend on
the draft head, so the rounding can change acceptance, never tokens. On one GB10 the pack loads, the
conformance gate's `drafted` check (drafted equals plain) passes, and the head drafts 4.27 tokens a round on
greedy code (nvidia's pack: 4.41 on a 6000), so the layout is right.

## Soak-tested on `default` and `agent`, not `optimised`

On one RTX PRO 6000 a 30-minute `recipe soak default` of the shipped weights is STABLE (2,325 requests, 262
cancels, 0 failed, probe hashes constant, free memory at least 14.78 GiB); the card peaked at 92 C at a 550 W cap.
On a second 6000 a 30-minute `recipe soak agent` (262,144 x 8, the NVMe tier on) is STABLE: 1,298 requests, 169
cancels, 0 failed, 7 probes constant, free memory at least 11.04 GiB, 67 C at 600 W (RUNBOOK G9). `optimised` has not
run a soak.

## `optimised` serves 12 streams, `agent` drops `TF_FLASHNEXT_CHUNK_SOLO`

On one RTX PRO 6000 `optimised` refused `--parallel 16`: `needs 15.15 GiB for its streams at a 16384-token window,
and the CUDA memory budget leaves 11.36 GiB`. `TF_FLASHNEXT_CHUNK_SOLO=8192`'s prompt passes of 8,192 rows take
about 2.6 GiB (`model 79.55 GiB` against `default`'s 76.91), so `optimised` now serves `--parallel 12`, with every
lever kept and no headroom: 12 streams of 0.95 GiB in the 11.4 GiB left. `agent` inherits the levers but at
139,264 x 3 needed 13.41 GiB of 11.34 GiB, so it drops `CHUNK_SOLO` and boots with 14.0 GiB left. `optimised`'s
x16 bench cells are queued behind its 12 streams and do not compare with `default`'s.

## The NVMe tier on a 6000 restores at about 1.85 GiB/s

On a 6000 a state read from NVMe lands in pinned host memory (`KEPT=host`) and crosses PCIe at admission. Measured on
one 6000 (RUNBOOK G6-G8): the disk reads run at 2.5-3.1 GiB/s, the whole restore (read and copy) at 1.85 GiB/s, so a
128K session (3.82 GiB) is back in 2.06 s and its turn 2 starts in 3.1 s against 17.2 s cold. The first read of a
state the page cache has never held was slower once (1.04 GiB in 3.0 s). Another host's disk moves these numbers.

## `agent`'s shared pool reserves each request's reply limit

`agent` puts every stream on one K/V page pool (`TF_FLASHNEXT_KV_PAGES=256`, `_ALL=1`, `patches/dynkv`). On one 6000
it boots with 1,900 pages of 256 rows (486,400 tokens, enough for one stream at the whole window), gives the same tokens
as fixed windows within 1% (RUNBOOK G2-G5), and queues requests the pool cannot hold yet
([CAPACITY](CAPACITY.md#one-shared-kv-pool-agent)). A request reserves its reply limit up front, so a client that sends
a large `max_tokens` holds that many tokens of pages until it ends; `--max-tokens` (4,096) is the reservation when a
client sends none.

## Cold 32K needs a window over 32K

The 16,384-token `default` and `optimised` profiles refuse the 32,768-token cold prompt, so `gates/hashes.json`
carries the cold 8K row only (bench and soak probe every cold row); `prompts/ids-32768.json` stays for a profile
with a wider window to record its row.

## The first cold prompt of a size on a fresh host can run 5x slow

On the second 6000 host the first cold 8K and the first cold 32K prompt the host ever served ran at 1,384 and 1,393
tok/s against about 7,500-8,000 in every later cell, fresh server or not; the kit's control arm then voided that bench.
Its page cache could not be dropped (read-only), so this looks like a first read of something from the host's disk.
Run one throwaway cold prompt of each size before a bench on a new host.

## `w4a4` is slower on a 6000's prompts

On one 6000 (RUNBOOK W1-W7) `w4a4` gives the GB10's tokens (12/12 decode hashes, cold 8K and 32K) and its eval, and
its `ck` kernels pass the kernel contract on sm_120; but cold 8K is 12.7% slower than `optimised` (38.4% on
`default`'s small passes), and nsys puts the routed experts at 326 ms against 195. It wins one-stream decode (+5.6%
code, +9.4% chat) and loses at 12 streams. Opt-in and unshipped as before.

## Speed needs an honest power reading

The first 6000 host's GPU read 790-940 W at 32-34 C against its 550 W limit, so the software power cap held the
SM clock at about 600 MHz of 3,090 and every speed number was void (code x1 39.6 tok/s against the reference's
291). `tools/rental.sh check` logs the power limit; before a bench, `nvidia-smi --query-gpu=clocks.sm,power.draw
--format=csv` under load should show the clock near its maximum and a draw at or under the limit.

## The codex32 cliff (old pin)

At 32 streams decode falls off the drafting rounds' 64-row cap shared across streams (each stream drafted
129-145 tokens a request against 205-215 at 16). One card does not fit 32 streams at the served windows
([CAPACITY](CAPACITY.md)); keep `--parallel` at 16 or below.

## `qmmf-prompt-mx-contract` segfaults (old pin)

On sm_120 the `qmmf-prompt-mx-contract` kernel contract exits rc 139; the other contracts pass. It belongs to
a lane no profile turns on. Not investigated; the RUNBOOK's verify step re-runs the contracts on the new pin.

## Measured on two hosts

The first speed numbers come from one rented RTX PRO 6000 Workstation Edition at a 550 W cap; RUNBOOK G1-W7 from a
Server Edition at 600 W on another host (decode x1 about 242 tok/s there against 267 on the first). Same-host ABBA
only: a number from one host does not predict another's. The old pin's numbers stay labeled as such in
[PERFORMANCE](PERFORMANCE.md).
