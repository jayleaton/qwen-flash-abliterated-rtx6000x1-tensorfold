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

## Soak-tested on `default` only

On one RTX PRO 6000 a 30-minute `recipe soak default` of the shipped weights is STABLE (2,325 requests, 262
cancels, 0 failed, probe hashes constant, free memory at least 14.78 GiB); the card peaked at 92 C at a 550 W cap.
`optimised` has not run a soak; `agent` has soaked on a GB10 only (RUNBOOK S8 and D6).

## `optimised` serves 12 streams, `agent` drops `TF_FLASHNEXT_CHUNK_SOLO`

On one RTX PRO 6000 `optimised` refused `--parallel 16`: `needs 15.15 GiB for its streams at a 16384-token window,
and the CUDA memory budget leaves 11.36 GiB`. `TF_FLASHNEXT_CHUNK_SOLO=8192`'s prompt passes of 8,192 rows take
about 2.6 GiB (`model 79.55 GiB` against `default`'s 76.91), so `optimised` now serves `--parallel 12`, with every
lever kept and no headroom: 12 streams of 0.95 GiB in the 11.4 GiB left. `agent` inherits the levers but at
139,264 x 3 needed 13.41 GiB of 11.34 GiB, so it drops `CHUNK_SOLO` and boots with 14.0 GiB left. `optimised`'s
x16 bench cells are queued behind its 12 streams and do not compare with `default`'s.

## The NVMe tier's restore rate on a 6000 is not measured

S8 and S9 ran on a GB10, where a state read from NVMe lands in the same memory the GPU reads. On a 6000 it lands in pinned
host memory (`KEPT=host`) and crosses PCIe at admission; an earlier recipe saw 0.7-1 GiB/s restores under WSL2. To
confirm on one 6000, with a bundle rebuilt for `patches/disk` (the published one predates it): S8's and S9's turn-2 TTFT (cold, RAM, NVMe after eviction and after restart), the restore
GiB/s, and a 30-minute `recipe soak agent` with the tier on.

## `agent`'s shared pool is measured on a GB10, not yet on a 6000

`agent` puts every stream on one K/V page pool (`TF_FLASHNEXT_KV_PAGES=256`, `_ALL=1`, `patches/dynkv`). On one GB10
it gives the same tokens as fixed windows, an ABBA within 0.3%, and it queues requests the pool cannot hold yet
([CAPACITY](CAPACITY.md#one-shared-kv-pool-agent)). The 6000's page count (about 1,900, estimated from the GB10's
page size), its own ABBA, and a soak need a 6000 run with a bundle of tree `0272db7d`. A request reserves its
reply limit up front, so a client that sends a large `max_tokens` holds that many tokens of pages until it ends;
`--max-tokens` (4,096) is the reservation when a client sends none.

## Cold 32K needs a window over 32K

The 16,384-token `default` and `optimised` profiles refuse the 32,768-token cold prompt, so `gates/hashes.json`
carries the cold 8K row only (bench and soak probe every cold row); `prompts/ids-32768.json` stays for a profile
with a wider window to record its row.

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

## Measured on one host

Every speed number here comes from one rented RTX PRO 6000 at a 550 W cap; the earlier one-GPU reference ran at
600 W on another host. Same-host ABBA only: a number from one host does not predict another's. The old pin's
numbers stay labeled as such in [PERFORMANCE](PERFORMANCE.md).
