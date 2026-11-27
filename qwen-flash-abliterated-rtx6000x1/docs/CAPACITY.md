# Capacity (ported, measured on the old pin)

One RTX PRO 6000 Blackwell (96 GB) serving the control pack at `fc694b54`, as measured in an earlier,
unpublished one-GPU recipe's session. On this pin the shipped abliterated pack takes the same card memory:
`default` reports `model 76.91 GiB` on both packs and holds 79,870 MiB at ready (the [RUNBOOK](../../RUNBOOK.md)'s
R2 row).

| window | streams that fit | why |
| --- | --- | --- |
| 16,384 | 16 (`--parallel 16`, the `default` profile) | about 0.65 GiB a stream after the weights |
| 24,576 | 12 | 17.34 GiB of streams against 13.99 GiB free after the NVFP4 weights |
| 32 streams | does not fit | the boot-time CUDA budget check refuses the launch |

The old recipe's `codex32` rows show what pushing past the budget costs elsewhere: the two-GPU profiles
admitted 32 slots but drafting fell off a 64-row cap shared across the streams. Keep `--parallel` at 16 or
below for decode throughput on one card; the `agent` profile serves the model's 262,144-token window to 8 streams
from one shared page pool instead (below).

## One shared K/V pool (`agent`)

`agent` serves `--context 262144 --parallel 8` with `TF_FLASHNEXT_KV_PAGES=256` and `TF_FLASHNEXT_KV_PAGES_ALL=1`
(upstream's Flash Next page pool plus `patches/dynkv`). Every stream, the engine's own too, takes 256-token pages
of one pool the boot budget sizes. The plan sets one full window of pages aside, so one stream can always reach
262,144 tokens, but the pages stay shared. A request reserves its prompt, its reply limit (`max_tokens`, else
`--max-tokens`) and one drafted window at admission. A request the pool cannot hold yet waits in the queue; a
running stream never asks for memory, so nothing runs out mid-reply.

Measured on one GB10 (sm_121), tree `dcc6f37a`. The 6000 has not booted it yet:

| profile on the GB10 | boot line | tokens |
| --- | --- | --- |
| 65,536 x 4, fixed windows | `4 streams at once, 2.26 GiB each at a 65536-token window` | 12/12 equal to `gates/spark.hashes.json` |
| 65,536 x 4, every stream on pages | `the engine's own and 3 more taking 1184 pages of 256 rows (8.69 GiB)` | 12/12 equal |
| 262,144 x 8, on pages | `the engine's own and 7 more taking 3075 pages of 256 rows (22.57 GiB)` | 12/12 equal |
| 262,144 x 8, on pages, `agent`'s levers and a 16 GiB host cache | `3447 pages of 256 rows (25.30 GiB)` | 12/12 equal |

- **Speed:** ABBA on one GB10 of fixed windows against every stream on pages, at the same window and stream
  count (2 rounds, bar 3%): decode code x1 -0.1%, chat x1 -0.1%, code x4 +1.5%, chat x4 +0.3%, cold 8K +0.3%, with
  every cell's token hash equal. The kit's verdict is no difference.
- **Mixed traffic:** at 262,144 x 8, six workers sent short chats beside one session that grew turn by turn to
  152,125 prompt tokens. There were 123 requests with 0 failed and up to 7 streams at once. With a named host
  cache each turn restored the previous one (133,402 of 152,125 tokens cached on the last turn).
- **Exhaustion:** twelve requests at once, each reserving about 263K tokens of a 787,200-token pool, ran two at a
  time while the rest waited. All 12 returned 200, and the pool's free tokens never fell below one reservation.
- **Soak:** 10 minutes at 262,144 x 8, 4 workers, 140 requests with 22 cancels, 0 failed, probe hashes constant.
- **With the disk tier:** on the recipe's tree `0272db7d` (`patches/disk` under `patches/dynkv`), `agent` at the
  GB10's sizes gives the same 12 hashes, serves the same mixed traffic with 0 failed while 8.2 GiB of states reach
  disk, and soaks STABLE for 10 minutes (136 requests, 0 failed).

On a 6000 a page of 256 rows is about 7.5 MiB (30 KB a token). The `agent` profile's 14.0 GiB budget is therefore
about 1,900 pages, roughly 488K tokens: one 262,144-token session beside about 225K tokens of other streams, or
eight shorter sessions. This is an estimate from the GB10's page size until a 6000 boots the profile.
