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
below for decode throughput on one card; the `agent` profile trades width for window (139,264 tokens, 4
streams) instead.
