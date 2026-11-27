# Performance (ported, measured on the old pin)

Every number here is ported from an earlier, unpublished one-GPU NVFP4 recipe's session and is
**measured on the old pin**: engine `924d50a9d` plus its 15-patch series (tree `3236c674a`), release binary
`15f909a78`, weights the control pack at `fc694b54`, one RTX PRO 6000 Blackwell. None of it is this
recipe's pin (`1a3a10d9` plus its series) or the shipped abliterated weights: this recipe's own numbers are in the
repository [README](../../README.md) and [RUNBOOK](../../RUNBOOK.md).

## One GPU (the `default` profile's source, old `tp1`)

Cold rows: one cold prompt each at a 139,264-token window, `--prompt-cache-gib 0`. Decode: N streams of the
code or chat prompt, 256 greedy tokens each, aggregate tok/s. Each figure is one run; A-vs-A noise was at
most 1.5%.

| arm | cold 8K | cold 32K | cold 128K | codex1 | codex4 | codex8 | codex16 | chatx1 | chatx4 | chatx8 | chatx16 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| one GPU | 8,818 | 8,963 | 8,528 | 291 | 923 | 1,453 | 1,607 | 200 | 663 | 1,045 | 1,311 |
| codex32 | does not fit (see [CAPACITY](CAPACITY.md)) | | | | | | | | | | |

## Token hashes (the reference this pin reproduces)

sha256 of the token ids: cold 8K `fa345c10f631`, 32K `b59d8e0ea7d8`, 128K `974fa8bcd14d`; decode code
`4e759dbd859c`, chat `764da77fbd3f` (full values in `gates/control.hashes.json`). The decode hashes matched
an earlier RTX PRO 6000 host's, and every concurrent reply equalled its alone and serial runs.

## Where a decode round's time goes (old pin, nsys)

| profile | compute only | idle (host, launch, sync) | compute ms a round |
| --- | --- | --- | --- |
| one GPU, 1 stream | 84% | 16% | 43.2 |
| one GPU, 16 streams | 85% | 15% | 105.1 |

A one-GPU round is compute-bound with no collectives: graphs and transport levers cannot pay here; the
levers that act are the prompt chunk, the draft chain and the batched lanes.

## Hardware (the old session's host)

Two RTX PRO 6000 Blackwell Workstation Edition cards, 600 W, SM clock up to 3,090 MHz (sustained about
2,830 MHz at about 418 W, no throttling), driver 595.112.07, AMD EPYC 9575F, Linux 6.8 in a container.
Memory 1.46 TB/s achieved; host to device 57 GB/s; dense BF16 557 / FP8 1,100 TFLOPS. Same-host ABBA only:
numbers from one host do not predict another's.
