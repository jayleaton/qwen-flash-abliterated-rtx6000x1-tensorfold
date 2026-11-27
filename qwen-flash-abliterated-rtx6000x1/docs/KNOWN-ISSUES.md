# Known issues (ported from the old recipe, plus this migration's own)

Ported rows are **measured on the old pin**; the migration rows are this recipe's, open now.

## The shipped weights load through the `patches/loader` series

The abliterated pack keeps the control pack's quantization values (ModelOpt NVFP4) but states them as
`quant_algo: NVFP4` with an ignore list, and stores its MTP experts fused in BF16; upstream main refuses both.
The two loader patches teach the config check the stated layout and host-quantize the fused experts into the
FP8 form the head already serves (nvidia's own layout, at FP8 rounding noise). Served tokens do not depend on
the draft head, so the rounding can change acceptance, never tokens. On one GB10 the pack loads, the
conformance gate's `drafted` check (drafted equals plain) passes, and the head drafts 4.27 tokens a round on
greedy code (nvidia's pack: 4.41 on a 6000), so the layout is right.

## Not soak-tested

No NVFP4 profile, old or new, has run a 30-minute soak of mixed traffic. On one GB10 a 10-minute soak of the
shipped weights is STABLE (133 requests, 20 cancels and disconnects, 0 failed, probe hashes constant); the
RUNBOOK's R4 step records the first 30-minute one on the 6000.

## The codex32 cliff

At 32 streams decode falls off the drafting rounds' 64-row cap shared across streams (each stream drafted
129-145 tokens a request against 205-215 at 16). One card does not fit 32 streams at the served windows
([CAPACITY](CAPACITY.md)); keep `--parallel` at 16 or below.

## `qmmf-prompt-mx-contract` segfaults

On sm_120 the `qmmf-prompt-mx-contract` kernel contract exits rc 139; the other contracts pass. It belongs to
a lane no profile turns on. Not investigated; the RUNBOOK's verify step re-runs the contracts on the new pin.

## Nothing is measured on this pin

The engine pin moved from the old recipe's `924d50a9d` + 15 patches to upstream main `1a3a10d9` with no
series, and the shipped weights differ from the control pack. Every expected lever is an experiment arm;
every hash table is empty until the GPU run fills it. The old pin's numbers stay labeled as such in
[PERFORMANCE](PERFORMANCE.md).
