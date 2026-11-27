# Known issues (ported from the old recipe, plus this migration's own)

Ported rows are **measured on the old pin**; the migration rows are this recipe's, open now.

## The shipped weights do not load yet

The abliterated pack keeps the control pack's quantization (ModelOpt NVFP4, `hf_quant_config.json`) but
stores per-layer expert shards (`layer-NNNNN-experts-XXXX-YYYY.safetensors` with `.complete.json` sidecars),
BF16 non-expert shards (`model-bf16-*`) and FP8 PLE shards (`model-plefp8-*`). The upstream loader reads the
control pack's `model-0000N-of-000M` layout, not this one. The loader series lands in `[engine.cuda] series`
when it arrives; until then the recipe serves the control pack.

## Not soak-tested

No NVFP4 profile, old or new, has run a 30-minute soak of mixed traffic. The RUNBOOK's soak step records the
first one on the shipped weights.

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
