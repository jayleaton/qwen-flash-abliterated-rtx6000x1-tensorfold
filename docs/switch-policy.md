# Switch policy

A switch is an environment variable or flag a recipe sets to change engine behaviour. Every switch is one of two
kinds, and a profile states which.

## Product switches

A product switch is part of how the recipe is served.

- It has a real reason: a measured gain or a correctness need on named hardware, with the receipt that shows it.
- It is documented beside the profile that sets it: what it does, the receipt, and the hardware it applies to.
- The engine binary must list it; `recipe serve` (R2) refuses a switch the binary does not read.

## Experiment switches

An experiment switch exists only to test or bench a candidate.

- It is set only by a test or bench run, never by a served profile.
- It names an owner, a decision condition (the ABBA result that decides it) and an expiry date.
- After its ABBA it is removed: the winning behaviour becomes the default or a product switch, the loser is
  deleted.

## Not a switch

Topology, such as tensor-parallel versus pipeline-parallel, is an interface adapter chosen by the profile, not an
environment toggle.

The kit spec sets the details: the manifest fields that carry a switch's kind, owner and expiry, and how gates
enforce them.
