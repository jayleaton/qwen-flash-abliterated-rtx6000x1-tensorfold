# Hardware profiles

A hardware profile holds the facts every recipe on that hardware shares, so a recipe on hardware another recipe
already uses supplies nothing for it. A recipe names one with `[hardware] profile = "NAME"`, which reads
`hardware/NAME.toml`, or an `[engine.NAME] hardware` for the profiles on that engine:
[`rtx6000x2`](rtx6000x2.toml), [`gb10x2`](gb10x2.toml), [`gb10x1`](gb10x1.toml),
[`rtx5070tix1`](rtx5070tix1.toml) and [`apple-silicon`](apple-silicon.toml).

```toml
[chip]
arch = 121            # CUDA compute capability as an integer, or the HIP or Metal architecture as a string
gpu = "spark"         # the engine's build target: -Dgpu on CUDA, -Dcpu on Metal
ranks = 2             # devices a served model spans
unified = true        # host and device share memory

[comm]
transport = "roce"    # pcie-p2p, nvlink, roce or none
# Names only. Values that identify a machine (interfaces, devices, GIDs) are read on the node at launch, never
# written into a repo or a receipt: "@name" asks the node for it.
env = { NCCL_IB_HCA = "@rdma-devices", NCCL_CUMEM_ENABLE = "0" }

[memory]
floor_gib = 4.0       # soak fails below this much free memory: host MemAvailable when unified, else the emptiest card
gate_gib = "by-recipe"
drop_caches = true    # drop the page cache before each bench arm and soak, where permitted

[thermal]
pause_c = 92          # work pauses
resume_c = 86         # work resumes
hard_c = 93           # the run stops when this holds for hard_hold_s
hard_hold_s = 30

[power]
limit_w = 600         # each card's power limit
clock_max_mhz = 2100  # the GPU clock cap

[bench]
note = "numbers from here do not predict a different card; ask for a run on it"
```

Rules:

- One file a hardware class (chip, count and interconnect), named for it, such as `rtx6000x2.toml`.
- Only facts of the hardware. Model switches stay in the recipe's profiles.
- No host names, addresses, ports, serials or credentials: `recipe publish-check` scans these files too.

| `@name` | Read on the node as |
| --- | --- |
| `@rdma-netdev` | the first up netdev (by name) with an RDMA device whose port holds a RoCE v2 IPv4 GID |
| `@roce-v2-gid` | that GID's index |
| `@rdma-devices` | every such RDMA device with a RoCE v2 GID at that index, comma-separated: a Spark's twin PCIe link is a second rail |

A name already set in the environment keeps its value, so a node pins its own; `recipe serve --dry-run` prints the
`@name` unread.
