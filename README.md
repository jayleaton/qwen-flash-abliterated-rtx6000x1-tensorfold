# TensorFold recipe kit

This repository hosts one recipe beside a copy of the kit: [`qwen-flash-abliterated-rtx6000x1/`](qwen-flash-abliterated-rtx6000x1/README.md),
the abliterated Qwen3.8 Flash Next NVFP4 recipe for one RTX PRO 6000, migrated from the private
`qwen-flash-nvfp4-rtx6000-tensorfold` recipe. Its operations are in the repository RUNBOOK.

The kit itself is a plain copy of the private `tensorfold-recipe-kit` at `main` **759c8b4e0abc**
(`759c8b4e0abcecdc4c67d1712310ca7aed6cf4ea`, 2026-11-26): not a fork, not a submodule, no history carried,
and the upstream kit takes no pull request from here. Every change to this repository lands by pull request
into `main`, which is protected. The recipe's `[kit] rev` names a commit of this repository: a run from a
checkout of exactly that revision is clean, and a run from any other checkout passes `--kit-dev`, which
stamps its receipts `dev:`.

Because a real recipe is hosted here, the real-names scan covers the template parts only
(`make names`, `TEMPLATE_PARTS` in the Makefile) and `make test` runs every hosted recipe's own tests with
the kit's.

The shared kit for TensorFold recipes. A recipe is one `recipe.toml` plus only what is the model's own; the kit
fetches and patches the engine, builds it, and (in later steps) serves, gates, benches, soaks and publishes it.

| Path | What |
| --- | --- |
| `kit/` | the `recipe` command |
| `testkit/` | the shared test framework, its own package and version: the served gates, ABBA, soak, receipts and a stand-in server |
| `hardware/` | shared hardware profiles; [the format](hardware/README.md) |
| `templates/recipe/` | [the starting recipe](templates/recipe/README.md) that `recipe new` writes, pinned to upstream TensorFold main |
| `docs/` | [the manifest schema](docs/SCHEMA.md) and [the switch policy](docs/switch-policy.md) |
| `tests/fixtures/` | two synthetic recipes the tests run: `example-single` (two cards, modes, arms, recorded ABBA, soak and replies) and `example-multi` (five engines on four backends) |

## Commands

Run `bin/recipe` from a clean kit checkout at the revision the recipe's `[kit] rev` names; any other revision
refuses to run unless `--kit-dev` is passed, and the output is then stamped `dev:`.

| Command | Does |
| --- | --- |
| `recipe new DIR` | write a new recipe from `templates/recipe` (`recipe.toml`, `gates/`, `README.md`, a `tests/` stub), named for DIR; `--hardware NAME` (default `rtx6000x2`) and `--engines a,b` (default `cuda`; each one after the first inherits it) |
| `recipe fetch DEST` | fetch the pin, apply the series, refuse any tree other than the pinned one, and record the env names it reads in `.recipe/engines/`; a DEST this pin and series made, unedited, is reused with no fetch |
| `recipe prepare DEST` | `fetch`, then build in place (zig's cache keeps a rebuild incremental): the zig version, the toolchain, each step and each output, then record the binary's capabilities beside the names; `--no-addons` skips the addons |
| `recipe update-engine DEST` | move the pin to upstream main's head (or `--rev`), apply the series, record the new tree |
| `recipe check [DIR]` | rules C1 to C9 over modes, switches, profiles and arms; C8 against each engine's names that `recipe fetch` recorded, or `--tree DIR` |
| `recipe serve TARGET ENGINE_DIR MODEL` | check, capabilities check, then start every rank of a profile or arm; `--dry-run` prints them |
| `recipe gate [TARGET]` | the served gates against `--url` (default the recipe's port on this host): conformance with every plant, token hashes, nondeterminism, canary, MMLU and tool chains; `--quick` runs greedy hashes only, `--skip`/`--only` choose gates, `--mock` runs them against the stand-in |
| `recipe bench A B ENGINE_DIR MODEL` | ABBA two profiles or arms on this host inside a control arm, below; `--quick` runs one A B pass, `--mock` runs every arm on the stand-in |
| `recipe soak TARGET ENGINE_DIR MODEL` | mixed traffic with cancels on one fresh server for `[soak] minutes`, at the hardware profile's floors, below; `--mock` soaks the stand-in |
| `recipe kit DEST ENGINE_DIR` | bundle the recipe's files, the built engine, the pinned tree's tools and this kit with `MANIFEST.txt` and `SHA256SUMS`, plus a `.tar.gz`; `--verify BUNDLE` checks one |
| `recipe env FILE...` | print `export` lines for KEY=VALUE files; a variable already set wins |
| `recipe publish DEST` | write the public recipe tree from this recipe's committed files (`--rev`, default HEAD): all but `[publish] exclude`, plus the pinned kit release with its sha256 and `kit/recipe`, which checks it before every run; any publish-check finding, in the tree or the release, writes nothing |
| `recipe publish-check [ROOT]` | scan a repo, and with `--history` its commits, for private details and credentials |

`fetch`, `prepare` and `update-engine` take `--recipe DIR` (default `.`), `--engine NAME` when the recipe has
several backends, `--kit-dev`, and `--source PATH` for a local TensorFold clone or bundle that holds the pin.

`update-engine` does not yet re-run gates: run `recipe gate` and `recipe bench` on the new pin before committing it.

## Walkthrough

A new recipe for two RTX PRO 6000 cards, from a clean kit checkout (`recipe` is its `bin/recipe`):

```sh
recipe new my-model --hardware rtx6000x2             # then set [recipe] model, model_type, format and [weights]
cd my-model
recipe update-engine ENGINE_DIR                      # pin upstream main's head and record its tree
recipe prepare ENGINE_DIR                            # fetch, patch, build; record the names and capabilities
recipe check                                         # C1 to C9, C8 from that record
recipe serve default ENGINE_DIR MODEL_DIR            # --dry-run prints each rank's argv and env
recipe gate --quick                                  # greedy hashes against the running server
recipe bench default fast ENGINE_DIR MODEL_DIR       # ABBA of two profiles or arms inside a control arm
```

The same steps run without a GPU on the synthetic `tests/fixtures/example-single` (made-up `org/Example-Model-FP8`,
a stand-in engine tree with upstream's bench and conformance clients, and recorded replies):

```sh
fx=tests/fixtures/example-single
recipe check $fx --kit-dev --tree $fx/engine         # 0 findings over 8 profiles and arms
recipe gate --recipe $fx --kit-dev --quick --mock    # PASS hashes: 2/2 equal
recipe bench --recipe $fx --kit-dev base fast $fx/engine --mock --cells decode.code.x1,cold.1024
```

`recipe serve` needs a built engine; `tests/test_serve_single.py` serves every target with a stand-in binary and
holds each rank's argv and env to `serve.json`, the same check a real recipe makes in its own repo.

## Gates

`recipe gate` reads a running server's URL, so it runs against any live server. Every gate can be skipped, and a
gate whose input is missing is skipped and says why.

| Gate | Passes when | Its planted fault (the stand-in's `--fault`) |
| --- | --- | --- |
| `conformance` | the pinned tree's `tools/conformance.py` passes the recipe's profile, and each `--plant` turns its gate red; `resumed` and `evict` need `--fresh-url`, a server started with `--prompt-cache-gib 0` | drafted replies differ from plain ones |
| `hashes` | each prompt's greedy and seeded reply carries the token hash in `gates/hashes.json`; with every seed moved by one it turns red | a recorded reply comes back with other tokens |
| `nondeterminism` | 8 copies of one greedy and 8 of one seeded request, sent at once and seen open together in `GET /v1/load`, return one output a mode | a request beside another returns one of two outputs |
| `canary` | three nonced probes hold their answers and do not look broken | wrong answers |
| `mmlu` | `--mmlu FILE` rows answered right reach `[gates] mmlu200_min` (default 0.7) | wrong letters |
| `tools` | each `[gates] tool_chains` chain calls its functions with JSON arguments and ends on its answer | arguments that are not JSON |

`python -m tensorfold_testkit selftest` runs every gate on the stand-in, green clean and red with its own fault,
and each conformance plant on a stand-in of its own.

## Bench

`recipe bench A B` runs `control A B B A control` (`[bench] rounds = 2`), each a fresh server: the page cache is
dropped before each boot where this user may (else the receipt says why not), every rank starts in its own process
group exactly as `recipe serve` starts it (an arm that needs the prompt cache off sets `prompt-cache-gib = 0`), and
the upstream bench clients of the pinned tree run the cells; the receipt names each arm's prompt cache. A cell is `decode.SET.xN` (`bench_openai` at 1 stream, `bench_concurrent` above it) or `cold.SIZE` (a
cold prefill of `gates/hashes.json`'s `cold` prompt, tok/s). Each cold sample puts a prefix no earlier request had
before the prompt (16 of its own token ids, picked by a per-run salt), so it misses the cache without turning the
cache off; the client requires the server to report it cold. A control equal to A or B runs as `NAME-control`.

| Verdict | When |
| --- | --- |
| `tokens differ`, `unmeasured` | a cell's token hash is not the recorded one, or a client failed |
| `void` | the control moved more than the bar on any cell between its ends |
| `worse` | B is at least the bar below A on any cell |
| `better` | B is at least the bar above A on the first cell |
| `no difference` | otherwise |

The receipt (`receipt.json`, also written by `recipe gate --receipt`) records the kit, recipe and engine, each arm's
settings and binary md5 and sha256, the order, every sample and the verdict, and holds no path or address.
`python -m tensorfold_testkit decide DIR A B` re-decides a run from its cells.

## Soak

`recipe soak TARGET` starts the profile or arm as `recipe serve` renders it (its own `--parallel`), drops the page
cache where the hardware profile says to, and runs mixed traffic for `[soak] minutes` (30): code, prose, structured,
tool, thinking, long-document, non-streamed and disconnecting requests from up to `[soak] workers` (4) clients, about
15% of streams cancelled mid-reply. Every `[soak] probe_min` (5) minutes the traffic pauses for the fixed probes:
greedy code and chat at one stream and one cold prompt, through the pinned tree's bench clients.

| STABLE needs | Read from |
| --- | --- |
| no failed request, no rank exit, no engine error in the rank logs | the clients, the ranks, their logs |
| probe token hashes constant, and the recipe's where it records them | `gates/hashes.json` |
| every slot free at the end | `GET /v1/load`: nothing running or waiting |
| the canary answered | `17*23` |
| free memory above `[memory] floor_gib`; under `[thermal] hard_c` for `hard_hold_s` | the hardware profile |

A card under `[power] limit_w` is noted. The receipt (`receipt.json`, kind `soak`) holds the record and the verdict.
`python -m tensorfold_testkit soak --replay DIR` judges a soak.sh log directory the same way.

## Publish

`recipe publish DEST` writes the public recipe from the private one, so there is one repo to keep and nothing to
drift. It takes the committed tree at `--rev` (never untracked files, and never `.check-public.local`), leaves out
`[publish] exclude`, and adds `kit/`: the release of the kit revision `[kit] rev` pins (`git archive` of that
commit's `bin`, `kit`, `testkit`, `hardware`, `templates` and README, with a `REVISION` file, the same bytes on every run), its
`SHA256SUMS` and `kit/recipe`, which checks the sum before it unpacks and runs the release. Every file that goes
out, each file of the release included, is scanned with publish-check's rules, the recipe's `.check-public.local`
and `.check-public.allow`; on any finding nothing is written. A file kept private by an exclude is not scanned. The
public tree has no history: commit it as one change to the public repo, which is a step for a person to take.
