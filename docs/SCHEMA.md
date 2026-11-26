# `recipe.toml`

The kit reads one file. Unknown tables and keys are refused, so a typo cannot pass for a setting nobody reads.
`recipe new DIR` writes one from [the template](../templates/recipe/recipe.toml). Worked, synthetic manifests:
[example-single](../tests/fixtures/example-single/recipe.toml) (two cards, inherited profiles, modes, switches and
arms) and [example-multi](../tests/fixtures/example-multi/recipe.toml) (five engines on four backends, inherited
builds, `false` drops and `when_model_has`).

| Step | Command | Reads |
| --- | --- | --- |
| new | `recipe new DIR [--hardware NAME] [--engines a,b]` | the template; writes `[recipe] name`, `[hardware]`, `[engine.*]` and `[kit] rev` |
| prepare | `recipe prepare ENGINE_DIR` | `[engine.NAME]` and its build; writes `.recipe/engines/NAME.json` |
| check | `recipe check [DIR]` | profiles, arms, `[[mode]]` and `[[switch]]`; C8 from that record |
| serve | `recipe serve TARGET ENGINE_DIR MODEL_DIR` | the target's profile, its engine's hardware and the binary's capabilities |
| gate | `recipe gate [TARGET]` | `[gates]` |
| bench | `recipe bench A B ENGINE_DIR MODEL_DIR` | `[bench]`, and `gates/hashes.json`'s `cold` rows |

On `tests/fixtures/example-single`, `recipe check --tree engine`, `recipe gate --quick --mock` and `recipe bench
base fast engine --mock` run without a GPU (each with `--kit-dev`).

| Table | Key | Type | Req | Meaning |
|---|---|---|---|---|
| `[recipe]` | `name` | string | yes | the recipe's public id |
| | `model` | string | yes | the checkpoint's repository id |
| | `model_type`, `format` | string | yes | a key of the binary's `capabilities --json` `families`, and one of its formats |
| | `ships` | [string] | yes | the profiles a user is told to run |
| | `served_name` | string | no | `--name`; default `name` |
| | `port` | int | no | rank 0's port, rank N takes port + N; default 8080 |
| `[kit]` | `rev` | 40 hex | yes | the one kit revision this recipe runs with |
| | `testkit` | string | yes | a version range, such as `>=0.1,<1` |
| `[engine.NAME]` | `repo`, `rev`, `tree`, `series` | | yes | one table a pin; `series` lists `DIR` or `DIR:NNNN` |
| | `inherits` | string | no | another engine; its keys fill these, tables (`build`, `vars`) key by key, and `false` drops one |
| | `backend` | string | no | the `capabilities --json` backend its binary has; default `NAME` |
| | `hardware`, `format` | string | no | this engine's hardware profile and weight format; default `[hardware]` and `[recipe]`'s |
| `[engine.NAME.build]` | `vars`, `executables`, `steps`, `outputs`, `[[addon]]` | | no | the first output is the server binary |
| `[weights]` | `rev`, `sha256` | string, path | yes | checkpoint revision and checksum file |
| `[hardware]` | `profile` | string | yes | the kit's `hardware/NAME.toml` |
| `[profile.NAME]` | `engine` | string | when several | which `[engine.NAME]` it runs on |
| | `inherits` | string | no | another profile; keys here override it, `flags`, `when_model_has`, `mode` and `settings` key by key, and `false` drops an inherited one |
| | `topology` | table | yes | `{}`, `{ tensor = N }` or `{ pipeline = N }`: `--tp N` or `--pp N`, and N ranks |
| | `context`, `parallel` | int | yes | `--context`, `--parallel` |
| | `flags` | table | no | other serve flags by name without `--`; `true` passes the bare flag, `"{model}"` the model directory |
| | `when_model_has` | table | no | `flag = "FILE"`: pass that flag only when the model directory holds FILE |
| | `mode` | table | no | `slot = "value"`, each matching a `[[mode]]` row |
| | `settings` | table | no | product switch names with values |
| `[arm.NAME]` | `base`, `experiments` | string, table | no | a bench arm: a profile plus experiment switches |
| `[[mode]]` | `slot`, `value`, `applies`, `sets`, `receipt`, `expires` | | | an adapter value; `applies` is single, tensor, pipeline or any |
| `[[switch]]` | `name`, `kind` | | | `product`: `reason`, `doc`, `hardware`, `receipt`; `experiment`: `owner`, `decides`, `added`, `expires` |
| `[gates]` | `conformance` | path | yes | a profile in the format of upstream's `tools/conformance_profiles/` |
| | `hashes` | path | yes | expected token hashes, below |
| | `mmlu200_min` | number | no | the share of `--mmlu` rows the mmlu gate must answer right; default 0.7 |
| | `tool_chains` | path | no | tool chains for the tools gate |
| | `checks` | path | no | read when format checks land |
| `[bench]` | `cells` | list | no | the cells `recipe bench` measures, the first deciding `better`; default `decode.code.x1`, `decode.chat.x1` |
| | `rounds` | int | no | samples an arm, 1 or 2 (default 2) |
| | `bar` | number | no | the share a cell must move to count (default 0.03) |
| | `control` | bool | no | run the first shipped profile at both ends (default true) |
| `[publish]` | `exclude` | [string] | no | paths `recipe publish` keeps private: `DIR/` and everything under it, or a glob over the whole path |
| `[soak]` | `minutes` | number | no | how long `recipe soak` runs (default 30) |
| | `workers` | int | no | client threads, 1 to this many at once (default 4) |
| | `probe_min` | number | no | minutes between the fixed probes (default 5) |

## Rendering

`recipe serve TARGET` renders the profile, or the arm's base profile, as
`serve MODEL --name N --context C --parallel P --tp|--pp N FLAGS... EXTRA... --master 127.0.0.1 --rank R
--port PORT+R`, one process a rank, with no flag or setting the recipe does not render (a recipe that wants
`--no-update-check` sets it); a single rank gets `--port PORT` and no `CUDA_VISIBLE_DEVICES` unless
`--gpus` names one. The env is the hardware profile's `comm.env`, then each mode's `sets`,
then `settings`, then the arm's `experiments`; a name set by two of them is refused. A variable already in the
environment keeps its value, and an `@name` value is read on the node at launch ([hardware](../hardware/README.md));
`--dry-run` prints it unread.

## `recipe check`

| Rule | Fails when |
|---|---|
| C1 | a row lacks a field its kind needs, or carries one it may not |
| C2 | a profile or arm sets a name with no `[[switch]]` row, or an arm adds a non-experiment |
| C3 | an experiment or mode is past `expires` (`--today` sets the date) |
| C4 | an experiment runs more than 90 days, or has no owner or decision |
| C5 | a profile sets an experiment |
| C6 | a mode value has no row, or does not apply to the profile's topology |
| C7 | a row no profile or arm uses |
| C8 | a pinned tree neither registers nor reads a name its engine's profiles and arms set: each engine `recipe fetch` recorded, or `--tree DIR` (the engine whose pinned tree it is, or `--engine NAME`) |
| C9 | a row is marked `exact = false` |

`recipe serve` runs the check against the engine tree, then the capabilities check against the binary (schema, family
and format, backend, the hardware's SM, every rendered flag), and starts nothing if either fails.

## `.recipe/engines/NAME.json`

`recipe fetch` records the env names the pinned tree reads, and `recipe prepare` also the built binary's
`capabilities --json` and sha256. `recipe check` runs C8 from these records; `recipe serve` takes the recorded
capabilities when the binary is the one recorded, else asks it. A record holds for its `tree` only, so a moved pin
ignores it. A record not written by those commands says how it was made in `by`.

## `gates/hashes.json`

```json
{"schema": 1, "tokens": 256, "greedy_seed": 1234, "seeds": [1234, 1235],
 "sampled": {"temperature": 0.7, "top_k": 20, "top_p": 0.95},
 "prompts": {"code": {"kind": "completion", "prompt": "..."}, "chat": {"kind": "chat", "prompt": "...", "thinking": false}},
 "expected": {"*": {"code": {"greedy": "153389030f94", "sampled": ["0a9d1d6f3e4d", "2106ea441ef4"]}}}}
```

Each request is upstream `tools/bench_openai.py`'s: streamed, `ignore_eos`, `max_tokens` = `tokens`, the greedy one
with `greedy_seed`, the sampled ones with `sampled` and one seed each. A `chat` prompt with `thinking` false sends
`enable_thinking: false`. A hash is `tensorfold.token_sha256` (64 hex) or the 12-hex `token_sha` bench receipts
print; a 12-hex hash matches a reply whose 64-hex one starts with it. `expected` is keyed by profile or arm, `*`
for every one without its own row.

An optional `cold` table names the bench's `cold.SIZE` cells: `{"8192": {"prompt": "../prompts/ids-8192.json",
"tokens": 16, "hash": "<64 hex>"}}`, the prompt a JSON list of token ids relative to this file, `tokens` the reply
length and `hash` (optional) the unsalted prompt's token hash, for the record. The bench salts each cold sample with
a prefix of the prompt's own ids, so a salted reply has its own tokens; bench_cache holds it to being cold and to
its repeated and concurrent replies equalling the first.
