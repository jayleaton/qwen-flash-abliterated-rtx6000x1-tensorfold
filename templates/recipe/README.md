# A new recipe

A recipe is one `recipe.toml` plus only what is the model's own: its patch series while the family is not
upstream, its prompts and format checks, and bars that differ from the kit's defaults. The kit runs the rest.

## Start

1. `recipe new DIR [--hardware NAME] [--engines a,b]` writes this directory into DIR, names the recipe for it and
   pins `[kit] rev` to the kit you run it from. Make DIR a private repo and set `[recipe] model`, `model_type`,
   `format` and `[weights]`. `tests/test_recipe.py` is the recipe's own test, run with the kit's testkit.
2. Move the engine pin to upstream main's head:

   ```sh
   recipe update-engine --recipe . /tmp/engine-check
   ```

   It fetches main, applies `series` and writes the new `rev` and `tree` into `recipe.toml`.
3. Build it:

   ```sh
   recipe prepare --recipe . engine
   ```

   `prepare` fetches the pin, applies the series, refuses any tree other than `tree`, then runs the build steps
   and checks each output exists. `NVCC`, `GPU` and `XGRAMMAR` in the environment override the defaults.
4. Start the server, then record its hashes: `python3 ENGINE/tools/bench_openai.py URL NAME --tokens 256
   --temperatures 0,0.7` and copy each prompt's `token_sha` values into `gates/hashes.json` (`expected["*"]`). Then `recipe gate --tree ENGINE` holds every later build to them; `--quick` is the inner loop.
5. Before publishing anything, `recipe publish-check .` must be clean. Private names (hosts, users, internal
   tools) go in `.check-public.local`, which is git-ignored: one extended regex a line, optionally followed by a
   tab and path globs that rule skips. Upstream identifiers a rule mistakes for private names go in the tracked
   `.check-public.allow`.

## Patches

Put the family's patches in `patches/<backend>/0001-*.patch` onwards (`git format-patch` output) and list the
directory in `series`. Each directory must hold exactly 0001..N with no gaps. Several directories apply in order,
so a backend can stack its own series on a shared one: `series = ["patches/cuda:0022", "patches/rtx"]`.

## Switches

Read [the switch policy](../../docs/switch-policy.md) before adding an env switch to a profile.
