# Contributing to the recipe kit

The kit is the shared code every TensorFold recipe runs: `kit/` is the `recipe` command, `testkit/` the test
framework, `hardware/` the shared hardware profiles and `templates/recipe/` the starting recipe.
Start from `main` and keep one change per pull request.

## Checks

```sh
make check        # every commit: the fast checks, under 5 s
make check-full   # before a pull request: make check and the slow tests
```

`make check` runs each of:

| Target | Checks |
| --- | --- |
| `lint` | `ruff check` and `ruff format --check` (ruff pinned in the Makefile, run with `uvx`) |
| `shell` | `shellcheck` on `bin/recipe` and `scripts/*.sh` |
| `test` | the kit's unit tests (`tests/`) and the testkit's (`testkit/tests/`) |
| `imports` | nothing under `testkit/` imports `recipe_kit`, so the testkit can move to its own repo |
| `scrub` | `recipe publish-check .` on this repo: no addresses, host names, home paths, keys or credentials |
| `names` | no real model, recipe or recipe repo name in any file or path (`scripts/real_names.py`); examples are made up |
| `lines` | no file over 600 lines |
| `whitespace` | `git diff --check` against the merge base with `origin/main`, and on the staged changes |

`make check-full` adds `test-full`: the `tests/full_*.py` files, which start servers and clients: every gate's
selftest, the mock bench from a bundle and the mock soak with each planted fault. A test that starts a stand-in
server or the bench clients belongs there; `tests/test_*.py` stay fast.

Requirements: Python 3.11 or newer, `uv`, `shellcheck` and `git`. Nothing else is installed; the kit and testkit
use only the standard library.

## Lean code

- Each module has one job and stays at or below 600 lines.
- Comments and docstrings are single lines of at most 120 columns. Measurements and history belong in a receipt
  or the pull request, not in source comments.
- The kit reads one `recipe.toml`. A new recipe need is a manifest field the kit understands, not a script the
  recipe carries.
- Prefer refactoring over compatibility shims when the change and its adopters land together.
- Write tests that show a real failure and its correction. Add only the tests the change needs.

## Proof

A change that can reach a recipe's engine or launch states its proof in the pull request: the recipe it was run
against, the command, and the before and after values (tree hashes, output hashes, rendered args). Builds run on
a machine with the toolchain; a GPU run needs the maintainer's approval.

## Versions

The kit (`pyproject.toml`) and the testkit (`testkit/pyproject.toml`) are versioned separately. Bump a version in
the pull request that changes what a recipe sees from it.

## Public identity and privacy

Use your approved public Git identity and GitHub noreply address; tool co-author trailers do not belong in the
history. Keep private e-mails, machine names, local user paths, addresses, ports of private services, keys and
confidential data out of everything committed. Test fixtures that must look private are assembled at run time so
`make scrub` stays clean without exceptions.
