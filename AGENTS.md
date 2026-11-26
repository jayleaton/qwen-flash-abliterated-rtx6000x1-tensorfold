# Working on the recipe kit

Before editing, read [CONTRIBUTING.md](CONTRIBUTING.md) and any AGENTS.md in the folders you change.

- Run `make check` before every commit (the pre-commit hook calls `scripts/ci-local.sh`, which runs it; keep it
  under 5 s) and `make check-full` before opening or updating a pull request (`scripts/ci-local.sh --full`); both
  must be green.
- One job per pull request, into `main` from a topic branch. Never push to `main`; the maintainer merges.
- Every feature is for every recipe: no model, recipe or machine name in `kit/` or `testkit/` code.
- Recipes pin a kit revision and call it. Never copy a kit or testkit file into a recipe.
- Build against upstream TensorFold main: new templates and examples pin a recent main commit.
- Ask before anything that changes state outside this repo (installing packages, `sudo`, network, GPU machines).
- No GPU rentals and no runs on shared GPU machines without the maintainer's approval.
- Commit with your approved public identity and GitHub noreply address; no tool co-author or "generated" trailers.
