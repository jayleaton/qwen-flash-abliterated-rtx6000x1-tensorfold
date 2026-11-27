# make check: the fast checks every commit passes (the pre-commit hook). make check-full: those and the slow
# tests (every selftest, the mock bench, bundle and soak), before a pull request.
RUFF ?= uvx ruff@0.17.6
BASE ?= origin/main
MAX_LINES := 600
SHELL_SCRIPTS := bin/recipe $(wildcard scripts/*.sh)
# The template parts a hosted recipe does not own: the real-names scan covers these and passes the rest.
TEMPLATE_PARTS := kit testkit templates hardware docs tests scripts bin pyproject.toml

# Every recipe this repository hosts beside the kit: its tests run with the kit's own. The kit's test
# fixtures are not recipes.
HOSTED_RECIPES := $(patsubst %/recipe.toml,%,$(shell git ls-files -co --exclude-standard -- '*/recipe.toml' ':(exclude)tests'))

.PHONY: check check-full lint shell test test-full imports scrub names lines whitespace

# The fast checks run side by side; the slow tests after them, one at a time (a busy CPU moves bench verdicts).
check:
	@$(MAKE) --no-print-directory -j4 lint shell test imports scrub names lines whitespace

check-full: check
	@$(MAKE) --no-print-directory test-full

lint:
	$(RUFF) check .
	$(RUFF) format --check .

shell:
	shellcheck $(SHELL_SCRIPTS)

test:
	PYTHONPATH=kit:testkit/src python3 -m unittest discover -s tests
	PYTHONPATH=testkit/src python3 -m unittest discover -s testkit/tests
	@for recipe in $(HOSTED_RECIPES); do \
		PYTHONPATH=kit:testkit/src python3 -m unittest discover -s "$$recipe/tests" || exit 1; \
	done

test-full:
	PYTHONPATH=kit:testkit/src python3 -m unittest discover -s tests -p 'full_*.py'

imports:
	! grep -rnE '^\s*(from|import)\s+recipe_kit' testkit

scrub:
	bin/recipe publish-check .

names:
	python3 scripts/real_names.py $(TEMPLATE_PARTS)

lines:
	python3 scripts/max_lines.py $(MAX_LINES)

whitespace:
	git diff --check $$(git merge-base $(BASE) HEAD)
	git diff --check --cached
