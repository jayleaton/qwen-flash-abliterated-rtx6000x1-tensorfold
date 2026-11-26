#!/usr/bin/env bash
# The local CI: make check, as the pre-commit hook runs it; with --full, make check-full, before a pull request.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# A hook's GIT_DIR and the like would point the tests' own git repos at this one: git finds it from the cwd instead.
# shellcheck disable=SC2046
unset $(git rev-parse --local-env-vars)
if [[ ${1:-} == --full ]]; then
  exec make check-full
fi
exec make check
