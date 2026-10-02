#!/bin/bash
set -e

mkdir -p "$(dirname "$0")/../logs"
timestamp=$(date +%Y%m%d_%H%M%S)
exec > >(tee "$(dirname "$0")/../logs/code_quality_${timestamp}.log") 2>&1

CI_MODE="${CI:-false}"

# S8541: --frozen locks dependency resolution; --no-build breaks editable installs
uv sync --frozen

if [[ "$CI_MODE" != "true" ]]; then
    # fmt failure must not stop the gate: sort_imports cannot auto-fix F401/
    # syntax leftovers, but the remaining sessions still report the rest.
    uv run --frozen nox -s fmt || echo "!! fmt failed (continuing)"
fi

uv run --frozen nox -s lint
uv run --frozen nox -s type_check
uv run --frozen nox -s spell