#!/bin/bash
set -e

mkdir -p "$(dirname "$0")/../logs"
timestamp=$(date +%Y%m%d_%H%M%S)
exec > >(tee "$(dirname "$0")/../logs/complexity_${timestamp}.log") 2>&1

CI_MODE="${CI:-false}"

# S8541: --frozen locks dependency resolution; --no-build breaks editable installs
uv sync --frozen

uv run --frozen nox -s complexity
uv run --frozen nox -s test_complexity