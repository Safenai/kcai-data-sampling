#!/bin/bash
set -e

mkdir -p "$(dirname "$0")/../logs"
timestamp=$(date +%Y%m%d_%H%M%S)
exec > >(tee "$(dirname "$0")/../logs/test_coverage_${timestamp}.log") 2>&1

CI_MODE="${CI:-false}"

# S8541: --frozen locks dependency resolution; --no-build breaks editable installs.
# UV_INDEX= keeps the private GitLab index (401 without credentials) out of resolution.
UV_INDEX= uv sync --frozen

UV_INDEX= uv run --frozen nox -s test_coverage