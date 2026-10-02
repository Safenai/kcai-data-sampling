#!/usr/bin/env bash
set -euo pipefail

GLOB="${1:?Usage: $0 <dist-glob> [publish-url]}"
URL="${2:-https://test.pypi.org/legacy/}"

OUT=$(uv publish "$GLOB" --publish-url "$URL" 2>&1) && RC=0 || RC=$?

if [ "$RC" -eq 0 ]; then
  printf '%s\n' "$OUT"
  exit 0
fi

if printf '%s' "$OUT" | grep -q "File already exists"; then
  echo "::warning::$GLOB already published ($URL); continuing."
  exit 0
fi

printf '%s\n' "$OUT"
exit "$RC"