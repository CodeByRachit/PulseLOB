#!/usr/bin/env bash
# Verifies that VERSION is valid SemVer, that CHANGELOG.md has a matching release
# section and link, and (optionally) that a release tag matches VERSION.
#   scripts/check_version.sh            # consistency check
#   scripts/check_version.sh v0.1.0     # additionally require tag == v$VERSION
set -euo pipefail
cd "$(dirname "$0")/.."

version="$(tr -d '[:space:]' < VERSION)"
semver='^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(-[0-9A-Za-z.-]+)?(\+[0-9A-Za-z.-]+)?$'
if ! [[ "$version" =~ $semver ]]; then
  echo "error: VERSION '$version' is not valid SemVer" >&2
  exit 1
fi

if ! grep -qE "^## \[${version//./\\.}\] - [0-9]{4}-[0-9]{2}-[0-9]{2}$" CHANGELOG.md; then
  echo "error: CHANGELOG.md has no '## [$version] - YYYY-MM-DD' section" >&2
  exit 1
fi

if ! grep -qE "^\[${version//./\\.}\]: " CHANGELOG.md; then
  echo "error: CHANGELOG.md has no link reference for [$version]" >&2
  exit 1
fi

latest="$(grep -oE '^## \[[0-9][^]]*\]' CHANGELOG.md | head -n1 | sed -E 's/^## \[(.*)\]/\1/')"
if [[ "$latest" != "$version" ]]; then
  echo "error: newest CHANGELOG release is '$latest' but VERSION is '$version'" >&2
  exit 1
fi

if [[ $# -ge 1 ]]; then
  tag="$1"
  if [[ "$tag" != "v$version" ]]; then
    echo "error: tag '$tag' does not match VERSION (expected 'v$version')" >&2
    exit 1
  fi
fi

echo "version check passed: $version"
