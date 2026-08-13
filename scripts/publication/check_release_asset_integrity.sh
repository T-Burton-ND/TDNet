#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 FREEZE_BUNDLE" >&2
  exit 2
fi

cd "$1"
sha256sum --check SHA256SUMS
