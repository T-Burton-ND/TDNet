#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 FREEZE_MANIFEST OUTPUT_SIGSTORE_BUNDLE" >&2
  exit 2
fi

cosign sign-blob --yes --bundle "$2" "$1"
