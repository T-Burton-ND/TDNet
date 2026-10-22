#!/usr/bin/env bash
# Freeze all study code and configuration at one committed source revision.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STUDY_ROOT=/groups/bsavoie2/tburton2/TDNet/scientific_120_hps_v1
SNAPSHOT="${STUDY_ROOT}/code_snapshot"
if [[ -e "${SNAPSHOT}" ]]; then
  echo "Snapshot already exists; refusing to overwrite it" >&2
  exit 2
fi
if [[ -n "$(git -C "${ROOT}" status --porcelain --untracked-files=no -- src scripts configs publication/2026/research/scientific_120_hps_v1)" ]]; then
  echo "Commit study sources before freezing the snapshot" >&2
  exit 2
fi
commit="$(git -C "${ROOT}" rev-parse HEAD)"
tmp="${STUDY_ROOT}/code_snapshot.tmp.${commit:0:12}"
mkdir -p "${tmp}"
git -C "${ROOT}" archive "${commit}" src scripts configs publication/2026/research/scientific_120_hps_v1 | tar -xf - -C "${tmp}"
printf '%s\n' "${commit}" > "${tmp}/source_commit.txt"
mv "${tmp}" "${SNAPSHOT}"
echo "Frozen HPS source at ${commit}"
