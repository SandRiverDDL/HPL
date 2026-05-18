#!/usr/bin/env bash
set -euo pipefail

BASHRC_PATH="${BASHRC_PATH:-${HOME}/.bashrc}"

BEGIN_MARKER="# >>> HPL AutoDL Codex defaults >>>"
END_MARKER="# <<< HPL AutoDL Codex defaults <<<"

if [[ ! -f "${BASHRC_PATH}" ]]; then
  echo "No ${BASHRC_PATH}; nothing to remove."
  exit 0
fi

tmp_file="$(mktemp)"
trap 'rm -f "${tmp_file}"' EXIT

awk -v begin="${BEGIN_MARKER}" -v end="${END_MARKER}" '
  $0 == begin {skip=1; next}
  $0 == end {skip=0; next}
  $0 == "# AutoDL local user tools" {skip_legacy_path=1; next}
  skip_legacy_path == 1 && $0 ~ /^export PATH=\/root\/\.local\/bin:/ {skip_legacy_path=0; next}
  $0 == "# AutoDL Codex reverse SOCKS proxy" {skip_legacy_proxy=3; next}
  skip_legacy_proxy > 0 {skip_legacy_proxy--; next}
  skip == 0 {print}
' "${BASHRC_PATH}" > "${tmp_file}"

cp "${tmp_file}" "${BASHRC_PATH}"

echo "Removed HPL AutoDL Codex PATH/proxy block from ${BASHRC_PATH}"
echo "Reload with: exec bash"
