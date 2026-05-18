#!/usr/bin/env bash
set -euo pipefail

BASHRC_PATH="${BASHRC_PATH:-${HOME}/.bashrc}"
PROXY_PORT="${1:-${CODEX_PROXY_PORT:-17891}}"
NODE_BIN_DIR="${NODE_BIN_DIR:-/root/autodl-tmp/tools/node-v24.14.0-linux-x64/bin}"

BEGIN_MARKER="# >>> HPL AutoDL Codex defaults >>>"
END_MARKER="# <<< HPL AutoDL Codex defaults <<<"

tmp_file="$(mktemp)"
trap 'rm -f "${tmp_file}"' EXIT

if [[ -f "${BASHRC_PATH}" ]]; then
  awk -v begin="${BEGIN_MARKER}" -v end="${END_MARKER}" '
    $0 == begin {skip=1; next}
    $0 == end {skip=0; next}
    $0 == "# AutoDL local user tools" {skip_legacy_path=1; next}
    skip_legacy_path == 1 && $0 ~ /^export PATH=\/root\/\.local\/bin:/ {skip_legacy_path=0; next}
    $0 == "# AutoDL Codex reverse SOCKS proxy" {skip_legacy_proxy=3; next}
    skip_legacy_proxy > 0 {skip_legacy_proxy--; next}
    skip == 0 {print}
  ' "${BASHRC_PATH}" > "${tmp_file}"
else
  : > "${tmp_file}"
fi

cat >> "${tmp_file}" <<EOF

${BEGIN_MARKER}
export PATH=/root/.local/bin:${NODE_BIN_DIR}:\$PATH
export ALL_PROXY=socks5h://127.0.0.1:${PROXY_PORT}
export HTTPS_PROXY=socks5h://127.0.0.1:${PROXY_PORT}
export HTTP_PROXY=socks5h://127.0.0.1:${PROXY_PORT}
${END_MARKER}
EOF

cp "${tmp_file}" "${BASHRC_PATH}"

echo "Configured Codex PATH and reverse SOCKS proxy in ${BASHRC_PATH}"
echo "Proxy endpoint: socks5h://127.0.0.1:${PROXY_PORT}"
echo "Reload with: exec bash"
