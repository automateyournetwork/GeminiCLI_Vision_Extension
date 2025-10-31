#!/usr/bin/env bash
set -euo pipefail

EXT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
SERVERS_DIR="$EXT_DIR/servers"
VENV="$SERVERS_DIR/visionmcp"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"

# 1) Create venv if missing
if [ ! -x "$VENV/bin/python3" ]; then
  echo "[vision] creating venv at $VENV" >&2
  "$PYTHON_BIN" -m venv "$VENV" 1>&2
  "$VENV/bin/pip" install -U pip wheel setuptools --disable-pip-version-check -q 1>&2
fi

# 2) Ensure deps (idempotent). All output -> STDERR.
if [ -f "$SERVERS_DIR/requirements.txt" ]; then
  "$VENV/bin/pip" install -r "$SERVERS_DIR/requirements.txt" \
    --disable-pip-version-check --no-input -q 1>&2
fi

# 3) Exec the MCP server (this must be the ONLY thing that writes to STDOUT)
exec "$VENV/bin/python3" "$SERVERS_DIR/vision_mcp.py"
