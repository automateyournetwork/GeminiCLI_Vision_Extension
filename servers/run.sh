#!/usr/bin/env bash
set -euo pipefail

EXT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
SRV="$EXT/servers"
VENV="$SRV/visionmcp"
PY="${PYTHON_BIN:-python3}"

# 1) Create venv if missing
if [ ! -x "$VENV/bin/python3" ]; then
  echo "[vision] creating venv at $VENV" >&2
  "$PY" -m venv "$VENV"
  "$VENV/bin/pip" install -U pip wheel setuptools
fi

# 2) Ensure deps (idempotent)
if [ -f "$SRV/requirements.txt" ]; then
  "$VENV/bin/pip" install -r "$SRV/requirements.txt"
fi

# 3) Exec the MCP
exec "$VENV/bin/python3" "$SRV/vision_mcp.py"
