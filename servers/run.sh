#!/usr/bin/env bash
set -euo pipefail

EXT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
SERVERS_DIR="$EXT_DIR/servers"
VENV="$SERVERS_DIR/visionmcp"
PYTHON_BIN="${PYTHON_BIN:-python3}"   # allow override via env if needed

# 1) make venv if missing
if [ ! -x "$VENV/bin/python3" ]; then
  echo "[vision] creating venv at $VENV" >&2
  "$PYTHON_BIN" -m venv "$VENV"
  "$VENV/bin/pip" install -U pip wheel setuptools
fi

# 2) install deps if missing/outdated (idempotent)
if [ -f "$SERVERS_DIR/requirements.txt" ]; then
  "$VENV/bin/pip" install -r "$SERVERS_DIR/requirements.txt"
fi

# 3) exec server
exec "$VENV/bin/python3" "$SERVERS_DIR/vision_mcp.py"
