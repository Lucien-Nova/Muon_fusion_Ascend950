#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}"

cd "${ROOT}"
if "${PYTHON_BIN}" -c "import pip" 2>/dev/null; then
  "${PYTHON_BIN}" -m pip wheel --no-build-isolation --no-deps --wheel-dir dist .
else
  "${PYTHON_BIN}" setup.py bdist_wheel --dist-dir dist
fi
