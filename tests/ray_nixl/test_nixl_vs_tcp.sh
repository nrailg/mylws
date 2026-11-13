#!/usr/bin/env bash
set -Eeuo pipefail

MYLWS_DIR="${MY_DFS:?set MY_DFS from the current Gemini session}/work/mylws"
source "${MYLWS_VENV:?set the verified venv path}/bin/activate"
cd "$MYLWS_DIR"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2
export RAY_USAGE_STATS_ENABLED=0
python -u tests/ray_nixl/test_nixl_vs_tcp.py "$@"
