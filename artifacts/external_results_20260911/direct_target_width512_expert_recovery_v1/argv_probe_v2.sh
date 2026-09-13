#!/usr/bin/env bash
set -euo pipefail
test "$#" -eq 2
test "$1" = FIXED_ARG_ONE
test "$2" = FIXED_ARG_TWO
printf '{"cwd":"%s","PYTHONPATH":"%s","OMP_NUM_THREADS":"%s","OPENBLAS_NUM_THREADS":"%s","MKL_NUM_THREADS":"%s","PYTHONDONTWRITEBYTECODE":"%s","argv":["%s","%s"],"linux_pid":%s,"probe_only":true}\n' "$PWD" "$PYTHONPATH" "$OMP_NUM_THREADS" "$OPENBLAS_NUM_THREADS" "$MKL_NUM_THREADS" "$PYTHONDONTWRITEBYTECODE" "$1" "$2" "$$"
