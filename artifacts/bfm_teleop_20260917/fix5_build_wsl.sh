#!/usr/bin/env bash
set -euo pipefail
cd /root/deploy_build
cmake --build . --target g1_true23_bfm_lowcmd_loop -j2