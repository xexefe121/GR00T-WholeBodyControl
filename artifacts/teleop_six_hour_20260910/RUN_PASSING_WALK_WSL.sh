#!/usr/bin/env bash
set -euo pipefail
if ! mountpoint -q /mnt/e; then
  mkdir -p /mnt/e
  mount -t drvfs E: /mnt/e
fi
exec "$@"
