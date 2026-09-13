#!/usr/bin/env bash
# Keep LF line endings: WSL Bash must not receive a carriage return in pipefail.
set -euo pipefail

service_dir=/opt/apps/roboticsservice
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${service_dir}:${service_dir}/lib:${service_dir}/SDK/x64"
export QT_PLUGIN_PATH="${service_dir}/plugins:${QT_PLUGIN_PATH:-}"
export QT_QML_PATH="${service_dir}/qml:${QT_QML_PATH:-}"

cd "${service_dir}"
exec ./RoboticsServiceProcess
