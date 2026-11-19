#!/bin/sh
set -eu
cd "$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)"
test -x .venv/bin/python || { echo "Ambiente .venv ausente; restaure o ambiente local validado."; exit 1; }
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
export HF_HUB_OFFLINE=1
exec .venv/bin/python -m sonar_vision.desktop.app "$@"
