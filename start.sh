#!/usr/bin/env bash
# Simple start script: activates venv and runs gunicorn with single worker
set -e
ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
VENV_DIR="$ROOT_DIR/.venv"

if [ -d "$VENV_DIR" ]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
fi

# Allow overriding host/port and model path via env vars
: ${HOST:=0.0.0.0}
: ${PORT:=8000}
: ${WORKERS:=1}
: ${MODEL_PATH:=}

if [ -n "$MODEL_PATH" ]; then
  export MODEL_PATH
fi

echo "Starting gunicorn on $HOST:$PORT with $WORKERS worker(s)"
exec gunicorn -w "$WORKERS" -b "$HOST:$PORT" src.app:app
