#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python_bin=".venv-emotion2vec/bin/python"
if [[ ! -x "$python_bin" ]]; then
  printf '%s\n' 'Missing .venv-emotion2vec. Create the environment as described in README.md.' >&2
  exit 1
fi
if ! "$python_bin" -c 'import fastapi, uvicorn, python_multipart' 2>/dev/null; then
  printf '%s\n' 'Install UI dependencies: .venv-emotion2vec/bin/python -m pip install -r requirements-ui.txt' >&2
  exit 1
fi
exec "$python_bin" -m uvicorn app:app --host 127.0.0.1 --port 8000 "$@"
