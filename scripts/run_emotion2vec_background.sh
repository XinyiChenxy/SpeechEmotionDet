#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
config="${1:-configs/emotion2vec.yaml}"
python_bin=".venv-emotion2vec/bin/python"
test -x "$python_bin"
"$python_bin" -m ser check-checkpoint --config "$config" > /dev/null
mkdir -p outputs
log="outputs/emotion2vec_run_$(date -u +%Y%m%dT%H%M%SZ).log"
nohup "$python_bin" -u -m ser run --config "$config" > "$log" 2>&1 &
pid=$!
printf '%s\n' "$pid" > outputs/emotion2vec_run.pid
printf 'PID: %s\nLog: %s\n' "$pid" "$log"
