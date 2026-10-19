#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
config="${1:-configs/cremad.yaml}"
checkpoint="${2:-checkpoints/Vesper-4.pt}"
# Fail visibly before starting a process when the required weights are missing.
python -m ser.vesper check-checkpoint --config "$config" --checkpoint "$checkpoint"
python -m ser.vesper prepare --config "$config"
mkdir -p outputs
log="outputs/train_$(date -u +%Y%m%dT%H%M%SZ).log"
nohup python -u -m ser.vesper train --config "$config" --checkpoint "$checkpoint" >"$log" 2>&1 &
pid=$!
printf '%s\n' "$pid" > outputs/train.pid
printf 'PID: %s\nLog: %s\n' "$pid" "$log"
