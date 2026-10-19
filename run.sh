#!/usr/bin/env bash
set -euo pipefail
mkdir -p /home/xinyic/SER/.cache/{home,pip,torch,huggingface,numba,matplotlib,pycache}
exec docker run --gpus all -it --rm \
  --name ser-workspace \
  --shm-size=8g \
  -e HTTP_PROXY -e HTTPS_PROXY -e http_proxy -e https_proxy \
  -e NO_PROXY -e no_proxy \
  --user "$(id -u):$(id -g)" \
  -v /home/xinyic/SER:/workspace \
  -w /workspace \
  ser-vesper bash
