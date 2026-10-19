#!/usr/bin/env bash
set -euo pipefail
mkdir -p /home/xinyic/SER/.cache/{home,pip,torch,huggingface,numba,matplotlib,pycache}
exec docker run --gpus all --rm --shm-size=8g \
  --user "$(id -u):$(id -g)" \
  -v /home/xinyic/SER:/workspace -w /workspace \
  ser-vesper python -c '
import importlib, os, torch
from pathlib import Path
for name in ("numpy", "scipy", "pandas", "sklearn", "librosa", "numba", "soundfile", "matplotlib", "einops", "yacs", "tqdm", "tensorboard"):
    importlib.import_module(name)
from torch.utils.tensorboard import SummaryWriter
print("torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
assert torch.cuda.is_available()
print("GPU:", torch.cuda.get_device_name(0))
assert torch.ones(8, device="cuda").sum().item() == 8
assert os.getcwd() == "/workspace"
p = Path("/workspace/.cache/verify-owner.txt")
p.write_text("workspace mount verified\n")
print("workspace:", os.getcwd(), "UID/GID:", os.getuid(), os.getgid())
print("Dependencies and GPU checks passed")
'
