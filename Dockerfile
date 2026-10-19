FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 git git-lfs ca-certificates build-essential \
    && git lfs install --system \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-docker.txt /tmp/requirements-docker.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements-docker.txt && python -m pip check
ARG USER_UID=1014
ARG USER_GID=1015
RUN groupadd --gid ${USER_GID} xinyic \
    && useradd --uid ${USER_UID} --gid ${USER_GID} --home-dir /workspace/.cache/home --no-create-home --shell /bin/bash xinyic
USER xinyic
ENV HOME=/workspace/.cache/home \
    XDG_CACHE_HOME=/workspace/.cache \
    PIP_CACHE_DIR=/workspace/.cache/pip \
    TORCH_HOME=/workspace/.cache/torch \
    HF_HOME=/workspace/.cache/huggingface \
    NUMBA_CACHE_DIR=/workspace/.cache/numba \
    MPLCONFIGDIR=/workspace/.cache/matplotlib \
    PYTHONPYCACHEPREFIX=/workspace/.cache/pycache
WORKDIR /workspace
CMD ["bash"]
