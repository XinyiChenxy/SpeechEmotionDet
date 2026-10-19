#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec docker build -t ser-vesper \
  --build-arg USER_UID="$(id -u)" --build-arg USER_GID="$(id -g)" \
  --build-arg HTTP_PROXY --build-arg HTTPS_PROXY \
  --build-arg http_proxy --build-arg https_proxy \
  --build-arg NO_PROXY --build-arg no_proxy "$@" .
