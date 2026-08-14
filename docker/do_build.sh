#!/usr/bin/env bash

# Stop at first error
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_IMAGE_TAG="isles26-nnunet"

# Stage the postprocessing module into the build context. It is NOT duplicated in
# the repo: the container must apply exactly the floor the CV was scored with, so
# src/isles26/postprocessing.py stays the single source and is copied at build
# time. Keeping the context at docker/ also avoids shipping dataset/ to the daemon.
cp "$SCRIPT_DIR/../src/isles26/postprocessing.py" "$SCRIPT_DIR/postprocessing.py"

docker build \
  --platform=linux/amd64 \
  --tag "$DOCKER_IMAGE_TAG"  \
  ${DOCKER_QUIET_BUILD:+--quiet} \
  "$SCRIPT_DIR" 2>&1
