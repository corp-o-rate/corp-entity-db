#!/bin/bash
set -e

if [ -z "$1" ]; then
  echo "Usage: ./build.sh <version>"
  echo "Example: ./build.sh 0.2.1"
  exit 1
fi

DOCKER_BUILDKIT=1 docker build --build-arg HF_TOKEN=$HF_TOKEN --platform linux/amd64 -t corp-entity-db-runpod .
docker tag corp-entity-db-runpod neilellis/corp-entity-db-runpod:v$1
docker push neilellis/corp-entity-db-runpod:v$1