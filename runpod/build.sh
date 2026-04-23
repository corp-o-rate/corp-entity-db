#!/bin/bash
set -e

if [ -z "$1" ]; then
  echo "Usage: ./build.sh <version>"
  echo "Example: ./build.sh 0.2.1"
  exit 1
fi

if [ -z "$HF_TOKEN" ]; then
  echo "ERROR: HF_TOKEN env var must be set (needed to download the gated embedding model)."
  echo "Get a token at https://huggingface.co/settings/tokens"
  exit 1
fi

# HF_TOKEN is passed as a BuildKit secret — NOT a build-arg — so it is never baked
# into any image layer. See docker/dockerfile:1.4 --mount=type=secret in the Dockerfile.
#
# --provenance=false --sbom=false disable BuildKit's attestation manifest. The default
# output is an OCI image index containing an "unknown/unknown" attestation manifest;
# RunPod's image puller has been seen to hang on "pending" when pulling that format.
# Plain single-manifest images pull reliably.
DOCKER_BUILDKIT=1 docker buildx build \
  --secret id=hf_token,env=HF_TOKEN \
  --platform linux/amd64 \
  --provenance=false \
  --sbom=false \
  --load \
  -t corp-entity-db-runpod .
docker tag corp-entity-db-runpod neilellis/corp-entity-db-runpod:v$1
docker push neilellis/corp-entity-db-runpod:v$1