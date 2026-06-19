#!/bin/bash
# build-image.sh — build the derived vLLM image with reasoning_effort support baked in.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TAG="${1:-mistral-small4-thor:reasoning}"
echo "Building $TAG from $REPO_ROOT/Dockerfile ..."
docker build -t "$TAG" "$REPO_ROOT"
echo ""
echo "Done: $TAG"
echo "Use it with:  REASONING_IMAGE=$TAG ./serve.sh"
