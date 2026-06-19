#!/bin/bash
# download-models.sh
# Download Mistral-Small-4-119B-2603 NVFP4 target model (~66 GB) and the Eagle-3
# draft head (~0.4 GB) into ./models/. Idempotent and resumable.
#
# NOTE: `huggingface-cli download` is dead in huggingface_hub >= 1.x — it prints a
# deprecation notice and does nothing. Use `hf download` (resume + local-dir copy
# are now the default behaviour; the old --resume-download / --local-dir-use-symlinks
# flags were removed).

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_DIR="$REPO_ROOT/models/Mistral-Small-4-NVFP4"
EAGLE_DIR="$REPO_ROOT/models/Mistral-Small-4-eagle"

NVFP4_REPO="mistralai/Mistral-Small-4-119B-2603-NVFP4"
EAGLE_REPO="mistralai/Mistral-Small-4-119B-2603-eagle"

command -v hf >/dev/null 2>&1 || pip install --quiet -U huggingface_hub

if [ ! -d "$MODEL_DIR" ] || [ -z "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]; then
    echo "Downloading $NVFP4_REPO (~66 GB) -> $MODEL_DIR"
    hf download "$NVFP4_REPO" --local-dir "$MODEL_DIR"
else
    echo "NVFP4 model already present: $MODEL_DIR"
fi

if [ ! -d "$EAGLE_DIR" ] || [ -z "$(ls -A "$EAGLE_DIR" 2>/dev/null)" ]; then
    echo "Downloading $EAGLE_REPO (~0.4 GB) -> $EAGLE_DIR"
    hf download "$EAGLE_REPO" --local-dir "$EAGLE_DIR"
else
    echo "Eagle model already present: $EAGLE_DIR"
fi

echo ""
echo "Done. Verify shards (expect 13):"
echo "  NVFP4 shards: $(ls "$MODEL_DIR"/consolidated-*.safetensors 2>/dev/null | wc -l)/13"
echo "  Eagle draft : $([ -f "$EAGLE_DIR/consolidated.safetensors" ] && echo present || echo MISSING)"
