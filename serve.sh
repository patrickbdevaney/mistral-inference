#!/bin/bash
# serve.sh
# Production vLLM server for Mistral-Small-4-119B-2603-NVFP4 on Jetson AGX Thor (SM110).
#
# Validated stack (2026-06-19, Thor 117 GB usable):
#   * Backend  : TRITON_MLA  (the model is deepseek_v2-style MLA — FLASHINFER has NO
#                MLA kernels and crashes at load. TRITON_MLA = model-card recommendation,
#                native NVFP4 via NvFp4LinearBackend.FLASHINFER_CUTLASS + CUTLASS MoE.)
#   * Spec     : Eagle-3 draft head, num_speculative_tokens=3 (default mode)
#   * Chat fix : patches/mistral.py mounted RO over the in-container vllm tokenizer file
#                (this image's vllm 0.19.0 forwards reasoning_effort to transformers
#                4.57.3 MistralCommonTokenizer, which rejects it -> every chat request 400s).
#
# Measured: base ~17.5 tok/s, Eagle-3 ~33.7 tok/s (1.93x).
#
# Modes:
#   ./serve.sh            # default: Eagle-3, 64k ctx
#   ./serve.sh no-eagle   # no spec decoding, full 256k ctx
#
# Reasoning note: reasoning_effort="high" is silently dropped (instruct mode) until the
# container's transformers/mistral_common are upgraded. Chat + tool-calling work.

set -euo pipefail
MODE="${1:-default}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_DIR="$REPO_ROOT/models/Mistral-Small-4-NVFP4"
EAGLE_DIR="$REPO_ROOT/models/Mistral-Small-4-eagle"
CONFIG_DIR="$REPO_ROOT/config"
PATCH_FILE="$REPO_ROOT/patches/mistral.py"
PATCH_TARGET="/opt/venv/lib/python3.12/site-packages/vllm/tokenizers/mistral.py"
IMAGE="ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor"
BACKEND="TRITON_MLA"
PORT=8002
CACHE_ROOT="$HOME/thor-vllm-cache/mistral-small4"
mkdir -p "$CONFIG_DIR" "$CACHE_ROOT"

[ ! -d "$MODEL_DIR" ] && { echo "ERROR: model missing at $MODEL_DIR — run ./download-models.sh first"; exit 1; }

case "$MODE" in
default)
    MAX_LEN=65536
    SPEC_YAML="speculative_config:
  model: /eagle
  method: eagle
  num_speculative_tokens: 3
  max_model_len: 65536"
    echo "Mode: default | Eagle-3 | 64k ctx | TRITON_MLA | NVFP4" ;;
no-eagle)
    MAX_LEN=262144
    SPEC_YAML="# Eagle disabled — full 256k context."
    echo "Mode: no-eagle | 256k ctx | TRITON_MLA | NVFP4" ;;
*)
    echo "Usage: $0 [default|no-eagle]"; exit 1 ;;
esac

# ── Write runtime config ───────────────────────────────────────────────────────
CONFIG="$CONFIG_DIR/mistral-small4-production.yaml"
cat > "$CONFIG" << YAML_EOF
host: 0.0.0.0
port: ${PORT}
gpu_memory_utilization: 0.72
max_model_len: ${MAX_LEN}
max_num_seqs: 1
max_num_batched_tokens: 8192
enable_chunked_prefill: true
quantization: compressed-tensors
kv_cache_dtype: auto
tokenizer_mode: mistral
reasoning_parser: mistral
enable_auto_tool_choice: true
tool_call_parser: mistral
trust_remote_code: true
enable_prefix_caching: false
${SPEC_YAML}
compilation_config:
  cudagraph_mode: PIECEWISE
  cudagraph_capture_sizes:
    - 4
    - 8
    - 12
YAML_EOF
echo "=== config: $CONFIG ==="; cat "$CONFIG"; echo "==============="

# ── Thor perf + memory mitigations ─────────────────────────────────────────────
sudo nvpmodel -m 1 2>/dev/null && echo "nvpmodel -m 1" || echo "WARN: nvpmodel failed (non-fatal)"
sudo jetson_clocks 2>/dev/null && echo "jetson_clocks locked" || echo "WARN: jetson_clocks failed (non-fatal)"
# Thor page-cache gotcha: weights stay cached after a stop and the next container's
# startup memory check fails. Drop caches and wait for headroom before launching.
sudo sync && sudo sysctl -w vm.drop_caches=3 >/dev/null 2>&1 || true
sudo fuser -k ${PORT}/tcp 2>/dev/null || true
until [ "$(free -g | awk '/^Mem:/{print $7}')" -gt 95 ]; do
    echo "  waiting for page cache to drain ($(free -g | awk '/^Mem:/{print $7}') GiB free)..."
    sudo sync; sudo sysctl -w vm.drop_caches=3 >/dev/null 2>&1; sleep 5
done

# ── Mounts ─────────────────────────────────────────────────────────────────────
EAGLE_MOUNT=""
[ -d "$EAGLE_DIR" ] && [ "$MODE" != "no-eagle" ] && EAGLE_MOUNT="-v ${EAGLE_DIR}:/eagle:ro"
PATCH_MOUNT=""
if [ -f "$PATCH_FILE" ]; then
    PATCH_MOUNT="-v ${PATCH_FILE}:${PATCH_TARGET}:ro"
    echo "chat-fix patch mounted"
else
    echo "!! WARNING: $PATCH_FILE missing — /v1/chat/completions will 400 on reasoning_effort"
fi

docker pull "${IMAGE}" || docker image inspect "${IMAGE}" &>/dev/null || { echo "ERROR: no image."; exit 1; }

CONTAINER_NAME="vllm-mistral-small4-$(date +%s)"
echo ""
echo "=== Launch — container: $CONTAINER_NAME on :$PORT ==="
echo "  Cold compile + graph capture can take several minutes. Silence is normal."
echo "  WANT in logs: 'TRITON_MLA backend' + 'FLASHINFER_CUTLASS' (NvFp4). FAIL: 'marlin_utils_fp4'."
echo ""

docker run --rm -it \
    --name "$CONTAINER_NAME" \
    --runtime nvidia --gpus all \
    --ipc=host --network host \
    --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=16g \
    -e VLLM_USE_FLASHINFER_MOE_FP4=1 \
    -e VLLM_FLASHINFER_MOE_BACKEND=latency \
    -e VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass \
    -e VLLM_USE_FUSED_MOE_GROUPED_TOPK=1 \
    -e VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=524288000 \
    -e VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE=163840 \
    -e VLLM_CPU_OMP_THREADS_BIND=auto \
    -e USE_FASTSAFETENSOR=true \
    -e VLLM_ENABLE_CUDAGRAPH_GC=0 \
    -e PYTORCH_ALLOC_CONF=expandable_segments:True \
    -e HF_HUB_DISABLE_XET=1 \
    -e LD_PRELOAD=/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1 \
    -v "${MODEL_DIR}:/model:ro" \
    ${EAGLE_MOUNT} \
    ${PATCH_MOUNT} \
    -v "${CACHE_ROOT}:/root/.cache/vllm" \
    -v "${CONFIG_DIR}:/root/config:ro" \
    "${IMAGE}" \
    vllm serve /model \
        --config /root/config/mistral-small4-production.yaml \
        --attention-backend "${BACKEND}" \
        --language-model-only
