#!/bin/bash
# benchmark.sh
# Benchmark decode throughput for Mistral-Small-4-119B-2603-NVFP4 on Jetson AGX Thor:
#   1. Base decode  (TRITON_MLA, no speculative decoding)
#   2. Eagle decode (TRITON_MLA + Eagle-3, num_speculative_tokens=3)
# Measures via /v1/completions (chat needs the patch and avoids chat-template variance).
# Run ./download-models.sh first.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_DIR="$REPO_ROOT/models/Mistral-Small-4-NVFP4"
EAGLE_DIR="$REPO_ROOT/models/Mistral-Small-4-eagle"
CONFIG_DIR="$REPO_ROOT/config"
RESULTS_DIR="$REPO_ROOT/results"
PATCH_FILE="$REPO_ROOT/patches/mistral.py"
PATCH_TARGET="/opt/venv/lib/python3.12/site-packages/vllm/tokenizers/mistral.py"
IMAGE="ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor"
BACKEND="TRITON_MLA"
PORT=8003
CACHE_ROOT="$HOME/thor-vllm-cache/mistral-small4-bench"
mkdir -p "$CONFIG_DIR" "$RESULTS_DIR" "$CACHE_ROOT"

BENCH_PROMPT="Write a complete Python implementation of a red-black tree with insert, delete, and search operations. Include docstrings and type hints.\n"
BENCH_MAX_TOKENS=512
BENCH_RUNS=3

[ ! -d "$MODEL_DIR" ] && { echo "ERROR: model missing — run ./download-models.sh first"; exit 1; }
PATCH_MOUNT=""; [ -f "$PATCH_FILE" ] && PATCH_MOUNT="-v ${PATCH_FILE}:${PATCH_TARGET}:ro"

sudo nvpmodel -m 1 2>/dev/null && echo "nvpmodel -m 1" || true
sudo jetson_clocks 2>/dev/null || true
docker pull "${IMAGE}" || docker image inspect "${IMAGE}" &>/dev/null || { echo "ERROR: no image."; exit 1; }

write_config() {  # cfg_path  spec_block
    cat > "$1" << YAML_EOF
host: 0.0.0.0
port: ${PORT}
gpu_memory_utilization: 0.72
max_model_len: 65536
max_num_seqs: 1
max_num_batched_tokens: 8192
enable_chunked_prefill: true
quantization: compressed-tensors
kv_cache_dtype: auto
tokenizer_mode: mistral
trust_remote_code: true
enable_prefix_caching: false
${2}
compilation_config:
  cudagraph_mode: PIECEWISE
  cudagraph_capture_sizes:
    - 4
    - 8
    - 12
YAML_EOF
}

run_benchmark() {  # label  config_path  eagle_mount  result_file
    local LABEL="$1" CONFIG_PATH="$2" EAGLE_MOUNT="${3:-}" RESULT_FILE="$4"
    echo ""; echo "=== Benchmark: $LABEL ($BACKEND) ==="
    sudo fuser -k ${PORT}/tcp 2>/dev/null || true
    sudo sync && sudo sysctl -w vm.drop_caches=3 >/dev/null 2>&1 || true
    until [ "$(free -g | awk '/^Mem:/{print $7}')" -gt 95 ]; do
        echo "  draining page cache..."; sudo sync; sudo sysctl -w vm.drop_caches=3 >/dev/null 2>&1; sleep 5
    done

    local CN="vllm-bench-$(date +%s)"
    docker run --rm -d --name "$CN" \
        --runtime nvidia --gpus all --ipc=host --network host \
        --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=16g \
        -e VLLM_USE_FLASHINFER_MOE_FP4=1 -e VLLM_FLASHINFER_MOE_BACKEND=latency \
        -e VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass -e VLLM_USE_FUSED_MOE_GROUPED_TOPK=1 \
        -e VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=524288000 -e VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE=163840 \
        -e VLLM_CPU_OMP_THREADS_BIND=auto -e USE_FASTSAFETENSOR=true -e VLLM_ENABLE_CUDAGRAPH_GC=0 \
        -e PYTORCH_ALLOC_CONF=expandable_segments:True -e HF_HUB_DISABLE_XET=1 \
        -e LD_PRELOAD=/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1 \
        -v "${MODEL_DIR}:/model:ro" ${EAGLE_MOUNT} ${PATCH_MOUNT} \
        -v "${CACHE_ROOT}:/root/.cache/vllm" -v "${CONFIG_DIR}:/root/config:ro" \
        "${IMAGE}" \
        vllm serve /model --config "/root/config/$(basename "$CONFIG_PATH")" \
            --attention-backend "$BACKEND" --language-model-only

    echo "  waiting for ready (up to 20 min)..."; local WAIT=0
    until curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; do
        if [ "$(docker inspect -f '{{.State.Status}}' "$CN" 2>/dev/null)" != "running" ]; then
            echo "  CONTAINER DIED:"; docker logs "$CN" 2>&1 | grep -iE "error|valueerror|traceback" | tail -15; return 1
        fi
        sleep 10; WAIT=$((WAIT+10)); [ $WAIT -ge 1200 ] && { echo "  TIMEOUT"; docker stop "$CN" 2>/dev/null; return 1; }
        [ $((WAIT % 60)) -eq 0 ] && echo "  ...${WAIT}s..."
    done
    echo "  ready after ${WAIT}s."
    docker logs "$CN" 2>&1 | grep -q "marlin_utils_fp4" && DEG="[MARLIN] " || DEG=""

    curl -sf "http://localhost:${PORT}/v1/completions" -H "Content-Type: application/json" \
        -d '{"model":"/model","prompt":"hello","max_tokens":16}' >/dev/null 2>&1; sleep 2
    local TT=0 TS=0
    for i in $(seq 1 "$BENCH_RUNS"); do
        local S=$(date +%s%3N)
        local R=$(curl -s "http://localhost:${PORT}/v1/completions" -H "Content-Type: application/json" \
            -d "{\"model\":\"/model\",\"prompt\":\"$BENCH_PROMPT\",\"max_tokens\":$BENCH_MAX_TOKENS,\"temperature\":0.0}")
        local E=$(date +%s%3N); local SEC=$(awk "BEGIN{printf \"%.3f\",($E-$S)/1000}")
        local CT=$(echo "$R" | python3 -c "import json,sys;print(json.load(sys.stdin)['usage']['completion_tokens'])" 2>/dev/null || echo 0)
        if [ "$CT" -gt 0 ]; then
            awk "BEGIN{printf \"    run $i: %d tok / %ss = %.1f tok/s\n\",$CT,$SEC,$CT/$SEC}"
            TT=$((TT+CT)); TS=$(awk "BEGIN{printf \"%.3f\",$TS+$SEC}")
        else echo "    run $i FAILED: $(echo "$R"|head -c 160)"; fi
        sleep 2
    done
    local AVG=$(awk "BEGIN{if($TS>0)printf \"%.1f\",$TT/$TS; else print \"N/A\"}")
    echo "  RESULT $LABEL: ${DEG}${AVG} tok/s"
    { echo "label=$LABEL"; echo "backend=$BACKEND"; echo "avg_toks_per_s=${DEG}${AVG}"
      echo "runs=$BENCH_RUNS"; echo "bench_max_tokens=$BENCH_MAX_TOKENS"; echo "endpoint=/v1/completions"
      echo "timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; } > "$RESULT_FILE"
    docker stop "$CN" 2>/dev/null || true; sleep 5
    sudo sync && sudo sysctl -w vm.drop_caches=3 >/dev/null 2>&1 || true; sleep 3
}

echo "$BACKEND" > "$RESULTS_DIR/confirmed-backend.txt"

echo ""; echo "=== Base decode (no Eagle) ==="
BASE_CFG="$CONFIG_DIR/base.yaml"; write_config "$BASE_CFG" "# no speculative decoding"
run_benchmark "base-decode-TRITON_MLA" "$BASE_CFG" "" "$RESULTS_DIR/base-decode.txt" || echo "base bench failed"

echo ""; echo "=== Eagle decode ==="
EAGLE_CFG="$CONFIG_DIR/eagle.yaml"
write_config "$EAGLE_CFG" "speculative_config:
  model: /eagle
  method: eagle
  num_speculative_tokens: 3
  max_model_len: 65536"
run_benchmark "eagle-decode-TRITON_MLA" "$EAGLE_CFG" "-v ${EAGLE_DIR}:/eagle:ro" "$RESULTS_DIR/eagle-decode.txt" || echo "eagle bench failed"

echo ""
echo "============== RESULTS =============="
[ -f "$RESULTS_DIR/base-decode.txt" ]  && echo "  Base  : $(grep avg_toks_per_s "$RESULTS_DIR/base-decode.txt"  | cut -d= -f2) tok/s"
[ -f "$RESULTS_DIR/eagle-decode.txt" ] && echo "  Eagle : $(grep avg_toks_per_s "$RESULTS_DIR/eagle-decode.txt" | cut -d= -f2) tok/s"
echo "===================================="
