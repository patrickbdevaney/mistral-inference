# mistral-inference

Serve **Mistral-Small-4-119B-2603-NVFP4** with **Eagle-3 speculative decoding** on a
**Jetson AGX Thor (SM110)** using the NVIDIA Jetson vLLM container.

Model: 119B total / 6.5B active (MoE, 128 experts, 4 active), **deepseek_v2-style MLA**,
Pixtral multimodal wrapper (served text-only). NVFP4 weights, ~66 GB.

> Weights are **not** in this repo (`.gitignore`d). `download-models.sh` pulls them from
> the Hugging Face Hub into `./models/`.

## Quickstart

```bash
./download-models.sh     # NVFP4 (~66 GB) + Eagle draft (~0.4 GB) -> ./models/
./serve.sh               # production server on :8002 (TRITON_MLA + Eagle + chat patch)
./benchmark.sh           # optional: measure base + Eagle decode tok/s

# modes
./serve.sh no-eagle      # no spec decoding, full 256k context
```

Requires: Jetson AGX Thor, Docker + `nvidia` runtime, `hf` CLI (`pip install -U huggingface_hub`),
and the image `ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor`.

## Measured results (Thor, SM110, `nvpmodel -m 1`)

| Config | tok/s | |
|---|---|---|
| Base decode (TRITON_MLA, no Eagle) | **~17.5** | `/v1/completions`, 512-tok runs |
| Eagle-3 decode (TRITON_MLA) | **~33.7** (up to ~44 in chat) | **1.93×** over base |

Backend: `TRITON_MLA` + `NvFp4LinearBackend.FLASHINFER_CUTLASS` GEMM + CUTLASS MoE
(**no Marlin fallback**). `gpu_memory_utilization=0.72` → ~96 GB used / ~26 GB free.

## Hard-won gotchas (this image, vLLM 0.19.0)

These cost real time during bring-up; the scripts already handle them.

1. **Backend must be `TRITON_MLA`, not FLASHINFER.** The model is MLA; FLASHINFER has no
   MLA kernels and crashes at load:
   `Selected backend FLASHINFER ... ['head_size not supported','MLA not supported']`.
   `TRITON_MLA` (the model-card recommendation) is the MLA-capable backend.

2. **Chat endpoint 400s without the patch.** vLLM 0.19.0 forwards `reasoning_effort` to
   `transformers 4.57.3` `MistralCommonTokenizer.apply_chat_template`, which rejects all
   extra kwargs → **every `/v1/chat/completions` request fails**. `patches/mistral.py` is a
   copy of the in-container file that only forwards `reasoning_effort` when transformers
   actually accepts it (introspected), otherwise drops it. It is mounted read-only over the
   container file:
   ```
   -v patches/mistral.py:/opt/venv/lib/python3.12/site-packages/vllm/tokenizers/mistral.py:ro
   ```
   Non-destructive, reversible, self-adapting. `/v1/completions` is unaffected either way.
   **Limitation:** `reasoning_effort="high"` is silently dropped (instruct mode only) until
   the container's `transformers` + `mistral_common` are upgraded. Chat + tool-calling work.

3. **Use `hf download`, not `huggingface-cli download`** — the latter is a dead no-op in
   `huggingface_hub >= 1.x`.

4. **Mistral consolidated format.** The repo ships `params.json` + `consolidated*.safetensors`
   + `tekken.json` (no `config.json`); vLLM auto-detects it with `tokenizer_mode: mistral`.

5. **Page-cache memory gate.** After a container stops, the 66 GB of weights stay in Linux
   page cache, and the next container's startup memory check fails
   (`Free memory on device cuda:0 ... less than gpu_memory_utilization`). Always
   `sudo sync && sudo sysctl -w vm.drop_caches=3` and wait for headroom before relaunch
   (the scripts loop until >95 GB is free).

## Layout

```
download-models.sh   # hf download -> ./models/  (weights gitignored)
serve.sh             # production vLLM server (:8002), default = Eagle-3
benchmark.sh         # base + Eagle decode benchmark (:8003)
patches/mistral.py   # chat-fix patch, mounted into the container
config/              # auto-generated runtime YAML (gitignored)
models/              # downloaded weights (gitignored)
```

## API examples

```bash
# chat
curl -s localhost:8002/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"/model","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'

# tool calling (Mistral parser)
curl -s localhost:8002/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"/model","messages":[{"role":"user","content":"Weather in Paris? Use the tool."}],
       "tools":[{"type":"function","function":{"name":"get_weather",
       "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]}'

# raw completion
curl -s localhost:8002/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"/model","prompt":"def fib(n):","max_tokens":128}'
```
