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
./build-image.sh         # derived image: chat fix + reasoning_effort (recommended)
./serve.sh               # production server on :8002 (TRITON_MLA + Eagle)
./benchmark.sh           # optional: measure base + Eagle decode tok/s

# modes
./serve.sh no-eagle      # no spec decoding, full 256k context
```

`serve.sh`/`benchmark.sh` use the derived image `mistral-small4-thor:reasoning` if it has
been built (patches baked in); otherwise they fall back to the base image and mount the
patches read-only. Override with `REASONING_IMAGE=<tag> ./serve.sh`.

Requires: Jetson AGX Thor, Docker + `nvidia` runtime, `hf` CLI (`pip install -U huggingface_hub`),
and the image `ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor`.

## Measured results (Thor, SM110, `nvpmodel -m 1`)

| Config | tok/s | |
|---|---|---|
| Base decode (TRITON_MLA, no Eagle) | **~17.5** | `/v1/completions`, 512-tok runs |
| Eagle-3 decode (TRITON_MLA) | **~33.7** (up to ~44 in chat) | **1.93×** over base |

Backend: `TRITON_MLA` + `NvFp4LinearBackend.FLASHINFER_CUTLASS` GEMM + CUTLASS MoE
(**no Marlin fallback**). `gpu_memory_utilization=0.72` → ~96 GB used / ~26 GB free.

## Reasoning vs instruct (the wall-time story)

Decode speed is identical between modes — what differs is **how many tokens** get
emitted, which drives wall-clock time-to-answer. Mistral Small 4 answers directly in
instruct mode (`reasoning_effort="none"`, the default); reasoning is opt-in per request.

| Question | Mode | Tokens | Wall | Result |
|---|---|---|---|---|
| "17 × 23?" | instruct | 12 | 0.4s | 391 |
| | reasoning | 181 | 4.7s | 391 — **15× tokens, 10.6× wall** |
| "Is 2027 prime?" | instruct | 632 | 14.3s | yes |
| | reasoning | 1394 | 31.6s | yes — **2.2× tokens, 2.2× wall** |

Same ~44 tok/s decode in both. So at equal decode speed, instruct mode beats
always-reasoning models on time-to-answer, and you pay the reasoning tax only when you
ask for it. See "Reasoning mode status" below for the current limitation.

## Hard-won gotchas (this image, vLLM 0.19.0)

These cost real time during bring-up; the scripts already handle them.

1. **Backend must be `TRITON_MLA`, not FLASHINFER.** The model is MLA; FLASHINFER has no
   MLA kernels and crashes at load:
   `Selected backend FLASHINFER ... ['head_size not supported','MLA not supported']`.
   `TRITON_MLA` (the model-card recommendation) is the MLA-capable backend.

2. **Chat endpoint 400s without the patches.** vLLM 0.19.0 forwards `reasoning_effort` to
   `transformers 4.57.3` `MistralCommonTokenizer.apply_chat_template`, which rejects all
   extra kwargs → **every `/v1/chat/completions` request fails**. Two patches fix this
   (`Dockerfile` bakes both in; `serve.sh` mounts them if you skip the build):
   - `patches/mistral.py` — vLLM tokenizer wrapper: forwards `reasoning_effort` to
     transformers only if transformers accepts it (introspected).
   - `patches/tokenization_mistral_common.py` — backports huggingface/transformers#44760
     (the `reasoning_effort` plumbing) into 4.57.3. The proper fix is transformers 5.x, but
     **transformers 5.x requires Python ≥ 3.13 and this SM110 image is py3.12**, so a
     straight upgrade is impossible — hence the backport. `mistral_common 1.11.0` (in the
     image) already has the `reasoning_effort` field, so only the shim was missing.

   Result: chat + Mistral tool-calling work. `/v1/completions` was never affected.

## Reasoning mode status (partial on this image)

`reasoning_effort="high"` is correctly injected into the prompt as
`[MODEL_SETTINGS]{"reasoning_effort":"high"}[/MODEL_SETTINGS]` (verified via
`/v1/chat/completions/render`). But two things are imperfect on this stack:

- This NVFP4 checkpoint **won't open the `[THINK]` turn on its own** (mistral_common 1.11.0
  doesn't append it). The transformers patch therefore **prefills `[THINK]`** for `high`, which
  reliably makes the model produce a reasoning trace.
- Because the opening `[THINK]` is in the *prompt*, `reasoning_parser="mistral"` doesn't split
  the trace into `reasoning_content` — the reasoning lands inline in `content` (and the model
  emits the closing `[/THINK]` inconsistently). Clean `reasoning_content` separation needs the
  full transformers-5.x / newer-mistral_common stack, which is gated by Python 3.13.

So: instruct mode (`"none"`, default) is fully production-clean; reasoning *works* (the model
reasons) but isn't cleanly delimited via the API on this image. Clients can split on `[/THINK]`.

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
download-models.sh                       # hf download -> ./models/  (weights gitignored)
build-image.sh                           # build derived image (patches baked in)
Dockerfile                               # base image + the two patches
serve.sh                                 # production vLLM server (:8002), default = Eagle-3
benchmark.sh                             # base + Eagle decode benchmark (:8003)
patches/mistral.py                       # vLLM tokenizer wrapper patch
patches/tokenization_mistral_common.py   # transformers reasoning_effort backport (#44760)
config/                                  # auto-generated runtime YAML (gitignored)
models/                                  # downloaded weights (gitignored)
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
