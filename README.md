# Mistral Small 4 (119B NVFP4) with EAGLE-3 on Jetson AGX Thor

**Serving recipe for `Mistral-Small-4-119B-2603-NVFP4` with EAGLE-3 speculative decoding on the NVIDIA Jetson vLLM
container, including the chat-template fixes that container needs**

| | |
|---|---|
| **Model** | [`mistralai/Mistral-Small-4-119B-2603-NVFP4`](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603-NVFP4): 119B total, 6.5B active; MoE with 128 experts, 4 active; DeepSeek-V2-style MLA; Pixtral wrapper, served text-only; ~66 GB |
| **Draft** | [`mistralai/Mistral-Small-4-119B-2603-eagle`](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603-eagle) (EAGLE-3, ~0.4 GB), 3 speculative tokens |
| **Runtime** | `ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor` (vLLM 0.19.0, transformers 4.57.3, mistral_common 1.11.0, Python 3.12), plus a derived image with two patches |
| **Target** | NVIDIA Jetson AGX Thor (`sm_110`), 128 GB unified memory, `nvpmodel -m 1` |
| **Kernels** | `TRITON_MLA` attention, FlashInfer-CUTLASS NVFP4 GEMM, CUTLASS MoE (no Marlin fallback) |
| **Validated** | 2026-06-19 |

---

## Performance

| Configuration | Decode | Speedup | Context |
|---|--:|--:|--:|
| Base (no speculation) | ~17.5 tok/s | — | 262,144 tokens |
| EAGLE-3, 3 speculative tokens | **~33.7 tok/s** (up to ~44 in chat) | **1.93×** | 65,536 tokens |

Unified memory at `gpu_memory_utilization=0.72` is ~96 GB used, ~26 GB free.

**Test conditions** ([`benchmark.sh`](benchmark.sh)):
- `/v1/completions` with one fixed coding prompt (a red-black tree implementation);
- 512 generated tokens per run, 3 runs per configuration, single stream;
- the rate is completion tokens divided by total request time, so it includes time to first token.

The ~44 tok/s chat figure was observed on individual chat requests rather than through this harness. Accuracy against
the BF16 model was not evaluated here.

### Instruct versus reasoning: time to answer

Decode speed is the same in both modes. What differs is how many tokens the model emits. Mistral Small 4 answers
directly by default (`reasoning_effort="none"`); reasoning is opt-in per request.

| Question | Mode | Tokens | Wall time | Answer |
|---|---|--:|--:|---|
| "17 × 23?" | instruct | 12 | 0.4 s | 391 |
| | reasoning | 181 | 4.7 s | 391 (15× tokens, 10.6× wall time) |
| "Is 2027 prime?" | instruct | 632 | 14.3 s | yes |
| | reasoning | 1,394 | 31.6 s | yes (2.2× tokens and wall time) |

Instruct mode gets to the answer faster at equal decode speed. Use reasoning only when the task warrants it.

---

## Quick start

```bash
./download-models.sh     # NVFP4 checkpoint (~66 GB) and EAGLE-3 draft (~0.4 GB) -> ./models/
./build-image.sh         # derived image mistral-small4-thor:reasoning (patches baked in; recommended)
./serve.sh               # OpenAI-compatible server on :8002, EAGLE-3, 64K context
./benchmark.sh           # optional: base and EAGLE-3 decode on :8003

./serve.sh no-eagle      # no speculation, full 256K context
```

**Prerequisites:**
- Jetson AGX Thor with Docker and the `nvidia` runtime;
- the `hf` CLI (`pip install -U huggingface_hub`);
- `sudo` for `nvpmodel`, `jetson_clocks` and dropping the page cache.

`serve.sh` and `benchmark.sh` use the derived image when it exists. Otherwise they use the base image and mount the
two patches read-only. Override with `REASONING_IMAGE=<tag>`.

**Verify:** the startup log should show `TRITON_MLA` and `FLASHINFER_CUTLASS` (NVFP4). `marlin_utils_fp4` in the log
means the wrong kernel path was selected.

---

## Configuration reference

### Serving profiles (`serve.sh`)

| Setting | `default` | `no-eagle` |
|---|---|---|
| Speculation | EAGLE-3, `num_speculative_tokens: 3` | off |
| `max_model_len` | 65,536 | 262,144 |
| `max_num_seqs` | 1 | 1 |
| `gpu_memory_utilization` | 0.72 | 0.72 |
| `max_num_batched_tokens` | 8,192 (chunked prefill) | 8,192 |
| CUDA graphs | `PIECEWISE`, capture sizes 4 / 8 / 12 | same |
| Prefix caching | off | off |
| Parsers | `tokenizer_mode: mistral`, `reasoning_parser: mistral`, `tool_call_parser: mistral` | same |

The runtime YAML is generated into `config/` on each launch.

### Image contents

| Component | Base image | Derived image (`./build-image.sh`) |
|---|---|---|
| vLLM | 0.19.0 | 0.19.0 + `patches/mistral.py` |
| transformers | 4.57.3 | 4.57.3 + `patches/tokenization_mistral_common.py` |
| mistral_common | 1.11.0 | 1.11.0 |
| `/v1/chat/completions` | **fails with 400 on every request** | works, including tool calling |
| `reasoning_effort` | rejected | honored (see [Reasoning mode](#reasoning-mode)) |

The derived image's build step runs `py_compile` on both patched files, so a broken patch fails the build rather
than the server.

---

## Patches

**The problem.** vLLM 0.19.0 forwards `reasoning_effort` to `MistralCommonTokenizer.apply_chat_template` in
transformers 4.57.3. That function rejects all extra keyword arguments, so every chat request in the stock image
returns 400. `/v1/completions` is unaffected.

| Patch | Replaces | Change |
|---|---|---|
| [`patches/mistral.py`](patches/mistral.py) | `vllm/tokenizers/mistral.py` | Forwards `reasoning_effort` only if the installed transformers accepts it, checked by signature introspection |
| [`patches/tokenization_mistral_common.py`](patches/tokenization_mistral_common.py) | `transformers/tokenization_mistral_common.py` | Backports the `reasoning_effort` plumbing from [huggingface/transformers#44760](https://github.com/huggingface/transformers/pull/44760) into 4.57.3 |

**Why a backport instead of an upgrade.** The upstream fix ships in transformers 5.x, which requires Python ≥ 3.13.
This `sm_110` image is Python 3.12. mistral_common 1.11.0 already supports the `reasoning_effort` field, so only the
transformers plumbing was missing.

---

## Reasoning mode

| Mode | Request | Behavior on this image |
|---|---|---|
| Instruct (default) | `reasoning_effort` absent or `"none"` | Clean `content`; production-ready |
| Reasoning | `reasoning_effort: "high"` | The model reasons, but the trace arrives inline in `content`, not in `reasoning_content` |

**Why reasoning output is not split cleanly:**
- `reasoning_effort="high"` is injected correctly as `[MODEL_SETTINGS]{"reasoning_effort":"high"}[/MODEL_SETTINGS]`.
  This was verified with `/v1/chat/completions/render`.
- This NVFP4 checkpoint does not open the `[THINK]` turn by itself, and mistral_common 1.11.0 does not append it. The
  transformers patch therefore prefills `[THINK]` for `"high"`.
- Because `[THINK]` is then part of the prompt, `reasoning_parser: mistral` cannot separate the trace. The model also
  emits the closing `[/THINK]` inconsistently.

Clients that need the trace separated can split on `[/THINK]`. A clean fix needs transformers 5.x and a newer
mistral_common, which this image's Python 3.12 rules out.

---

## API examples

```bash
# Chat
curl -s localhost:8002/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"/model","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'

# Reasoning on request
curl -s localhost:8002/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"/model","reasoning_effort":"high","messages":[{"role":"user","content":"Is 2027 prime?"}]}'

# Tool calling (Mistral parser)
curl -s localhost:8002/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"/model","messages":[{"role":"user","content":"Weather in Paris? Use the tool."}],
       "tools":[{"type":"function","function":{"name":"get_weather",
       "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]}'

# Raw completion
curl -s localhost:8002/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"/model","prompt":"def fib(n):","max_tokens":128}'
```

---

## Troubleshooting

| Symptom | Cause | Resolution |
|---|---|---|
| `Selected backend FLASHINFER ... ['head_size not supported','MLA not supported']` at load | The model uses MLA; FlashInfer attention has no MLA kernels | Use `--attention-backend TRITON_MLA` (the scripts do) |
| Every `/v1/chat/completions` request returns 400 | Stock image rejects `reasoning_effort` | Build the derived image, or let `serve.sh` mount the patches |
| `Free memory on device cuda:0 ... less than gpu_memory_utilization` on relaunch | The previous container's 66 GB of weights are still in the page cache | `sudo sync && sudo sysctl -w vm.drop_caches=3`. The scripts loop until more than 95 GB is free |
| `huggingface-cli download` does nothing | It is a no-op in `huggingface_hub` ≥ 1.x | Use `hf download` |
| No `config.json` in the checkpoint | Mistral consolidated format (`params.json`, `consolidated*.safetensors`, `tekken.json`) | Expected; vLLM detects it with `tokenizer_mode: mistral` |
| Reasoning text appears inside `content` | `[THINK]` is prefilled in the prompt | Expected on this image; split on `[/THINK]` |

---

## Known limitations

- **Single stream** (`max_num_seqs: 1`). Throughput under concurrency was not measured.
- **Text only.** The Pixtral vision path is disabled with `--language-model-only`.
- **Reasoning output is not separated** into `reasoning_content` (see [Reasoning mode](#reasoning-mode)).
- **Throughput comes from one prompt type** and three runs per configuration; it is not a benchmark suite.
- **The base tag is not pinned.** `latest-jetson-thor` was vLLM 0.19.0 when this was validated (2026-06), and the
  patches replace whole files of that exact version. If the tag moves, re-check the patches before using them; a
  newer base may not need them at all.

## Security considerations

`vllm serve` binds `0.0.0.0` with host networking and no API key, and runs with `trust_remote_code`. Expose it only on
a trusted network, or add `--api-key` behind a TLS reverse proxy.

## Repository layout

| Path | Contents |
|---|---|
| [`download-models.sh`](download-models.sh) | `hf download` of the checkpoint and draft into `./models/` (git-ignored) |
| [`Dockerfile`](Dockerfile), [`build-image.sh`](build-image.sh) | Derived image with both patches baked in |
| [`serve.sh`](serve.sh) | Production server on :8002 (`default` or `no-eagle`) |
| [`benchmark.sh`](benchmark.sh) | Base and EAGLE-3 decode benchmark on :8003 |
| [`patches/`](patches/) | Patched vLLM tokenizer wrapper and transformers backport |

## License

- **This repository:** Apache-2.0 ([`LICENSE`](LICENSE)). Both patched files are derived from Apache-2.0 sources
  (vLLM; transformers by Mistral AI and Hugging Face) and keep their original headers.
- **Mistral Small 4 and its EAGLE-3 draft:** Apache-2.0 (Mistral AI). No weights are distributed here.
- **NVIDIA Jetson container images** are governed by their NVIDIA license terms.

This is an independent community project, not affiliated with NVIDIA or Mistral AI.
