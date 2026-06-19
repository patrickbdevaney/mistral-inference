# Derived image: NVIDIA Jetson Thor vLLM + reasoning_effort support for Mistral Small 4.
#
# The base image ships vllm 0.19.0 / transformers 4.57.3 / mistral_common 1.11.0 (Python 3.12).
# mistral_common 1.11.0 ALREADY supports the reasoning_effort field, but:
#   * transformers 4.57.3 MistralCommonTokenizer.apply_chat_template rejects the kwarg and
#     doesn't forward it (the reasoning_effort plumbing is huggingface/transformers#44760,
#     which only exists in transformers 5.x — and transformers 5.x requires Python >= 3.13,
#     which this SM110 image is not). So we backport just that plumbing into 4.57.3.
#   * vllm 0.19.0 forwards reasoning_effort to the tokenizer only via a guarded path; the
#     patched vllm tokenizer forwards it iff transformers actually accepts it (introspected).
#
# Net effect vs the base image:
#   * /v1/chat/completions works (base image 400s on every request)
#   * reasoning_effort="high" actually triggers reasoning ([THINK] / reasoning_content);
#     "none" / absent = instruct mode
#
# Build:  docker build -t mistral-small4-thor:reasoning .
# (or run ./build-image.sh)

FROM ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor

# vLLM tokenizer wrapper: forward reasoning_effort to transformers iff it accepts it.
COPY patches/mistral.py \
     /opt/venv/lib/python3.12/site-packages/vllm/tokenizers/mistral.py

# transformers MistralCommonTokenizer: backport reasoning_effort plumbing (#44760).
COPY patches/tokenization_mistral_common.py \
     /opt/venv/lib/python3.12/site-packages/transformers/tokenization_mistral_common.py

# Fail the build early if either patch is syntactically broken.
RUN python3 -m py_compile \
      /opt/venv/lib/python3.12/site-packages/vllm/tokenizers/mistral.py \
      /opt/venv/lib/python3.12/site-packages/transformers/tokenization_mistral_common.py
