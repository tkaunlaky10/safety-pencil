#!/usr/bin/env bash
# Hinglish versions of the harmful prompts, and XSTest in Hindi and Hinglish (CPU job, about 30 min).
#
# The LLM is Aya Expanse 32B through Hugging Face's inference router, billed to the account's credits, so no
# GPU is needed. Set HINGLISH_MODEL and HINGLISH_API_BASE to use another LiteLLM model, e.g. a vLLM server.
# Pushes the new data and the review samples (prompts only: rejected harmful outputs are never written).
set -euo pipefail

python jobs/sync.py pull "data/processed/**"

export OPENAI_API_KEY=$HF_TOKEN  # LiteLLM's OpenAI-compatible client authenticates to the router with it
llm=(
  --model "${HINGLISH_MODEL:-openai/CohereLabs/aya-expanse-32b:cohere}"
  --api-base "${HINGLISH_API_BASE:-https://router.huggingface.co/v1}"
)
python scripts/01_make_hinglish.py --set harmful "${llm[@]}"
python jobs/sync.py push data/processed/harmful/hinglish outputs/review

python scripts/01_build_xstest.py "${llm[@]}"
python scripts/01_check_contamination.py
python jobs/sync.py push data/processed/xstest outputs/review outputs/results
