#!/usr/bin/env bash
set -euo pipefail

LLAMAFACTORY_DIR="${LLAMAFACTORY_DIR:-}"
if [[ -z "$LLAMAFACTORY_DIR" ]]; then
  echo "Set LLAMAFACTORY_DIR to the cloned LlamaFactory repository." >&2
  exit 2
fi

cd "$LLAMAFACTORY_DIR"
llamafactory-cli train examples/train_lora/qwen3vl_lora_sft.yaml
