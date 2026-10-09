#!/usr/bin/env bash
set -euo pipefail

STAGE2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GRPO_DIR="$(cd "$STAGE2_DIR/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-$GRPO_DIR/stage1/output/qwen3-vl-8b-cot-sft-rag_merged}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_vl_8b_recipe_grpo}"

cd "$STAGE2_DIR"
python3 -m verl.trainer.main \
  config=examples/config.yaml \
  data.train_files="$STAGE2_DIR/dataset/recipe@train" \
  data.format_prompt="$STAGE2_DIR/examples/format_prompt/recipe_cot.jinja" \
  worker.actor.model.model_path="$MODEL_PATH" \
  worker.reward.reward_function="$STAGE2_DIR/examples/reward_function/reward.py:compute_score" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.n_gpus_per_node="$N_GPUS_PER_NODE"
