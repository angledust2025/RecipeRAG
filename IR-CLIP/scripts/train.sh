#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:?Set MODEL to checkpoints/base, checkpoints/large, or another IR-CLIP model directory}
TRAIN_JSON=${TRAIN_JSON:?Set TRAIN_JSON to the Recipe1M training JSON/JSONL}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/irclip}
NUM_GPUS=${NUM_GPUS:-1}
IMAGE_ROOT=${IMAGE_ROOT:-}

ARGS=(--model_name_or_path "$MODEL" --data_path "$TRAIN_JSON" --output_dir "$OUTPUT_DIR"
      --per_device_train_batch_size 64 --gradient_accumulation_steps 1
      --add_seg_loss True
      --num_train_epochs 4 --learning_rate 1e-5 --weight_decay 0.001
      --warmup_ratio 0.03 --lr_scheduler_type cosine --save_strategy epoch
      --logging_steps 10 --dataloader_num_workers 8 --remove_unused_columns False
      --bf16 True --report_to none)
if [[ -n "$IMAGE_ROOT" ]]; then ARGS+=(--image_root "$IMAGE_ROOT"); fi
if (( NUM_GPUS > 1 )); then
  torchrun --nproc_per_node="$NUM_GPUS" --module train.train_irclip "${ARGS[@]}"
else
  python -m train.train_irclip "${ARGS[@]}"
fi
