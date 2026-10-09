# Stage 2: EasyR1 GRPO

Stage 2 uses the EasyR1/veRL implementation to run ordinary GRPO from the Stage 1 merged checkpoint. The reward function in `examples/reward_function/reward.py` follows the R1V version: it requires the `<thinking>`, `<title>`, `<ingredients>`, and `<instructions>` tags, then combines title similarity, ingredient F1, instruction ROUGE-L precision, and instruction BLEU with sigmoid or exponential mappings. It uses `sentence-transformers/all-MiniLM-L6-v2` for title similarity by default. Set `TITLE_EMBEDDING_MODEL` to a local model path when running offline.

Model weights are not included in this package. Download `angledust/RecipeRAG-SFT` into `../stage1/output/qwen3-vl-8b-cot-sft-rag_merged` using the commands in the [release README](../../README.md#model-checkpoints), or set `MODEL_PATH` to another compatible local checkpoint.

## Environment

Use a Linux environment with the CUDA, PyTorch, vLLM, and EasyR1 dependencies required by this code. Install the package and reward dependencies from this directory:

```bash
pip install -e .
```

## Prepare data

Stage 2 uses training data prepared after the recipe retrieval step. The prepared data and Recipe1M images are not included here. Provide the path to your prepared JSON file and the image root. The JSON must be an array of records with `images` and `conversations` fields; each record needs an image path and human/GPT messages. Absolute image paths are used as-is; relative paths are resolved under `--image-root`.

For example, if your retrieval pipeline has produced `retrieval_prepared_train.json`:

```bash
python dataset/process_rag.py \
  --input-json /path/to/retrieval_prepared_train.json \
  --image-root /path/to/Recipe1M
```

By default, the script randomly samples up to 50,000 records (seed 42), reads their images, and writes only training Parquet shards to `dataset/recipe/`. It does not create a validation split. Use `--max-samples 0` to process every record in the source JSON.

## Train

After installing dependencies and preprocessing the images, the launcher uses the Stage 1 merged checkpoint by default. All processed rows are used for training; validation is disabled. Override `MODEL_PATH` if the checkpoint is stored elsewhere. The default config requests 4 GPUs on one node; adjust `N_GPUS_PER_NODE` to match the available GPUs and rollout tensor-parallel configuration:

```bash
bash examples/train_recipe_grpo.sh
MODEL_PATH=/path/to/qwen3-vl-8b-cot-sft-rag_merged N_GPUS_PER_NODE=4 \
  bash examples/train_recipe_grpo.sh
```

The GRPO configuration is in `examples/config.yaml`. Checkpoints and run logs are written under `checkpoints/easy_r1/`.
