# Stage 1

Stage 1 uses LlamaFactory to fine-tune Qwen3-VL-8B with LoRA on the CoT RAG recipe data. Model weights are not included. The merged checkpoint can be downloaded from Hugging Face into `output/qwen3-vl-8b-cot-sft-rag_merged/` using the commands in the [release README](../../README.md#model-checkpoints).

## Setup

Clone LlamaFactory and install its dependencies following its installation instructions. In the cloned repository, replace `data/` with this package's `data/` directory, and copy `examples/train_lora/qwen3vl_lora_sft.yaml` to the same path in LlamaFactory:

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cp -r /path/to/GRPO/stage1/data/. /path/to/LLaMA-Factory/data/
cp /path/to/GRPO/stage1/examples/train_lora/qwen3vl_lora_sft.yaml \
  /path/to/LLaMA-Factory/examples/train_lora/qwen3vl_lora_sft.yaml
```

The YAML uses `Qwen/Qwen3-VL-8B-Instruct` as its starting model; LlamaFactory downloads it from Hugging Face. Change `model_name_or_path` if you use a local copy. Image files are not included. Place the Recipe1M `train/` image directory under `LLaMA-Factory/data/`; JSON image paths are relative to that directory. The training configuration uses BF16 and expects suitable GPU memory for Qwen3-VL-8B.

## Train

Run the supplied launcher with `LLAMAFACTORY_DIR` set to the cloned repository:

```bash
LLAMAFACTORY_DIR=/path/to/LLaMA-Factory bash scripts/train.sh
```

Training saves the LoRA adapter to `LLaMA-Factory/saves/qwen3-vl-8b-cot-sft-rag/lora/sft/`. To use the merged checkpoint as Stage 2's starting point, download `angledust/RecipeRAG-SFT` into the Stage 1 `output/` path described above.
