# RecipeRAG Release

This release contains the code for the cross-modal recipe retrieval and RecipeRAG training pipeline. Model weights are hosted on Hugging Face and are downloaded separately.

## Workflow

1. **Train IR-CLIP** on paired recipe images and text. Optional SAM3 foreground images are used by the training loss.
2. **Build FAISS indexes and retrieve recipes** with IR-CLIP Large. Image-image and image-recipe candidates are fused with RRF. The retrieved recipe examples are used to prepare RecipeRAG data.
3. **Run Stage 1 SFT** with LlamaFactory on the image and retrieved-recipe prompts.
4. **Run Stage 2 RFT** with EasyR1 GRPO from the merged Stage 1 checkpoint.

Each package README describes its environment, data format, and commands. Start with [IR-CLIP](IR-CLIP/README.md), [Retrieval](Retrieval/README.md), [Stage 1](GRPO/stage1/README.md), or [Stage 2](GRPO/stage2/README.md), depending on the step you need.

## Download model weights

Install the Hugging Face CLI:

```bash
python -m pip install -U huggingface_hub
```

The repositories below are public. If Hugging Face requires authentication in your environment, log in with `hf auth login`.

Run these commands from this repository's root directory to download a model into the path expected by the corresponding scripts:

```bash
# IR-CLIP retrieval checkpoints
hf download angledust/IR-CLIP-base --local-dir IR-CLIP/checkpoints/base
hf download angledust/IR-CLIP-large --local-dir IR-CLIP/checkpoints/large

# Merged RecipeRAG SFT checkpoint; this is the default Stage 2 starting point
hf download angledust/RecipeRAG-SFT \
  --local-dir GRPO/stage1/output/qwen3-vl-8b-cot-sft-rag_merged

# Optional trained RecipeRAG RFT checkpoint for inference
hf download angledust/RecipeRAG-RFT \
  --local-dir GRPO/stage2/output/RecipeRAG-RFT
```

Stage 1 training also uses the public base model `Qwen/Qwen3-VL-8B-Instruct`; LlamaFactory downloads it from Hugging Face using the model ID in the Stage 1 YAML. The Stage 2 reward downloads `sentence-transformers/all-MiniLM-L6-v2` on first use, unless `TITLE_EMBEDDING_MODEL` points to a local copy.

The Stage 1 JSON file is larger than GitHub's 100 MB per-file limit and is tracked with Git LFS. Install Git LFS before cloning or pushing this repository; after cloning, run `git lfs pull` if the data file is still an LFS pointer.

## Data and licenses

Stage 1 includes its CoT SFT dataset. Stage 2's original top-1-filtered RAG JSON and Recipe1M image files are not bundled; provide their paths as described in the Stage 2 README. Check the license and access terms of each dataset and upstream model before use. The SAM3 segmentation model may require accepting its Hugging Face access conditions.
