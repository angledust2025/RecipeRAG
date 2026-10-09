# RecipeRAG

RecipeRAG is a multimodal recipe retrieval and generation pipeline. It combines image–recipe retrieval with supervised fine-tuning and GRPO training to generate recipe content grounded in visually and textually similar examples.

## Pipeline

1. **IR-CLIP** learns joint image and recipe representations. Optional SAM3 foreground images support foreground-aware training.
2. **Retrieval** builds FAISS indexes, searches by image similarity and image–recipe similarity, then combines the ranked candidates with Reciprocal Rank Fusion (RRF).
3. **Stage 1** fine-tunes Qwen3-VL-8B with LLaMA-Factory using retrieved recipe examples.
4. **Stage 2** applies GRPO with EasyR1, starting from the Stage 1 SFT model.

## Model checkpoints

Model weights are hosted on Hugging Face and are downloaded separately. Install the Hugging Face CLI with `python -m pip install -U huggingface_hub`. If required, authenticate with `hf auth login`.

Run these commands from the repository root to download checkpoints to the paths used by the scripts:

```bash
# IR-CLIP retrieval checkpoints
hf download angledust/IR-CLIP-base --local-dir IR-CLIP/checkpoints/base
hf download angledust/IR-CLIP-large --local-dir IR-CLIP/checkpoints/large

# Stage 1 SFT checkpoint used to initialize Stage 2
hf download angledust/RecipeRAG-SFT \
  --local-dir GRPO/stage1/output/qwen3-vl-8b-cot-sft-rag_merged

# Optional Stage 2 RFT checkpoint
hf download angledust/RecipeRAG-RFT \
  --local-dir GRPO/stage2/output/RecipeRAG-RFT
```

## Documentation

| Component | Guide |
| --- | --- |
| IR-CLIP training and evaluation | [IR-CLIP/README.md](IR-CLIP/README.md) |
| FAISS retrieval | [Retrieval/README.md](Retrieval/README.md) |
| Stage 1 SFT | [GRPO/stage1/README.md](GRPO/stage1/README.md) |
| Stage 2 GRPO | [GRPO/stage2/README.md](GRPO/stage2/README.md) |
