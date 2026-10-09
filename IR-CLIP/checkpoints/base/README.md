---
license: apache-2.0
base_model: openai/clip-vit-base-patch16
language:
  - en
tags:
  - image-text-retrieval
  - cross-modal-retrieval
  - recipe1m
  - clip
  - ir-clip
---

# IR-CLIP Base

IR-CLIP Base is a CLIP-based model for cross-modal food image and recipe retrieval. It is initialized from OpenAI CLIP ViT-B/16 and fine-tuned on Recipe1M with separate text representations for recipe titles, ingredients, and instructions.

## Intended use

Use the model to retrieve recipes from food images or retrieve food images from recipe text. The checkpoint requires the custom `IRCLIPModel` implementation included in the IR-CLIP code repository.

## Model details

- Vision encoder: CLIP ViT-B/16
- Text and image embedding dimension: 512
- Checkpoint precision: BF16
- Training: 4 epochs; foreground alignment during training uses SAM3-generated food/drink masks
- Retrieval inference uses the original food image; segmented images are not required

## Evaluation

Recipe1M cross-modal retrieval results reported in the project paper. MedR is lower-is-better; Recall@K is higher-is-better.

| Test set | Direction | MedR | R@1 | R@5 | R@10 |
|---|---|---:|---:|---:|---:|
| 1K | Image-to-Recipe | 1.0 | 79.1 | 95.6 | 97.7 |
| 1K | Recipe-to-Image | 1.0 | 79.2 | 95.8 | 98.1 |
| 10K | Image-to-Recipe | 1.0 | 52.0 | 78.0 | 85.6 |
| 10K | Recipe-to-Image | 1.0 | 52.4 | 78.4 | 86.0 |

## Loading

Clone the IR-CLIP code repository and install its requirements, then load this checkpoint with the custom model class:

```python
from transformers import CLIPImageProcessor, CLIPTokenizer
from model.clip_strc.irclip import IRCLIPModel

model_id = "angledust/IR-CLIP-base"
model = IRCLIPModel.from_pretrained(model_id)
tokenizer = CLIPTokenizer.from_pretrained(model_id)
image_processor = CLIPImageProcessor.from_pretrained(model_id)
```

For the project's retrieval preprocessing and feature combination, see `retrieval_global.py` in the IR-CLIP code repository.

## Limitations

The model is optimized for food images and recipe text. Retrieval scores depend on the candidate set and preprocessing; they should not be interpreted as calibrated probabilities. The benchmark figures above are from the project's Recipe1M evaluation setup.
