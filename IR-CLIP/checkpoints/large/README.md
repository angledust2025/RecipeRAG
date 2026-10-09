---
license: apache-2.0
base_model: openai/clip-vit-large-patch14-336
language:
  - en
tags:
  - image-text-retrieval
  - cross-modal-retrieval
  - recipe1m
  - clip
  - ir-clip
---

# IR-CLIP Large

IR-CLIP Large is a CLIP-based model for cross-modal food image and recipe retrieval. It is initialized from OpenAI CLIP ViT-L/14 at 336px and fine-tuned on Recipe1M with separate text representations for recipe titles, ingredients, and instructions.

## Intended use

Use the model to retrieve recipes from food images or retrieve food images from recipe text. The checkpoint requires the custom `IRCLIPModel` implementation included in the IR-CLIP code repository.

## Model details

- Vision encoder: CLIP ViT-L/14 at 336px
- Vision hidden size: 1024; text hidden size: 768; joint embedding dimension: 768
- Checkpoint precision: BF16
- Training: 4 epochs; foreground alignment during training uses SAM3-generated food/drink masks
- Retrieval inference uses the original food image; segmented images are not required

## Evaluation

Recipe1M cross-modal retrieval results reported in the project paper. MedR is lower-is-better; Recall@K is higher-is-better.

| Test set | Direction | MedR | R@1 | R@5 | R@10 |
|---|---|---:|---:|---:|---:|
| 1K | Image-to-Recipe | 1.0 | 85.1 | 97.6 | 99.0 |
| 1K | Recipe-to-Image | 1.0 | 85.9 | 97.8 | 99.2 |
| 10K | Image-to-Recipe | 1.0 | 61.5 | 85.4 | 91.2 |
| 10K | Recipe-to-Image | 1.0 | 62.3 | 86.0 | 91.6 |

## Loading

Clone the IR-CLIP code repository and install its requirements, then load this checkpoint with the custom model class:

```python
from transformers import CLIPImageProcessor, CLIPTokenizer
from model.clip_strc.irclip import IRCLIPModel

model_id = "angledust/IR-CLIP-large"
model = IRCLIPModel.from_pretrained(model_id)
tokenizer = CLIPTokenizer.from_pretrained(model_id)
image_processor = CLIPImageProcessor.from_pretrained(model_id)
```

For the project's retrieval preprocessing and feature combination, see `retrieval_global.py` in the IR-CLIP code repository.

## Limitations

The model is optimized for food images and recipe text. Retrieval scores depend on the candidate set and preprocessing; they should not be interpreted as calibrated probabilities. The benchmark figures above are from the project's Recipe1M evaluation setup.
