# IR-CLIP Retrieval

IR-CLIP weights are not included in this package. Download `angledust/IR-CLIP-large` to `IR-CLIP/checkpoints/large` using the commands in the [release README](../README.md#download-model-weights). The scripts use this large checkpoint by default.

## Environment

Install a PyTorch build compatible with your CUDA version, then install dependencies from this directory:

```bash
python -m pip install -r requirements.txt
```

The scripts use the sibling `IR-CLIP` package and checkpoint. The default checkpoint path is `IR-CLIP/checkpoints/large`.

## Data format

Input may be a JSON array or JSONL file. Each record uses the same fields as `IR-CLIP/data/test_sample_1000.json`:

```json
{"id":"recipe-001","image_id":"image-001.jpg","image_path":"Recipe1M/train/0/0/0/0/image-001.jpg","title":"Recipe title","ingredients":"Ingredient list","instructions":"Recipe instructions"}
```

Relative image paths are resolved beneath `--image-root`. The indexed candidate data and query data must use compatible IDs and metadata. Training images and test images should be stored under the same image root, following the paths in their records.

## Run

Run commands from the `Retrieval/` directory. Provide your own Recipe1M JSON files and image directory; these data files are not part of this repository. First, build image and recipe indexes from the training split:

```bash
python build_index.py \
  --data /path/to/recipe1m_train.json \
  --image-root /path/to/Recipe1M \
  --output-dir outputs/large/train-index \
  --device cuda
```

Retrieve training examples against the training index. This mode removes the query itself and candidates with image-image similarity `>= 0.99`:

```bash
python retrieve.py \
  --index-dir outputs/large/train-index \
  --query-data /path/to/recipe1m_train.json \
  --image-root /path/to/Recipe1M \
  --mode train-to-train \
  --output outputs/large/train-retrieval.json \
  --device cuda
```

Retrieve test examples against the same training index. This mode does not remove self matches or near duplicates:

```bash
python retrieve.py \
  --index-dir outputs/large/train-index \
  --query-data /path/to/recipe1m_test.json \
  --image-root /path/to/Recipe1M \
  --mode test-to-train \
  --output outputs/large/test-retrieval.json \
  --device cuda
```

Both retrieval modes intersect the image-image and image-recipe Top-100 results and rank the common candidates with RRF. Output is a JSON array with one `retrieved_top1` result per query; it is `[]` when no valid candidate is found. Use `--checkpoint` to select another IR-CLIP checkpoint, and `--device cpu` to run without CUDA.
