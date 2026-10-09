"""Convert the top-1-filtered RecipeRAG JSON into EasyR1 training Parquet shards.

Images are read from the paths in the JSON. Relative image paths can be resolved
with ``--image-root``. No image files are copied into this repository.
"""

import argparse
import json
import math
import random
from pathlib import Path

from datasets import Dataset, Image as ImageData, Sequence
from PIL import Image


def process_single_item(item):
    try:
        image = Image.open(item["image_path"]).convert("RGB")
        return {
            "images": [image],
            "problem": item["problem"],
            "answer": item["answer"],
            "valid": True,
            "error": "",
        }
    except Exception as exc:
        return {
            "images": [],
            "problem": item["problem"],
            "answer": item["answer"],
            "valid": False,
            "error": str(exc),
        }


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-json",
        type=Path,
        required=True,
        help="Original recipe1m_train_cot_sft_rag_top1_filtered.json file.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Prefix for relative image paths; absolute paths in the JSON are used as-is.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "recipe")
    parser.add_argument("--max-samples", type=int, default=50000,
                        help="Randomly sample up to this many records; use 0 to process all records.")
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--num-processes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = get_args()
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be positive.")
    if args.num_processes < 1:
        raise ValueError("--num-processes must be positive.")
    if args.max_samples < 0:
        raise ValueError("--max-samples must be non-negative; use 0 for all records.")

    with args.input_json.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    rng = random.Random(args.seed)
    if args.max_samples and len(raw_data) > args.max_samples:
        raw_data = rng.sample(raw_data, args.max_samples)

    flat_data = []
    for item in raw_data:
        conversations = item["conversations"]
        prompt = next(message["value"] for message in conversations if message["from"] == "human")
        answer = next(message["value"] for message in conversations if message["from"] == "gpt")
        image_path = Path(item["images"][0]).expanduser()
        if not image_path.is_absolute():
            if args.image_root is None:
                raise ValueError(
                    f"Relative image path {image_path} requires --image-root."
                )
            image_path = args.image_root / image_path
        flat_data.append({"problem": prompt, "answer": answer, "image_path": str(image_path)})

    rng.shuffle(flat_data)
    raw_dataset = Dataset.from_list(flat_data)
    processed = raw_dataset.map(
        process_single_item,
        num_proc=args.num_processes,
        remove_columns=raw_dataset.column_names,
        desc="Loading Recipe1M images",
        load_from_cache_file=False,
    )
    bad = processed.filter(lambda item: not item["valid"])
    if len(bad):
        print(f"Skipping {len(bad)} records whose images could not be read.")
        if "error" in bad.column_names:
            print(f"Example image error: {bad[0]['error']}")
    processed = processed.filter(lambda item: item["valid"]).remove_columns(["valid", "error"])
    processed = processed.cast_column("images", Sequence(ImageData()))

    if len(processed) == 0:
        raise ValueError("No readable images were found; check the JSON image paths and --image-root.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    n_shards = math.ceil(len(processed) / args.chunk_size)
    for shard_index in range(n_shards):
        start = shard_index * args.chunk_size
        end = min(start + args.chunk_size, len(processed))
        shard = processed.select(range(start, end))
        filename = f"train-{shard_index:05d}-of-{n_shards:05d}.parquet"
        shard.to_parquet(args.output_dir / filename)
        print(f"Wrote {filename} ({len(shard)} rows)")

    print(f"Finished: {len(processed)} training examples; no validation split was created.")


if __name__ == "__main__":
    main()
