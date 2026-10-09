#!/usr/bin/env python3
"""Build normalized image and recipe FAISS indexes from an IR-CLIP checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss

from common import IRCLIPEncoder, read_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_checkpoint = Path(__file__).resolve().parents[1] / "IR-CLIP" / "checkpoints" / "large"
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint,
                        help="IR-CLIP checkpoint (defaults to the sibling large checkpoint)")
    parser.add_argument("--data", type=Path, required=True, help="Candidate recipes as JSON array or JSONL")
    parser.add_argument("--image-root", type=Path, default=None, help="Root prefix for relative image paths")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.device.startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            parser.error("CUDA requested but unavailable; pass --device cpu to use CPU")

    rows = read_rows(args.data)
    required = ("image_path", "title", "ingredients", "instructions")
    for i, row in enumerate(rows):
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"Row {i} is missing required fields: {', '.join(missing)}")
    if not rows:
        raise ValueError("Candidate dataset is empty")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    encoder = IRCLIPEncoder(args.checkpoint, args.device)
    image_vectors = encoder.encode_images(rows, args.image_root, args.batch_size)
    text_vectors = encoder.encode_recipes(rows, args.batch_size)

    image_index = faiss.IndexFlatIP(image_vectors.shape[1])
    text_index = faiss.IndexFlatIP(text_vectors.shape[1])
    image_index.add(image_vectors)
    text_index.add(text_vectors)
    faiss.write_index(image_index, str(args.output_dir / "images.faiss"))
    faiss.write_index(text_index, str(args.output_dir / "recipes.faiss"))
    (args.output_dir / "metadata.json").write_text(
        json.dumps(rows, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "index_info.json").write_text(
        json.dumps({
            "checkpoint": str(args.checkpoint),
            "source_data": str(args.data),
            "count": len(rows),
            "metric": "inner_product_on_l2_normalized_vectors",
            "image_index": "images.faiss",
            "recipe_index": "recipes.faiss",
        }, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Built {len(rows)} aligned image and recipe vectors in {args.output_dir}")


if __name__ == "__main__":
    main()
