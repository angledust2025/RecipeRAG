#!/usr/bin/env python3
"""Retrieve recipes through image-image and image-recipe indexes, then fuse by RRF."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import faiss

from common import IRCLIPEncoder, read_rows, same_sample, sample_key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_checkpoint = Path(__file__).resolve().parents[1] / "IR-CLIP" / "checkpoints" / "large"
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint,
                        help="IR-CLIP checkpoint (defaults to the sibling large checkpoint)")
    parser.add_argument("--index-dir", type=Path, required=True, help="Directory created by build_index.py")
    parser.add_argument("--query-data", type=Path, required=True, help="Query recipes as JSON array or JSONL")
    parser.add_argument("--image-root", type=Path, default=None, help="Root prefix for relative query image paths")
    parser.add_argument("--mode", choices=("train-to-train", "test-to-train"), required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output JSON array (use .json)")
    parser.add_argument("--candidate-k", type=int, default=100, help="Candidates retrieved per modality before intersection")
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF rank constant")
    parser.add_argument("--max-image-similarity", type=float, default=0.99,
                        help="Drop candidates with image-image similarity >= this value")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.candidate_k < 1 or args.rrf_k < 0 or args.batch_size < 1:
        parser.error("candidate-k and batch-size must be positive; rrf-k must be non-negative")
    if args.device.startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            parser.error("CUDA requested but unavailable; pass --device cpu to use CPU")

    image_index_path = args.index_dir / "images.faiss"
    recipe_index_path = args.index_dir / "recipes.faiss"
    metadata_path = args.index_dir / "metadata.json"
    image_index = faiss.read_index(str(image_index_path))
    recipe_index = faiss.read_index(str(recipe_index_path))
    metadata: list[dict[str, Any]] = json.loads(metadata_path.read_text(encoding="utf-8"))
    if image_index.ntotal != len(metadata) or recipe_index.ntotal != len(metadata):
        raise ValueError("FAISS index vector count does not match metadata row count")

    queries = read_rows(args.query_data)
    for i, row in enumerate(queries):
        if not row.get("image_path"):
            raise ValueError(f"Query row {i} has no image_path")
    if not queries:
        raise ValueError("Query dataset is empty")

    encoder = IRCLIPEncoder(args.checkpoint, args.device)
    query_vectors = encoder.encode_images(queries, args.image_root, args.batch_size)
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_without_candidates = 0
    num_self_removed = 0
    num_near_duplicate_removed = 0

    with output_path.open("w", encoding="utf-8") as out:
        out.write("[\n")
        first_record = True
        for query_index, query in enumerate(queries):
            image_scores, image_ids = image_index.search(query_vectors[query_index:query_index + 1], args.candidate_k)
            text_scores, text_ids = recipe_index.search(query_vectors[query_index:query_index + 1], args.candidate_k)
            image_ranks = {sample_key(metadata[int(idx)]): rank
                           for rank, idx in enumerate(image_ids[0], start=1) if idx >= 0}
            text_ranks = {sample_key(metadata[int(idx)]): rank
                          for rank, idx in enumerate(text_ids[0], start=1) if idx >= 0}
            image_rows = {sample_key(metadata[int(idx)]): (int(idx), float(score), rank)
                          for rank, (score, idx) in enumerate(zip(image_scores[0], image_ids[0]), start=1) if idx >= 0}
            text_rows = {sample_key(metadata[int(idx)]): (int(idx), float(score), rank)
                         for rank, (score, idx) in enumerate(zip(text_scores[0], text_ids[0]), start=1) if idx >= 0}

            fused = []
            for key in image_ranks.keys() & text_ranks.keys():
                candidate_index, image_score, image_rank = image_rows[key]
                _, text_score, text_rank = text_rows[key]
                candidate = metadata[candidate_index]
                if any(not str(candidate.get(field, "") or "").strip()
                       for field in ("title", "ingredients", "instructions")):
                    continue
                if args.mode == "train-to-train" and same_sample(query, candidate):
                    num_self_removed += 1
                    continue
                if args.mode == "train-to-train" and image_score >= args.max_image_similarity:
                    num_near_duplicate_removed += 1
                    continue
                fusion_score = 1.0 / (args.rrf_k + image_rank) + 1.0 / (args.rrf_k + text_rank)
                fused.append({
                    "id": candidate.get("id", ""),
                    "image_id": candidate.get("image_id", ""),
                    "image_path": candidate.get("image_path", ""),
                    "title": candidate.get("title", ""),
                    "ingredients": candidate.get("ingredients", ""),
                    "instructions": candidate.get("instructions", ""),
                    "image_rank": image_rank,
                    "image_similarity": image_score,
                    "text_rank": text_rank,
                    "text_similarity": text_score,
                    "rrf_score": fusion_score,
                })
            fused.sort(key=lambda item: (-item["rrf_score"], str(item["id"]), item["image_path"]))
            if not fused:
                num_without_candidates += 1
            record = {
                "query_index": query_index,
                "query_id": query.get("id", ""),
                "query_image_id": query.get("image_id", ""),
                "query_image_path": query.get("image_path", ""),
                "query_title": query.get("title", ""),
                "query_ingredients": query.get("ingredients", ""),
                "query_instructions": query.get("instructions", ""),
                "mode": args.mode,
                "candidate_intersection_count": len(fused),
                "retrieved_top1": fused[:1],
            }
            if not first_record:
                out.write(",\n")
            first_record = False
            out.write(json.dumps(record, ensure_ascii=False, indent=2))
        out.write("\n]\n")

    print(f"Queries: {len(queries)}")
    print(f"Queries without fused candidates: {num_without_candidates}")
    if args.mode == "train-to-train":
        print(f"Self matches removed: {num_self_removed}")
    if args.mode == "train-to-train":
        print(f"Near-duplicate candidates removed (image similarity >= {args.max_image_similarity}): {num_near_duplicate_removed}")
    print(f"Saved JSON results to {output_path}")


if __name__ == "__main__":
    main()
