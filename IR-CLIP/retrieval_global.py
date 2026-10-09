#!/usr/bin/env python3
"""Global image-to-recipe and recipe-to-image retrieval evaluation for IR-CLIP.

Accepts one paired JSON/JSONL file or a directory of sample_*.json files.
For a directory, reports each subset and the arithmetic mean over subsets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPTokenizer

from model.clip_strc.irclip import IRCLIPModel


def read_rows(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    rows = json.loads(text)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return rows


def resolve_image(raw_path: str, image_root: Path | None) -> Path:
    path = Path(raw_path)
    if image_root is None or path.is_absolute():
        return path
    return image_root / path


def recall_metrics(similarity: np.ndarray) -> dict[str, float]:
    n = similarity.shape[0]
    target = np.arange(n)
    # Each row in the paired evaluation list defines its positive image/recipe pair.
    i2r_rank = np.argsort(-similarity, axis=1).argsort(axis=1)[target, target] + 1
    r2i_rank = np.argsort(-similarity.T, axis=1).argsort(axis=1)[target, target] + 1
    metrics: dict[str, float] = {}
    for prefix, ranks in (("I2R", i2r_rank), ("R2I", r2i_rank)):
        metrics[f"{prefix}_MedR"] = float(np.median(ranks))
        for k in (1, 5, 10):
            metrics[f"{prefix}_R@{k}"] = float(np.mean(ranks <= k))
    return metrics


def encode_and_evaluate(
    rows: list[dict],
    model: IRCLIPModel,
    tokenizer: CLIPTokenizer,
    processor: CLIPImageProcessor,
    device: torch.device,
    image_root: Path | None,
    batch_size: int,
) -> dict[str, float]:
    required = ("image_path", "title", "ingredients", "instructions")
    for index, row in enumerate(rows):
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"Row {index} is missing: {', '.join(missing)}")

    image_features: list[torch.Tensor] = []
    text_features: list[torch.Tensor] = []
    for start in tqdm(range(0, len(rows), batch_size), desc="Encoding pairs", leave=False):
        batch = rows[start : start + batch_size]
        images = []
        for row in batch:
            path = resolve_image(str(row["image_path"]), image_root)
            with Image.open(path) as image:
                # Match retrieval_global.py's fixed 224px image input.
                images.append(image.convert("RGB").resize((224, 224)))
        pixels = processor(images=images, return_tensors="pt").pixel_values.to(device)

        with torch.inference_mode():
            image_embeds = F.normalize(model.get_image_features(pixels), dim=-1)
            text_branches = []
            for mode in ("title", "ingredients"):
                ids = tokenizer(
                    [str(row[mode]) for row in batch],
                    padding="max_length",
                    truncation=True,
                    max_length=77,
                    return_tensors="pt",
                ).input_ids.to(device)
                features = model.get_text_features(ids, walk_short_pos=True, mode=mode)
                text_branches.append(F.normalize(features, dim=-1))

            # This follows retrieval_global.py: instructions use the extended 248-token position path.
            ids = tokenizer(
                [str(row["instructions"]) for row in batch],
                padding="max_length",
                truncation=True,
                max_length=248,
                return_tensors="pt",
            ).input_ids.to(device)
            instruction_features = model.get_text_features(
                ids, walk_short_pos=False, mode="instructions"
            )
            text_branches.append(F.normalize(instruction_features, dim=-1))

            global_text = F.normalize(sum(text_branches) / 3.0, dim=-1)
        image_features.append(image_embeds.cpu())
        text_features.append(global_text.cpu())

    image_matrix = torch.cat(image_features).numpy()
    text_matrix = torch.cat(text_features).numpy()
    similarity = image_matrix @ text_matrix.T
    return recall_metrics(similarity)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path(__file__).parent / "checkpoints/base")
    parser.add_argument("--data", type=Path, required=True,
                        help="Paired JSON/JSONL file or directory containing sample_*.json files")
    parser.add_argument("--image-root", type=Path, default=None,
                        help="Prefix for relative image_path values")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    device = torch.device(args.device)
    model = IRCLIPModel.from_pretrained(args.checkpoint).to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained(args.checkpoint)
    processor = CLIPImageProcessor.from_pretrained(args.checkpoint)

    if args.data.is_dir():
        files = sorted(args.data.glob("sample_*.json"))
        if not files:
            files = sorted([*args.data.glob("*.json"), *args.data.glob("*.jsonl")])
    else:
        files = [args.data]
    if not files:
        raise FileNotFoundError(f"No evaluation JSON/JSONL files found in {args.data}")

    per_file: dict[str, dict[str, float]] = {}
    for path in files:
        rows = read_rows(path)
        if not rows:
            raise ValueError(f"No paired rows found in {path}")
        print(f"Evaluating {path.name}: {len(rows)} pairs", flush=True)
        per_file[path.name] = encode_and_evaluate(
            rows, model, tokenizer, processor, device, args.image_root, args.batch_size
        )
        print(json.dumps(per_file[path.name], indent=2), flush=True)

    mean_metrics = {
        key: float(np.mean([metrics[key] for metrics in per_file.values()]))
        for key in next(iter(per_file.values()))
    }
    print("Mean over subsets:")
    print(json.dumps(mean_metrics, indent=2))


if __name__ == "__main__":
    main()
