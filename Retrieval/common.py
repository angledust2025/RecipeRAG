"""Shared data loading, IR-CLIP feature extraction, and path helpers."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPImageProcessor, CLIPTokenizer

IR_CLIP_ROOT = Path(__file__).resolve().parents[1] / "IR-CLIP"
if str(IR_CLIP_ROOT) not in sys.path:
    sys.path.insert(0, str(IR_CLIP_ROOT))

from model.clip_strc.irclip import IRCLIPModel  # noqa: E402


def read_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    rows = json.loads(text)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON array or JSONL file: {path}")
    return rows


def resolve_path(raw_path: str, image_root: Path | None) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute() or image_root is None:
        return path
    return image_root / path


def sample_key(row: dict[str, Any]) -> str:
    if row.get("id"):
        return str(row["id"])
    if row.get("image_id"):
        return str(row["image_id"])
    return str(row.get("image_path", ""))


def same_sample(query: dict[str, Any], candidate: dict[str, Any]) -> bool:
    for key in ("id", "image_id"):
        if query.get(key) and candidate.get(key) and str(query[key]) == str(candidate[key]):
            return True
    q_path = str(query.get("image_path", ""))
    c_path = str(candidate.get("image_path", ""))
    return bool(q_path and c_path and Path(q_path).as_posix() == Path(c_path).as_posix())


class IRCLIPEncoder:
    def __init__(self, checkpoint: Path, device: str):
        self.device = torch.device(device)
        self.model = IRCLIPModel.from_pretrained(checkpoint).to(self.device).eval()
        self.tokenizer = CLIPTokenizer.from_pretrained(checkpoint)
        self.image_processor = CLIPImageProcessor.from_pretrained(checkpoint)

    @torch.inference_mode()
    def encode_images(self, rows: list[dict[str, Any]], image_root: Path | None,
                      batch_size: int) -> np.ndarray:
        all_features = []
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            images = []
            for offset, row in enumerate(batch, start=start):
                image_path = resolve_path(str(row["image_path"]), image_root)
                try:
                    with Image.open(image_path) as image:
                        # Match IR-CLIP's retrieval evaluator before applying the checkpoint processor.
                        images.append(image.convert("RGB").resize((224, 224)))
                except Exception as exc:
                    raise RuntimeError(f"Cannot read image for row {offset}: {image_path}") from exc
            pixels = self.image_processor(images=images, return_tensors="pt").pixel_values.to(self.device)
            features = F.normalize(self.model.get_image_features(pixel_values=pixels), dim=-1)
            all_features.append(features.float().cpu().numpy())
        if not all_features:
            raise ValueError("No image rows to encode")
        return np.ascontiguousarray(np.concatenate(all_features), dtype=np.float32)

    def _encode_texts(self, texts: list[str], mode: str, max_length: int,
                      walk_short_pos: bool, batch_size: int) -> np.ndarray:
        all_features = []
        for start in range(0, len(texts), batch_size):
            tokens = self.tokenizer(
                texts[start:start + batch_size],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).input_ids.to(self.device)
            features = self.model.get_text_features(
                input_ids=tokens, mode=mode, walk_short_pos=walk_short_pos
            )
            all_features.append(F.normalize(features, dim=-1).float().cpu().numpy())
        if not all_features:
            raise ValueError(f"No {mode} texts to encode")
        return np.concatenate(all_features).astype(np.float32, copy=False)

    @torch.inference_mode()
    def encode_recipes(self, rows: list[dict[str, Any]], batch_size: int) -> np.ndarray:
        titles = [str(row.get("title", "")) for row in rows]
        ingredients = [str(row.get("ingredients", "")) for row in rows]
        instructions = [str(row.get("instructions", "")) for row in rows]
        title_features = self._encode_texts(titles, "title", 77, True, batch_size)
        ingredient_features = self._encode_texts(ingredients, "ingredients", 77, True, batch_size)
        instruction_features = self._encode_texts(instructions, "instructions", 248, False, batch_size)
        global_features = (title_features + ingredient_features + instruction_features) / 3.0
        return np.ascontiguousarray(
            global_features / np.maximum(np.linalg.norm(global_features, axis=1, keepdims=True), 1e-12),
            dtype=np.float32,
        )
