"""Fine-tune IR-CLIP on image and recipe JSON pairs."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
import transformers
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor, CLIPTokenizer, HfArgumentParser

from model.clip_strc.irclip import IRCLIPModel
from train.irclip_trainer import IRCLIPTrainer


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "IR-CLIP base or large checkpoint directory"})
    base_model: Optional[str] = field(default=None, metadata={"help": "Tokenizer and image processor directory; defaults to model_name_or_path"})


@dataclass
class DataArguments:
    data_path: str = field(metadata={"help": "JSON array or JSONL with image_path, title, ingredients, instructions"})
    image_root: Optional[str] = None
    add_seg_loss: bool = False
    base_seq_length: int = 77
    instruction_seq_length: int = 77


@dataclass
class IRCLIPTrainingArguments(transformers.TrainingArguments):
    text_model_lr: Optional[float] = None
    from_openai: bool = False


def read_rows(path: Path):
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    rows = json.loads(text)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return rows


class RecipePairDataset(Dataset):
    def __init__(self, data_args: DataArguments, tokenizer: CLIPTokenizer,
                 processor: CLIPImageProcessor):
        if data_args.base_seq_length > 77 or data_args.instruction_seq_length > 77:
            raise ValueError("This IR-CLIP checkpoint uses the short text position table; sequence lengths must be <= 77")
        self.data_args = data_args
        self.rows = read_rows(Path(data_args.data_path))
        self.tokenizer = tokenizer
        self.processor = processor
        self.image_root = Path(data_args.image_root) if data_args.image_root else None

    def __len__(self):
        return len(self.rows)

    def _path(self, raw):
        path = Path(raw)
        if self.image_root is None:
            return path
        if not path.is_absolute():
            return self.image_root / path
        for split in ("train", "val", "test"):
            if split in path.parts:
                candidate = self.image_root.joinpath(*path.parts[path.parts.index(split):])
                if candidate.exists():
                    return candidate
        candidate = self.image_root / path.name
        return candidate if candidate.exists() else path

    def _tokens(self, text, length):
        return self.tokenizer(str(text or ""), max_length=length, padding="max_length",
                              truncation=True, return_tensors="pt").input_ids[0]

    def _image(self, raw):
        path = self._path(raw)
        with Image.open(path) as image:
            return self.processor(images=image.convert("RGB"), return_tensors="pt").pixel_values[0]

    def __getitem__(self, index):
        row = self.rows[index]
        required = ("image_path", "title", "ingredients", "instructions")
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"Row {index} is missing fields: {', '.join(missing)}")
        item = {
            "image": self._image(row["image_path"]),
            "title_texts": self._tokens(row["title"], self.data_args.base_seq_length),
            "ingredients_texts": self._tokens(row["ingredients"], self.data_args.base_seq_length),
            "instructions_texts": self._tokens(row["instructions"], self.data_args.instruction_seq_length),
            "add_seg_loss": self.data_args.add_seg_loss,
        }
        if self.data_args.add_seg_loss:
            seg_path = row.get("seg_image_path") or row.get("foreground_image_path")
            if not seg_path:
                raise ValueError(f"Row {index} needs seg_image_path when --add_seg_loss is enabled")
            item["seg_image"] = self._image(seg_path)
        return item


class RecipePairCollator:
    def __call__(self, rows: Sequence[Dict]):
        batch = {
            "image": torch.stack([row["image"] for row in rows]),
            "title_texts": torch.stack([row["title_texts"] for row in rows]),
            "ingredients_texts": torch.stack([row["ingredients_texts"] for row in rows]),
            "instructions_texts": torch.stack([row["instructions_texts"] for row in rows]),
            "add_seg_loss": bool(rows[0]["add_seg_loss"]),
        }
        if batch["add_seg_loss"]:
            batch["seg_image"] = torch.stack([row["seg_image"] for row in rows])
        return batch


def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, IRCLIPTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    base = model_args.base_model or model_args.model_name_or_path
    tokenizer = CLIPTokenizer.from_pretrained(base)
    processor = CLIPImageProcessor.from_pretrained(base)
    model = IRCLIPModel.from_pretrained(model_args.model_name_or_path)
    model.config.architectures = ["IRCLIPModel"]
    if training_args.from_openai:
        model.resize_postion_embeding()
        model.copy_weight()
    if training_args.fp16:
        raise ValueError("FP16 is not supported by this IR-CLIP training implementation; use --bf16 True or FP32")
    if training_args.bf16:
        model.to(dtype=torch.bfloat16)
    dataset = RecipePairDataset(data_args, tokenizer, processor)
    trainer = IRCLIPTrainer(model=model, args=training_args, train_dataset=dataset,
                            data_collator=RecipePairCollator())
    trainer.train(resume_from_checkpoint=True if list(Path(training_args.output_dir).glob("checkpoint-*")) else None)
    trainer.save_model(training_args.output_dir)
    trainer.save_state()
    tokenizer.save_pretrained(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()
