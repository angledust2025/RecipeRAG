#!/usr/bin/env python3
"""Generate SAM 3 foreground images and apply the IR-CLIP alpha-edge postprocess."""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_erosion, gaussian_filter
from tqdm import tqdm
from transformers import Sam3Model, Sam3Processor

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PRIMARY_PROMPT = "food"
FALLBACK_PROMPT = "drink"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True, help="Root directory containing source images")
    parser.add_argument("--output-root", type=Path, required=True, help="Root directory for generated PNG images")
    parser.add_argument("--model", default="facebook/sam3", help="Hugging Face model ID or local model directory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=0.60, help="SAM 3 instance confidence threshold")
    parser.add_argument("--mask-threshold", type=float, default=0.30, help="SAM 3 pixel mask threshold")
    parser.add_argument("--black-threshold", type=int, default=20, help="RGB threshold used to build the output alpha mask")
    parser.add_argument("--feather-sigma", type=float, default=1.0, help="Gaussian blur sigma for soft alpha edges; 0 disables feathering")
    parser.add_argument("--erosion-iterations", type=int, default=0, help="Optional binary mask erosion iterations")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing output images")
    return parser.parse_args()


def predict_mask(image, prompt, model, processor, device, threshold, mask_threshold):
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    result = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=inputs["original_sizes"].tolist(),
    )[0]
    masks = result.get("masks")
    if masks is None or len(masks) == 0:
        return None
    return torch.any(masks, dim=0).detach().cpu().numpy().astype(bool)


def make_foreground(image, mask):
    rgb = np.asarray(image.convert("RGB"))
    if mask is None:
        return rgb
    foreground = np.zeros_like(rgb)
    foreground[mask] = rgb[mask]
    return foreground


def add_feathered_alpha(rgb, args):
    data = np.zeros((*rgb.shape[:2], 4), dtype=np.uint8)
    data[..., :3] = rgb
    alpha_mask = np.max(rgb, axis=2) > args.black_threshold
    if args.erosion_iterations > 0:
        alpha_mask = binary_erosion(alpha_mask, iterations=args.erosion_iterations)
    if args.feather_sigma > 0:
        alpha = gaussian_filter(alpha_mask.astype(np.float32) * 255.0, sigma=args.feather_sigma)
        data[..., 3] = np.clip(alpha, 0, 255).astype(np.uint8)
    else:
        data[..., 3] = alpha_mask.astype(np.uint8) * 255
    return Image.fromarray(data, mode="RGBA")


def main():
    args = parse_args()
    if not args.input_root.is_dir():
        raise SystemExit(f"Input directory does not exist: {args.input_root}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")

    image_paths = sorted(
        path for path in args.input_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise SystemExit(f"No supported images found under {args.input_root}")

    model = Sam3Model.from_pretrained(args.model).to(args.device).eval()
    processor = Sam3Processor.from_pretrained(args.model)

    failures = []
    for image_path in tqdm(image_paths, desc="Generating foregrounds"):
        relative_path = image_path.relative_to(args.input_root).with_suffix(".png")
        output_path = args.output_root / relative_path
        if output_path.exists() and not args.overwrite:
            continue
        try:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            mask = predict_mask(image, PRIMARY_PROMPT, model, processor, args.device, args.threshold, args.mask_threshold)
            if mask is None:
                mask = predict_mask(image, FALLBACK_PROMPT, model, processor, args.device, args.threshold, args.mask_threshold)
            foreground = make_foreground(image, mask)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            add_feathered_alpha(foreground, args).save(output_path, format="PNG")
        except Exception as exc:
            failures.append((str(image_path), str(exc)))

    print(f"Images found: {len(image_paths)}; failed: {len(failures)}")
    for path, error in failures:
        print(f"[ERROR] {path}: {error}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
