#!/usr/bin/env python3
"""Preview and remove image watermarks from a user-provided region."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
REGIONS = {
    "top-left",
    "top-right",
    "bottom-left",
    "bottom-right",
    "top",
    "bottom",
    "left",
    "right",
    "center",
}
CORNER_REGIONS = {"top-left", "top-right", "bottom-left", "bottom-right"}
EDGE_REGIONS = {"top", "bottom", "left", "right"}


def json_result(**payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def error_payload(code: str, message: str, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": False,
        "error": code,
        "message": message,
        "warnings": [],
    }
    payload.update(extra)
    return payload


def parse_bbox(value: str) -> tuple[int, int, int, int]:
    parts = value.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--bbox must be x1,y1,x2,y2")
    try:
        x1, y1, x2, y2 = [int(round(float(part.strip()))) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--bbox values must be numbers") from exc
    return x1, y1, x2, y2


def clamp_bbox(
    bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
    padding: int = 0,
) -> tuple[int, int, int, int]:
    width, height = image_size
    x1, y1, x2, y2 = bbox
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    x1 -= padding
    y1 -= padding
    x2 += padding
    y2 += padding

    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    return x1, y1, x2, y2


def default_region_size(region: str) -> tuple[float, float]:
    if region in CORNER_REGIONS:
        return 0.28, 0.16
    if region in {"top", "bottom"}:
        return 1.0, 0.18
    if region in {"left", "right"}:
        return 0.18, 1.0
    return 0.35, 0.22


def ratio_arg(value: str) -> float:
    try:
        ratio = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ratio must be a number from 0.01 to 1.0") from exc
    if ratio <= 0 or ratio > 1:
        raise argparse.ArgumentTypeError("ratio must be from 0.01 to 1.0")
    return ratio


def bbox_from_region(
    region: str,
    image_size: tuple[int, int],
    region_width: float | None = None,
    region_height: float | None = None,
    padding: int = 0,
) -> tuple[int, int, int, int]:
    width, height = image_size
    default_width, default_height = default_region_size(region)
    width_ratio = default_width if region_width is None else region_width
    height_ratio = default_height if region_height is None else region_height

    box_width = max(1, int(round(width * width_ratio)))
    box_height = max(1, int(round(height * height_ratio)))

    if region.endswith("left") or region == "left":
        x1 = 0
    elif region.endswith("right") or region == "right":
        x1 = width - box_width
    else:
        x1 = (width - box_width) // 2

    if region.startswith("top") or region == "top":
        y1 = 0
    elif region.startswith("bottom") or region == "bottom":
        y1 = height - box_height
    else:
        y1 = (height - box_height) // 2

    return clamp_bbox((x1, y1, x1 + box_width, y1 + box_height), image_size, padding)


def make_mask(image_size: tuple[int, int], bbox: tuple[int, int, int, int]):
    from PIL import Image, ImageDraw

    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle(bbox, fill=255)
    return mask


def save_mask(image_path: Path, bbox: tuple[int, int, int, int], mask_path: Path) -> None:
    from PIL import Image

    with Image.open(image_path) as image:
        mask = make_mask(image.size, bbox)
    mask.save(mask_path)


def make_preview(image_path: Path, bbox: tuple[int, int, int, int], mask_path: Path, preview_path: Path) -> None:
    from PIL import Image, ImageDraw

    image = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle(bbox, fill=(255, 0, 0, 85), outline=(255, 0, 0, 255), width=3)
    mask = make_mask(image.size, bbox)
    mask.save(mask_path)
    Image.alpha_composite(image, overlay).save(preview_path)


def output_paths(output_path: Path, save_mask: Path | None = None) -> tuple[Path, Path]:
    stem = output_path.with_suffix("")
    mask_path = save_mask if save_mask else stem.with_name(f"{stem.name}_mask.png")
    preview_path = stem.with_name(f"{stem.name}_preview.png")
    return mask_path, preview_path


def resolve_device(choice: str) -> str:
    if choice != "auto":
        return choice
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def remove_with_lama(image_path: Path, output_path: Path, mask_path: Path, device: str) -> None:
    try:
        import cv2
        import numpy as np
        from iopaint.model_manager import ModelManager
        from iopaint.schema import HDStrategy, LDMSampler, InpaintRequest as Config
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "runtime_missing: install dependencies with scripts/setup_runtime.py --warmup"
        ) from exc

    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    manager = ModelManager(name="lama", device=device)
    config = Config(
        ldm_steps=50,
        ldm_sampler=LDMSampler.ddim,
        hd_strategy=HDStrategy.CROP,
        hd_strategy_crop_margin=64,
        hd_strategy_crop_trigger_size=800,
        hd_strategy_resize_limit=1600,
    )

    result = manager(np.array(image), np.array(mask), config)
    if result.dtype in [np.float64, np.float32]:
        result = np.clip(result, 0, 255).astype(np.uint8)
    Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)).save(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, dest="input_path", help="Input image path.")
    parser.add_argument("--output", required=True, dest="output_path", help="Output image path.")
    parser.add_argument("--mode", required=True, choices=["preview", "remove"], help="Operation mode.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--bbox", type=parse_bbox, help="Watermark bbox as x1,y1,x2,y2.")
    group.add_argument("--region", choices=sorted(REGIONS), help="Named watermark region.")
    parser.add_argument("--region-width", type=ratio_arg, help="Region width ratio, 0.01 to 1.0.")
    parser.add_argument("--region-height", type=ratio_arg, help="Region height ratio, 0.01 to 1.0.")
    parser.add_argument("--padding", type=int, default=0, help="Expand bbox by this many pixels.")
    parser.add_argument("--save-mask", type=Path, help="Optional mask output path.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Inpaint device.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    warnings: list[str] = []

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    if not args.bbox and not args.region:
        print(json_result(**error_payload("missing_region", "Provide either --bbox or --region.")))
        return 2

    if not input_path.exists():
        print(json_result(**error_payload("missing_input", "Input file does not exist.", input=str(input_path))))
        return 2

    if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        print(json_result(**error_payload("unsupported_input", "Only PNG, JPG, JPEG, and WEBP images are supported.", input=str(input_path))))
        return 2

    if input_path.resolve() == output_path.resolve():
        print(json_result(**error_payload("same_input_output", "Output path must not equal input path.", input=str(input_path), output=str(output_path))))
        return 2

    if output_path.exists() and not args.overwrite:
        print(json_result(**error_payload("output_exists", "Output exists. Use --overwrite or choose another path.", output=str(output_path))))
        return 2

    try:
        from PIL import Image

        with Image.open(input_path) as image:
            image_size = image.size
    except ImportError:
        print(json_result(**error_payload("runtime_missing", "Pillow is required. Run scripts/setup_runtime.py --warmup.")))
        return 3
    except Exception as exc:
        print(json_result(**error_payload("invalid_image", f"Could not open image: {exc}", input=str(input_path))))
        return 2

    padding = max(0, args.padding)
    if args.bbox:
        bbox = clamp_bbox(args.bbox, image_size, padding)
    else:
        bbox = bbox_from_region(args.region, image_size, args.region_width, args.region_height, padding)

    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        print(json_result(**error_payload("empty_bbox", "Computed bbox is empty.", bbox=list(bbox))))
        return 2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path, preview_path = output_paths(output_path, args.save_mask)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "preview" and preview_path.exists() and not args.overwrite:
        print(json_result(**error_payload("output_exists", "Preview exists. Use --overwrite or choose another output path.", preview=str(preview_path))))
        return 2

    if mask_path.exists() and not args.overwrite:
        print(json_result(**error_payload("output_exists", "Mask exists. Use --overwrite or choose another output path.", mask=str(mask_path))))
        return 2

    try:
        device: str | None = None
        preview: str | None = str(preview_path)
        if args.mode == "preview":
            make_preview(input_path, bbox, mask_path, preview_path)
        else:
            save_mask(input_path, bbox, mask_path)
            device = resolve_device(args.device)
            remove_with_lama(input_path, output_path, mask_path, device)
            preview = None
    except RuntimeError as exc:
        code = "runtime_missing" if str(exc).startswith("runtime_missing:") else "processing_failed"
        message = str(exc).replace("runtime_missing: ", "")
        print(json_result(**error_payload(code, message)))
        return 3
    except Exception as exc:
        print(json_result(**error_payload("processing_failed", f"Processing failed: {exc}")))
        return 3

    payload = {
        "ok": True,
        "mode": args.mode,
        "input": str(input_path),
        "output": str(output_path),
        "bbox": list(bbox),
        "mask": str(mask_path),
        "preview": preview,
        "device": device,
        "warnings": warnings,
    }
    print(json_result(**payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
