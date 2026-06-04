---
name: remove-watermarks-skill
description: Remove visible watermarks from authorized local image files by previewing a user- or agent-provided region and applying LaMA/iopaint inpainting. Use when the user asks to remove a watermark, logo, stamp, or overlay from a PNG, JPG, JPEG, or WEBP image and can identify the rough region such as bottom-right, top-left, center, or a bounding box.
---

# Remove Watermarks Skill

Use this skill to remove watermarks from local images the user owns or is authorized to edit. This skill is intentionally lightweight: it does not use Florence-2 or automatic semantic detection. The agent or user must provide the region to remove.

## Quick Start

Run commands from the skill directory.

First-time setup:

```bash
python scripts/setup_runtime.py --warmup
```

Preview a common corner watermark:

```bash
python scripts/remove_watermark_lite.py --input image.png --output clean.png --mode preview --region bottom-right
```

Remove after the preview region is acceptable:

```bash
python scripts/remove_watermark_lite.py --input image.png --output clean.png --mode remove --region bottom-right
```

Use a precise bounding box when the user gives coordinates or the agent has measured them:

```bash
python scripts/remove_watermark_lite.py --input image.png --output clean.png --mode remove --bbox 1200,850,1800,980
```

Use `python3` instead of `python` if that is the active Python command.

## Workflow

1. Confirm the image is local and the user is authorized to edit it.
2. Choose either `--region` or `--bbox`.
3. Run `preview` unless the region is already precise and low risk.
4. Inspect the JSON output and preview overlay path.
5. Run `remove` with the same region settings.
6. Report the final output image path, mask path, bbox, and any warnings.

## Region Selection

Supported regions:

```text
top-left, top-right, bottom-left, bottom-right, top, bottom, left, right, center
```

Defaults:

| Region type | Default size |
|---|---|
| Corner | width 28%, height 16% |
| Top/bottom edge | height 18%, full width |
| Left/right edge | width 18%, full height |
| Center | width 35%, height 22% |

Override region size with `--region-width` and `--region-height` as ratios from `0.01` to `1.0`. Add `--padding` to expand the mask by pixels, especially for semi-transparent watermark edges.

## Output

All script results are JSON. Successful preview includes:

```json
{
  "ok": true,
  "mode": "preview",
  "input": "image.png",
  "output": "clean.png",
  "bbox": [1200, 850, 1800, 980],
  "mask": "clean_mask.png",
  "preview": "clean_preview.png",
  "device": null,
  "warnings": []
}
```

Successful removal includes `output` and `mask`; `preview` is `null` unless the caller also ran preview.

## Runtime

`scripts/setup_runtime.py` creates `.venv` in the skill directory, installs the runtime, and downloads the LaMA model through iopaint.

Useful setup variants:

```bash
python scripts/setup_runtime.py --warmup --china-mirror
python scripts/setup_runtime.py --warmup --cuda cu124
```

If setup is not run, `preview` still works with only Pillow installed. `remove` requires PyTorch and iopaint with the LaMA model available.

## Failure Signals

| Signal | Action |
|---|---|
| `missing_region` | Ask the user for a region or bbox, or choose a conservative region from their description. |
| `unsupported_input` | Ask for PNG, JPG, JPEG, or WEBP. Do not process videos. |
| `output_exists` | Re-run with a different output path or `--overwrite`. |
| `same_input_output` | Choose a separate output path. Never overwrite the source image. |
| `runtime_missing` | Run `scripts/setup_runtime.py --warmup`. |
| poor repair quality | Re-run with a tighter bbox or adjusted `--padding`; avoid large center watermarks on structured content. |

## Quality Guidance

LaMA works best on small watermarks over natural backgrounds. It is weaker on faces, text, UI screenshots, product labels, grids, and large centered watermarks. Prefer a tight mask with a small padding value over a broad region.
