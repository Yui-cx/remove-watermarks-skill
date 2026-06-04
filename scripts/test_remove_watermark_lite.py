#!/usr/bin/env python3
"""Tests for remove_watermark_lite.py that do not require LaMA."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

import remove_watermark_lite as lite


SCRIPT = Path(__file__).with_name("remove_watermark_lite.py")


class RegionTests(unittest.TestCase):
    def test_corner_regions(self) -> None:
        self.assertEqual(lite.bbox_from_region("bottom-right", (1000, 500)), (720, 420, 1000, 500))
        self.assertEqual(lite.bbox_from_region("top-left", (1000, 500)), (0, 0, 280, 80))

    def test_bottom_region(self) -> None:
        self.assertEqual(lite.bbox_from_region("bottom", (1000, 500)), (0, 410, 1000, 500))

    def test_center_region(self) -> None:
        self.assertEqual(lite.bbox_from_region("center", (1000, 500)), (325, 195, 675, 305))

    def test_clamps_bbox(self) -> None:
        self.assertEqual(lite.clamp_bbox((-10, -5, 120, 90), (100, 80)), (0, 0, 100, 80))

    def test_padding_clamps(self) -> None:
        self.assertEqual(lite.clamp_bbox((10, 10, 20, 20), (25, 25), padding=15), (0, 0, 25, 25))


class CommandTests(unittest.TestCase):
    def run_cmd(self, *args: str) -> tuple[int, dict]:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            text=True,
            capture_output=True,
            check=False,
        )
        return result.returncode, json.loads(result.stdout)

    def test_missing_region(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "image.png"
            Image.new("RGB", (20, 20), "white").save(src)
            code, payload = self.run_cmd("--input", str(src), "--output", str(Path(tmp) / "out.png"), "--mode", "preview")
            self.assertNotEqual(code, 0)
            self.assertEqual(payload["error"], "missing_region")

    def test_rejects_video_extension(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "video.mp4"
            src.write_bytes(b"not a real video")
            code, payload = self.run_cmd("--input", str(src), "--output", str(Path(tmp) / "out.png"), "--mode", "preview", "--region", "bottom-right")
            self.assertNotEqual(code, 0)
            self.assertEqual(payload["error"], "unsupported_input")

    def test_rejects_same_input_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "image.png"
            Image.new("RGB", (20, 20), "white").save(src)
            code, payload = self.run_cmd("--input", str(src), "--output", str(src), "--mode", "preview", "--region", "bottom-right")
            self.assertNotEqual(code, 0)
            self.assertEqual(payload["error"], "same_input_output")

    def test_preview_does_not_need_lama(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "image.png"
            out = Path(tmp) / "out.png"
            Image.new("RGB", (100, 50), "white").save(src)
            code, payload = self.run_cmd("--input", str(src), "--output", str(out), "--mode", "preview", "--region", "bottom-right")
            self.assertEqual(code, 0)
            self.assertTrue(payload["ok"])
            self.assertTrue(Path(payload["mask"]).exists())
            self.assertTrue(Path(payload["preview"]).exists())
            self.assertIsNone(payload["device"])


if __name__ == "__main__":
    unittest.main()
