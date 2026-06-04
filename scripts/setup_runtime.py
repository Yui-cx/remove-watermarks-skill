#!/usr/bin/env python3
"""Create the local runtime and warm up the LaMA model."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import venv
from pathlib import Path
from typing import Any


SKILL_DIR = Path(__file__).resolve().parents[1]
DEFAULT_VENV = SKILL_DIR / ".venv"


def json_result(**payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def python_executable(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, check=True, env=env)


def pip_args(china_mirror: bool) -> list[str]:
    if not china_mirror:
        return []
    return ["-i", "https://pypi.tuna.tsinghua.edu.cn/simple", "--trusted-host", "pypi.tuna.tsinghua.edu.cn"]


def install_runtime(py: Path, cuda: str | None, china_mirror: bool) -> None:
    mirror_args = pip_args(china_mirror)
    run([str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel", *mirror_args])

    torch_index = "https://download.pytorch.org/whl/cpu"
    if cuda:
        if cuda != "cu124":
            raise ValueError("Only --cuda cu124 is supported.")
        torch_index = "https://download.pytorch.org/whl/cu124"

    run([
        str(py),
        "-m",
        "pip",
        "install",
        "--upgrade",
        "torch",
        "torchvision",
        "--index-url",
        torch_index,
    ])
    run([
        str(py),
        "-m",
        "pip",
        "install",
        "--upgrade",
        "Pillow>=10.0.0",
        "opencv-python-headless>=4.8.0,<4.12.0",
        "numpy<2",
        "iopaint",
        "pydantic",
        "typer",
        "einops",
        "omegaconf",
        "easydict",
        "yacs",
        *mirror_args,
    ])


def warmup(py: Path, china_mirror: bool) -> None:
    env = os.environ.copy()
    if china_mirror:
        env["HF_ENDPOINT"] = "https://hf-mirror.com"
    run([str(py), "-m", "iopaint", "download", "--model", "lama"], env=env)
    run([str(py), "-c", "from iopaint.model_manager import ModelManager; ModelManager(name='lama', device='cpu'); print('OK')"], env=env)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, default=DEFAULT_VENV, help="Virtual environment directory.")
    parser.add_argument("--cuda", choices=["cu124"], help="Install CUDA PyTorch wheels.")
    parser.add_argument("--china-mirror", action="store_true", help="Use China PyPI and Hugging Face mirrors.")
    parser.add_argument("--warmup", action="store_true", help="Download and load the LaMA model.")
    parser.add_argument("--skip-install", action="store_true", help="Skip dependency installation.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    venv_dir = args.venv.resolve()
    py = python_executable(venv_dir)

    try:
        if not py.exists():
            venv.EnvBuilder(with_pip=True).create(venv_dir)

        if not args.skip_install:
            install_runtime(py, args.cuda, args.china_mirror)

        if args.warmup:
            warmup(py, args.china_mirror)
    except Exception as exc:
        print(json_result(ok=False, error="setup_failed", message=str(exc), venv=str(venv_dir)))
        return 1

    print(json_result(
        ok=True,
        status="ready" if args.warmup else "installed",
        venv=str(venv_dir),
        python=str(py),
        cuda=args.cuda,
        china_mirror=args.china_mirror,
        models={"inpainter": "lama"} if args.warmup else {},
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
