"""Small shared helpers: seeding, devices, precision, environment capture, JSON IO."""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger("tinylm")

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def setup_logging(level: int = logging.INFO) -> None:
    """Configure a single, timestamped console handler for CLI scripts."""
    # Windows consoles default to a legacy code page that cannot print emoji or
    # curly quotes from generated text; force UTF-8 for script output.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device={device!r} requested but CUDA is not available.")
    return torch.device(device)


def resolve_precision(precision: str, device: torch.device) -> str:
    """Turn ``auto`` into a concrete precision and reject unsupported combinations.

    ``auto`` picks bf16 on GPUs that support it (Ampere and newer), fp16 on older
    GPUs such as the T4 (Turing has no bf16), and fp32 on CPU.
    """
    if precision not in {"auto", *DTYPES}:
        raise ValueError(f"precision must be one of auto/fp32/fp16/bf16, got {precision!r}")
    if device.type != "cuda":
        if precision in {"fp16"}:
            raise ValueError(
                "fp16 autocast is only supported here on CUDA; use fp32 or bf16 on CPU."
            )
        return "fp32" if precision == "auto" else precision
    if precision == "auto":
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError(f"{torch.cuda.get_device_name()} does not support bf16; use fp16.")
    return precision


def autocast_context(device: torch.device, precision: str):
    """Mixed-precision autocast context (a no-op for fp32)."""
    if precision == "fp32":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=DTYPES[precision])


def hardware_name(device: torch.device | None = None) -> str:
    """Human-readable accelerator name used to key benchmark records."""
    device = device or resolve_device()
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return f"CPU ({cpu_model()}, {os.cpu_count()} threads)"


def cpu_model() -> str:
    """CPU brand string (platform.processor() is uninformative on Windows and Linux)."""
    try:
        if sys.platform == "win32":
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        if sys.platform == "darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            )
            return out.stdout.strip()
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return platform.processor() or platform.machine()


def git_commit(short: bool = False) -> str | None:
    """Current commit hash, with a ``-dirty`` suffix for uncommitted changes."""
    try:
        args = ["git", "rev-parse", "--short" if short else "--verify", "HEAD"]
        sha = subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{sha}-dirty" if dirty else sha


def environment_info(device: torch.device | None = None) -> dict[str, Any]:
    """Library versions and hardware, written next to every run and benchmark."""
    import datasets
    import tokenizers
    import transformers

    device = device or resolve_device()
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "datasets": datasets.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "hardware": hardware_name(device),
        "cpu_count": os.cpu_count(),
        "git_commit": git_commit(),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info.update(
            cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(),
            gpu_memory_gb=round(props.total_memory / 1024**3, 2),
            compute_capability=f"{props.major}.{props.minor}",
            bf16_supported=torch.cuda.is_bf16_supported(),
        )
    return info


def write_json(path: str | Path, obj: Any) -> None:
    """Write JSON atomically (temp file + rename) so a crash never leaves half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_json_default) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_jsonl(path: str | Path, records: list[dict] | dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(records, dict):
        records = [records]
    with path.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, default=_json_default) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")
