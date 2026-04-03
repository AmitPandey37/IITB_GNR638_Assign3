"""Shared utilities for the scratch TIRG implementation."""

from __future__ import annotations

import json
import os
import random
from datetime import datetime

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seeds Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str = "auto") -> torch.device:
    """Resolves the best available torch device."""
    if device_name and device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: str) -> None:
    """Creates a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def timestamp() -> str:
    """Returns a compact timestamp for run directories."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_json(path: str, payload: dict) -> None:
    """Writes JSON with deterministic formatting."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def append_jsonl(path: str, payload: dict) -> None:
    """Appends one JSON record per line."""
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")

