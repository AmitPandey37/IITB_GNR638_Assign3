"""Helpers to read scalar summaries from TensorBoard event files."""

from __future__ import annotations

import glob
import os
import struct
from collections import Counter, defaultdict
from typing import Dict

from tensorboardX.proto import event_pb2


def _read_events(path: str):
    with open(path, "rb") as handle:
        while True:
            header = handle.read(8)
            if not header or len(header) < 8:
                break
            length = struct.unpack("Q", header)[0]
            handle.read(4)
            data = handle.read(length)
            handle.read(4)
            event = event_pb2.Event()
            event.ParseFromString(data)
            yield event


def read_scalar_series(run_dir: str) -> Dict[str, Dict[int, float]]:
    """Reads scalar series from all TensorBoard event files in a run directory."""
    scalars = defaultdict(dict)
    pattern = os.path.join(run_dir, "events.out.tfevents.*")
    for path in sorted(glob.glob(pattern)):
        for event in _read_events(path):
            if not event.HasField("summary"):
                continue
            step = int(event.step)
            for value in event.summary.value:
                scalar = None
                if value.HasField("simple_value"):
                    scalar = float(value.simple_value)
                elif value.HasField("tensor"):
                    tensor = value.tensor
                    if tensor.float_val:
                        scalar = float(tensor.float_val[0])
                    elif tensor.double_val:
                        scalar = float(tensor.double_val[0])
                if scalar is not None:
                    scalars[value.tag][step] = scalar
    return scalars


def infer_steps_per_epoch(loss_steps) -> int:
    """Infers how many training steps correspond to one epoch."""
    diffs = [b - a for a, b in zip(loss_steps, loss_steps[1:])]
    return Counter(diffs).most_common(1)[0][0]

