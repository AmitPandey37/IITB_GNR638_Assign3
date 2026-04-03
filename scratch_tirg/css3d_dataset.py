"""CSS3D dataset loader for the scratch TIRG implementation."""

from __future__ import annotations

import os
import random
from typing import Dict, List

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


def build_css_transform() -> T.Compose:
    """Builds the image transform used for CSS3D experiments."""
    return T.Compose(
        [
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


class CSS3DTrainingDataset(Dataset):
    """Random-query training dataset for CSS3D."""

    def __init__(self, dataset_root: str, transform=None):
        self.dataset_root = dataset_root
        self.transform = transform or build_css_transform()
        self._data = np.load(
            os.path.join(dataset_root, "css_toy_dataset_novel2_small.dup.npy"),
            allow_pickle=True,
            encoding="latin1",
        ).item()
        self._split = self._data["train"]
        self._mods = self._split["mods"]
        self._images = self._split["objects_img"]
        self._imgid_to_modtarget: dict[int, list[tuple[int, int]]] = {
            image_idx: [] for image_idx in range(len(self._images))
        }
        for mod_idx, mod in enumerate(self._mods):
            for src_idx, tgt_idx in zip(mod["from"], mod["to"]):
                self._imgid_to_modtarget[int(src_idx)].append((mod_idx, int(tgt_idx)))
        self._last_from: int | None = None
        self._last_mods: list[int] = []

    def __len__(self) -> int:
        # Match the official loader's epoch length by sampling one query per image.
        return len(self._images)

    def __getitem__(self, index: int) -> Dict[str, object]:
        del index
        source_idx, mod_idx, target_idx = self._sample_query_target()
        return {
            "query_image": self._load_image(source_idx, split="train"),
            "target_image": self._load_image(target_idx, split="train"),
            "modification_text": str(self._mods[mod_idx]["to_str"]),
        }

    def all_texts(self) -> List[str]:
        return [str(mod["to_str"]) for mod in self._mods]

    def _sample_first_query(self) -> tuple[int, int, int]:
        mod_idx = random.randrange(len(self._mods))
        mod = self._mods[mod_idx]
        edge_idx = random.randrange(len(mod["from"]))
        self._last_from = int(mod["from"][edge_idx])
        self._last_mods = [mod_idx]
        return self._last_from, mod_idx, int(mod["to"][edge_idx])

    def _sample_second_query(self) -> tuple[int, int, int]:
        assert self._last_from is not None
        options = self._imgid_to_modtarget[self._last_from]
        mod_idx, target_idx = random.choice(options)
        while mod_idx in self._last_mods:
            mod_idx, target_idx = random.choice(options)
        self._last_mods.append(mod_idx)
        return self._last_from, mod_idx, target_idx

    def _sample_query_target(self) -> tuple[int, int, int]:
        try:
            if len(self._last_mods) < 2:
                return self._sample_second_query()
            return self._sample_first_query()
        except Exception:
            return self._sample_first_query()

    def _load_image(self, image_idx: int, split: str) -> torch.Tensor:
        image_path = os.path.join(
            self.dataset_root,
            "images",
            f"css_{split}_{int(image_idx):06d}.png",
        )
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            return self.transform(image)


class CSS3DEvaluationDataset:
    """Evaluation helper for CSS3D retrieval metrics."""

    def __init__(self, dataset_root: str, split: str, transform=None):
        if split not in {"train", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        self.dataset_root = dataset_root
        self.split = split
        self.transform = transform or build_css_transform()
        self._data = np.load(
            os.path.join(dataset_root, "css_toy_dataset_novel2_small.dup.npy"),
            allow_pickle=True,
            encoding="latin1",
        ).item()
        self._split = self._data[split]
        self._mods = self._split["mods"]
        self._images = self._split["objects_img"]
        self._labels = self._split.get("labels")
        self._captions = []
        for image_idx in range(len(self._images)):
            label = image_idx
            if self._labels is not None:
                label = self._labels[image_idx]
            self._captions.append(str(label))
        self._queries = []
        for mod in self._mods:
            for src_idx, tgt_idx in zip(mod["from"], mod["to"]):
                self._queries.append(
                    {
                        "source_idx": int(src_idx),
                        "target_idx": int(tgt_idx),
                        "target_caption": self._captions[int(tgt_idx)],
                        "text": str(mod["to_str"]),
                    }
                )

    def __len__(self) -> int:
        return len(self._images)

    @property
    def queries(self) -> List[Dict[str, object]]:
        return self._queries

    @property
    def target_captions(self) -> List[str]:
        return self._captions

    def load_image(self, image_idx: int) -> torch.Tensor:
        image_path = os.path.join(
            self.dataset_root,
            "images",
            f"css_{self.split}_{int(image_idx):06d}.png",
        )
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            return self.transform(image)


def training_collate(batch: List[Dict[str, object]]) -> Dict[str, object]:
    """Collates a batch of CSS3D query/target pairs."""
    return {
        "query_images": torch.stack([item["query_image"] for item in batch]),
        "target_images": torch.stack([item["target_image"] for item in batch]),
        "modification_texts": [item["modification_text"] for item in batch],
    }
