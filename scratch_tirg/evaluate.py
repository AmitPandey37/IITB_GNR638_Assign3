"""Evaluation utilities and CLI for the scratch TIRG implementation."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from .css3d_dataset import CSS3DEvaluationDataset
from .model import TIRGRetrievalModel, Vocabulary
from .utils import resolve_device, save_json


def _batched_image_features(
    model: TIRGRetrievalModel,
    dataset: CSS3DEvaluationDataset,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
) -> np.ndarray:
    all_features = []
    images = []
    amp_enabled = bool(use_amp and device.type == "cuda")
    for image_idx in tqdm(range(len(dataset)), desc=f"{dataset.split}: encode targets"):
        images.append(dataset.load_image(image_idx))
        if len(images) >= batch_size or image_idx == len(dataset) - 1:
            batch = torch.stack(images).to(device, non_blocking=device.type == "cuda")
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                features = torch.nn.functional.normalize(model.encode_target_images(batch), dim=1)
            features = features.cpu().numpy()
            all_features.append(features)
            images = []
    return np.concatenate(all_features, axis=0)


def _batched_query_features(
    model: TIRGRetrievalModel,
    dataset: CSS3DEvaluationDataset,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
) -> tuple[np.ndarray, List[int]]:
    all_features = []
    source_indices = []
    images = []
    texts = []
    amp_enabled = bool(use_amp and device.type == "cuda")
    for query in tqdm(dataset.queries, desc=f"{dataset.split}: encode queries"):
        images.append(dataset.load_image(query["source_idx"]))
        texts.append(query["text"])
        source_indices.append(query["source_idx"])
        if len(images) >= batch_size or query is dataset.queries[-1]:
            batch = torch.stack(images).to(device, non_blocking=device.type == "cuda")
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                features = torch.nn.functional.normalize(model.compose_query(batch, texts), dim=1)
            features = features.cpu().numpy()
            all_features.append(features)
            images = []
            texts = []
    return np.concatenate(all_features, axis=0), source_indices


@torch.no_grad()
def compute_css3d_metrics(
    model: TIRGRetrievalModel,
    dataset: CSS3DEvaluationDataset,
    batch_size: int,
    device: torch.device,
    use_amp: bool = False,
) -> Dict[str, float]:
    """Computes CSS3D retrieval metrics."""
    model.eval()
    target_features = _batched_image_features(model, dataset, batch_size, device, use_amp)
    query_features, source_indices = _batched_query_features(
        model,
        dataset,
        batch_size,
        device,
        use_amp,
    )
    target_captions = dataset.target_captions
    query_target_captions = [query["target_caption"] for query in dataset.queries]
    ks = [1, 5, 10, 50, 100]
    max_k = max(ks)
    correct_counts = {k: 0 for k in ks}

    target_tensor = torch.from_numpy(target_features).to(device)
    caption_lookup = np.asarray(target_captions)
    chunk_size = max(1, min(batch_size * 8, len(query_features)))
    for start in tqdm(range(0, len(query_features), chunk_size), desc=f"{dataset.split}: retrieval"):
        end = min(start + chunk_size, len(query_features))
        query_tensor = torch.from_numpy(query_features[start:end]).to(device)
        scores = query_tensor @ target_tensor.transpose(0, 1)
        local_source_indices = torch.tensor(source_indices[start:end], device=device)
        scores[torch.arange(end - start, device=device), local_source_indices] = -1e10
        top_indices = torch.topk(scores, k=max_k, dim=1).indices.cpu().numpy()
        for row_idx in range(end - start):
            ranking_captions = caption_lookup[top_indices[row_idx]]
            target_caption = query_target_captions[start + row_idx]
            for k in ks:
                if target_caption in ranking_captions[:k]:
                    correct_counts[k] += 1

    recalls = {
        f"recall_top{k}_correct_composition": correct_counts[k] / len(query_features)
        for k in ks
    }
    return recalls


def load_scratch_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[TIRGRetrievalModel, dict]:
    """Loads a scratch checkpoint and rebuilds the model."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    vocab = Vocabulary(tokens=checkpoint["vocab_tokens"])
    model = TIRGRetrievalModel(
        vocab=vocab,
        embed_dim=checkpoint["config"]["embed_dim"],
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    return model, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the scratch TIRG implementation.")
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device)
    model, checkpoint = load_scratch_checkpoint(args.checkpoint, device)
    dataset = CSS3DEvaluationDataset(args.dataset_path, split=args.split)
    metrics = compute_css3d_metrics(model, dataset, args.batch_size, device, use_amp=args.amp)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if args.output_json:
        payload = {
            "checkpoint": args.checkpoint,
            "split": args.split,
            "metrics": metrics,
            "epoch": checkpoint.get("epoch"),
            "iteration": checkpoint.get("iteration"),
        }
        save_json(args.output_json, payload)
        print(f"Saved metrics to {args.output_json}")


if __name__ == "__main__":
    main()
