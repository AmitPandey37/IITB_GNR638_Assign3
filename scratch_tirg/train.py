"""Training entry point for the scratch TIRG implementation."""

from __future__ import annotations

import argparse
import math
import os
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .compare_runs import METRICS
from .css3d_dataset import (
    CSS3DEvaluationDataset,
    CSS3DTrainingDataset,
    training_collate,
)
from .evaluate import compute_css3d_metrics
from .model import TIRGRetrievalModel, Vocabulary
from .utils import append_jsonl, ensure_dir, resolve_device, save_json, seed_everything, timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scratch CSS3D TIRG trainer.")
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--run-root", type=str, default="scratch_tirg/outputs")
    parser.add_argument("--run-name", type=str, default="css3d_tirg_scratch")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--num-iters", type=int, default=160000)
    parser.add_argument("--eval-every-epochs", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--eval-train-split", action="store_true")
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def create_optimizer(model: TIRGRetrievalModel, learning_rate: float, weight_decay: float):
    return torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=weight_decay,
    )


def save_checkpoint(
    path: str,
    model: TIRGRetrievalModel,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    run_dir: str,
    epoch: int,
    iteration: int,
    best_test_top1: float,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "iteration": iteration,
            "best_test_top1": best_test_top1,
            "config": vars(args),
            "run_dir": run_dir,
            "vocab_tokens": model.vocab.tokens,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(
    checkpoint_path: str,
    model: TIRGRetrievalModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    model.to(device)
    return checkpoint


@torch.no_grad()
def maybe_run_eval(
    model: TIRGRetrievalModel,
    train_eval_dataset: CSS3DEvaluationDataset | None,
    test_eval_dataset: CSS3DEvaluationDataset,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    payload = {
        "test_metrics": compute_css3d_metrics(
            model,
            test_eval_dataset,
            args.eval_batch_size,
            device,
            use_amp=args.amp,
        )
    }
    if train_eval_dataset is not None:
        payload["train_metrics"] = compute_css3d_metrics(
            model,
            train_eval_dataset,
            args.eval_batch_size,
            device,
            use_amp=args.amp,
        )
    return payload


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    train_dataset = CSS3DTrainingDataset(args.dataset_path)
    test_dataset = CSS3DEvaluationDataset(args.dataset_path, split="test")
    train_eval_dataset = (
        CSS3DEvaluationDataset(args.dataset_path, split="train")
        if args.eval_train_split
        else None
    )

    vocab = Vocabulary.from_texts(train_dataset.all_texts() + [query["text"] for query in test_dataset.queries])
    model = TIRGRetrievalModel(
        vocab=vocab,
        embed_dim=args.embed_dim,
        pretrained=not args.no_pretrained,
    ).to(device)
    optimizer = create_optimizer(model, args.learning_rate, args.weight_decay)
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    if args.checkpoint:
        checkpoint = load_checkpoint(args.checkpoint, model, optimizer, device)
        run_dir = checkpoint.get("run_dir", os.path.dirname(args.checkpoint))
        start_epoch = checkpoint.get("epoch", -1) + 1
        iteration = checkpoint.get("iteration", 0)
        best_test_top1 = checkpoint.get("best_test_top1", float("-inf"))
    else:
        run_dir = os.path.join(args.run_root, f"{timestamp()}_{args.run_name}")
        ensure_dir(run_dir)
        save_json(os.path.join(run_dir, "config.json"), vars(args))
        start_epoch = 0
        iteration = 0
        best_test_top1 = float("-inf")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=training_collate,
    )
    steps_per_epoch = len(train_loader)
    total_epochs = math.ceil(args.num_iters / steps_per_epoch)

    latest_checkpoint = os.path.join(run_dir, "latest.pt")
    best_checkpoint = os.path.join(run_dir, "best.pt")
    history_path = os.path.join(run_dir, "metrics_history.jsonl")

    print(f"Run directory: {run_dir}")
    print(f"Device: {device}")
    print(f"AMP enabled: {amp_enabled}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total epochs: {total_epochs}")

    for epoch in range(start_epoch, total_epochs):
        model.train()
        running_losses = []
        progress = tqdm(train_loader, desc=f"epoch {epoch}")
        for batch in progress:
            query_images = batch["query_images"].to(device, non_blocking=device.type == "cuda")
            target_images = batch["target_images"].to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad()
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                loss = model.training_loss(query_images, batch["modification_texts"], target_images)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_losses.append(float(loss.detach().cpu()))
            progress.set_postfix(loss=f"{running_losses[-1]:.5f}")
            iteration += 1

        avg_loss = sum(running_losses) / len(running_losses)
        save_checkpoint(
            latest_checkpoint,
            model,
            optimizer,
            args,
            run_dir,
            epoch,
            iteration,
            best_test_top1,
        )
        record = {
            "epoch": epoch,
            "iteration": iteration,
            "train_loss": avg_loss,
        }
        print(f"Epoch {epoch}: avg soft_triplet={avg_loss:.6f}")

        should_eval = (epoch % args.eval_every_epochs == 0) or epoch == total_epochs - 1
        if should_eval:
            metrics_payload = maybe_run_eval(model, train_eval_dataset, test_dataset, args, device)
            record.update(metrics_payload)
            for split_name, split_metrics in metrics_payload.items():
                print(split_name)
                for metric_name in METRICS:
                    if metric_name in split_metrics:
                        print(f"  {metric_name}: {split_metrics[metric_name]:.4f}")
            current_test_top1 = metrics_payload["test_metrics"]["recall_top1_correct_composition"]
            if current_test_top1 > best_test_top1:
                best_test_top1 = current_test_top1
                save_checkpoint(
                    best_checkpoint,
                    model,
                    optimizer,
                    args,
                    run_dir,
                    epoch,
                    iteration,
                    best_test_top1,
                )
                save_json(
                    os.path.join(run_dir, "best_metrics.json"),
                    {
                        "epoch": epoch,
                        "iteration": iteration,
                        "test_metrics": metrics_payload["test_metrics"],
                        "train_metrics": metrics_payload.get("train_metrics", {}),
                        "train_loss": avg_loss,
                    },
                )
        append_jsonl(history_path, record)

    print(f"Finished training. Latest checkpoint: {latest_checkpoint}")
    print(f"Best checkpoint: {best_checkpoint}")


if __name__ == "__main__":
    main()
