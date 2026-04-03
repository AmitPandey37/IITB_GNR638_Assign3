# Scratch TIRG Reimplementation

This folder contains an independent PyTorch reimplementation of the TIRG model
for the CSS3D toy dataset. It is intentionally separate from the official
`google/tirg` code at the repository root so the assignment can distinguish:

- the official implementation
- the scratch reimplementation
- the comparison between the two

## Folder Layout

- `css3d_dataset.py`: CSS3D data pipeline written independently for this folder
- `model.py`: ResNet18 + LSTM + TIRG composition model
- `train.py`: training script with checkpointing and JSONL metrics logging
- `evaluate.py`: evaluation script for a saved scratch checkpoint
- `event_reader.py`: TensorBoard event parser for the official implementation
- `compare_runs.py`: compares official and scratch runs and writes JSON/LaTeX
- `report/assignment_report.tex`: LaTeX report template

## Important

The current scratch model definition is more faithful to the official TIRG
pipeline than the earlier scratch checkpoints. Older scratch checkpoints should
not be resumed after these changes. Start a fresh scratch run.

## Recommended GPU Run

Train the scratch reimplementation on the same CSS3D toy dataset used for the
official run:

```bash
cd /mnt/nas/amitpandey/main/projects
. /mnt/nas/amitpandey/venvs/tirg-py311-cu118/bin/activate

CUDA_VISIBLE_DEVICES=2 python -m scratch_tirg.train \
  --dataset-path /mnt/nas/amitpandey/CSSDataset \
  --run-name css3d_tirg_scratch_v2_gpu2 \
  --num-iters 160000 \
  --batch-size 64 \
  --eval-batch-size 512 \
  --num-workers 8 \
  --eval-every-epochs 10 \
  --amp \
  --device cuda
```

This creates a run directory under `scratch_tirg/outputs/`.

## Resume Training

```bash
cd /mnt/nas/amitpandey/main/projects
. /mnt/nas/amitpandey/venvs/tirg-py311-cu118/bin/activate

CUDA_VISIBLE_DEVICES=2 python -m scratch_tirg.train \
  --dataset-path /mnt/nas/amitpandey/CSSDataset \
  --checkpoint scratch_tirg/outputs/<run_dir>/latest.pt \
  --num-workers 4 \
  --device cuda
```

## Evaluate a Scratch Checkpoint

```bash
cd /mnt/nas/amitpandey/main/projects
. /mnt/nas/amitpandey/venvs/tirg-py311-cu118/bin/activate

CUDA_VISIBLE_DEVICES=2 python -m scratch_tirg.evaluate \
  --dataset-path /mnt/nas/amitpandey/CSSDataset \
  --checkpoint scratch_tirg/outputs/<run_dir>/best.pt \
  --split test \
  --batch-size 512 \
  --amp \
  --device cuda \
  --output-json scratch_tirg/outputs/<run_dir>/eval_test.json
```

## Compare Official vs Scratch

The official run can be compared directly against the scratch run:

```bash
cd /mnt/nas/amitpandey/main/projects
. /mnt/nas/amitpandey/venvs/tirg-py311-cu118/bin/activate

python -m scratch_tirg.compare_runs \
  --official-run-dir runs/Apr02_18-06-06_catcss3d_tirg_cat_gpu2 \
  --scratch-run-dir scratch_tirg/outputs/<run_dir> \
  --output-json scratch_tirg/report/comparison.json \
  --latex-table scratch_tirg/report/comparison_table.tex
```

## Compile the Report

```bash
cd /mnt/nas/amitpandey/main/projects/scratch_tirg/report
pdflatex assignment_report.tex
pdflatex assignment_report.tex
```
