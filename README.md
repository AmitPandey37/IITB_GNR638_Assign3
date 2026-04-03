# IITB_GNR638_Assign3

This repository contains my Assignment 3 submission for GNR638 based on the paper:

**Composing Text and Image for Image Retrieval - An Empirical Odyssey**  
Nam Vo, Lu Jiang, Chen Sun, Kevin Murphy, Li-Jia Li, Li Fei-Fei, James Hays  
CVPR 2019

The assignment required:
- an independent implementation of the selected paper,
- running the paper method on a toy dataset or a small sample,
- and comparing the scratch implementation against the official GitHub implementation.

This repository contains both:
- the official GitHub implementation used as the baseline,
- and an independent PyTorch reimplementation for the same task.

## Repository Layout

- `main.py`, `datasets.py`, `img_text_composition_models.py`, `text_model.py`, `test_retrieval.py`, `torch_functions.py`
  Official GitHub implementation used as the baseline.

- `scratch_tirg/`
  Independent PyTorch reimplementation of TIRG.

- `scratch_tirg/report/assignment_report.tex`
  LaTeX source of the report.

- `scratch_tirg/report/assignment_report.pdf`
  Compiled report PDF.

- `results/`
  Compact experiment summaries used in the report.

- `images/`
  Figures from the original repository README.

## Paper Summary

The paper studies composed image retrieval. The query is not just text and not just an image. Instead, it is:
- a reference image,
- plus a short text instruction describing the desired modification.

The model must combine both into a single embedding and retrieve the target image that satisfies the modification. The proposed TIRG module does this using:
- a CNN image encoder,
- an LSTM text encoder,
- a gated branch,
- and a residual branch.

This paper fits the course constraints because it is a CNN/RNN model, not a transformer-based model.

## Dataset Used

The experiments in this repository use the **CSS3D** toy dataset.

CSS3D is a synthetic composed retrieval dataset where each image contains simple colored geometric objects. The text modifications describe operations such as:
- adding an object,
- removing an object,
- changing size,
- changing color,
- changing shape,
- or changing object position.

Examples:
- `add brown object`
- `make blue object small`
- `add large object to top-left`
- `make circle brown`

Dataset statistics used in the runs:
- train split: `19,012` images, `6,004` modification templates, `18,012` composed queries
- test split: `19,057` images, `6,019` modification templates, `18,057` composed queries

Expected dataset structure:

```text
CSSDataset/
├── css_toy_dataset_novel2_small.dup.npy
└── images/
    ├── css_train_000000.png
    ├── ...
    └── css_test_000000.png
```

CSS3D dataset link from the original TIRG repository:

- Original repository: `https://github.com/google/tirg`
- CSS3D download link used in that repository: `https://drive.google.com/file/d/1wPqMw-HKmXUG2qTgYBiTNUnjz83hA2tY/view?usp=sharing`

## Environment Setup

On shared Linux servers where system Python is externally managed, use a virtual environment:

```bash
cd /path/to/repo
python3 -m venv /path/to/venv
. /path/to/venv/bin/activate
pip install --upgrade pip
pip install torch torchvision numpy tqdm tensorboardX pillow scikit-image
```

## Running the Official Baseline

Train the official GitHub implementation on CSS3D:

```bash
cd /path/to/repo
. /path/to/venv/bin/activate

python3 main.py --dataset=css3d --dataset_path=/path/to/CSSDataset \
  --num_iters=160000 --model=tirg --loss=soft_triplet \
  --comment=css3d_tirg --loader_num_workers=4 --device=cuda
```

Run evaluation only from a saved official checkpoint:

```bash
python3 main.py --dataset=css3d --dataset_path=/path/to/CSSDataset \
  --model=tirg --loss=soft_triplet --device=cuda --eval_only \
  --checkpoint runs/<official_run_dir>/latest_checkpoint.pth \
  --result_json results/official_eval.json
```

## Running the Scratch Reimplementation

Train the scratch implementation:

```bash
cd /path/to/repo
. /path/to/venv/bin/activate

CUDA_VISIBLE_DEVICES=0 python -m scratch_tirg.train \
  --dataset-path /path/to/CSSDataset \
  --run-name css3d_tirg_scratch_v2_gpu0 \
  --num-iters 160000 \
  --batch-size 64 \
  --eval-batch-size 512 \
  --num-workers 8 \
  --eval-every-epochs 15 \
  --amp \
  --device cuda
```

Evaluate a saved scratch checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scratch_tirg.evaluate \
  --dataset-path /path/to/CSSDataset \
  --checkpoint scratch_tirg/outputs/<scratch_run_dir>/best.pt \
  --split test \
  --batch-size 512 \
  --amp \
  --device cuda \
  --output-json results/scratch_eval.json
```

## Running in Background on IITB Servers

For long jobs on shared servers, use `screen`:

```bash
screen -L -Logfile scratch_tirg/outputs/css3d_tirg_scratch.console.log \
  -S tirg-scratch
```

Inside the screen session, start training. Detach with:

```text
Ctrl+A D
```

Reattach later with:

```bash
screen -r tirg-scratch
```

## Comparing Official and Scratch Runs

The repository includes a comparison utility:

```bash
python -m scratch_tirg.compare_runs \
  --official-run-dir runs/<official_run_dir> \
  --scratch-run-dir scratch_tirg/outputs/<scratch_run_dir> \
  --output-json results/comparison.json \
  --latex-table scratch_tirg/report/comparison_table.tex
```

## Report

The final report files are:

- `scratch_tirg/report/assignment_report.tex`
- `scratch_tirg/report/assignment_report.pdf`

Compile the report with:

```bash
cd scratch_tirg/report
pdflatex assignment_report.tex
pdflatex assignment_report.tex
```

## Results Included in the Repository

The report compares the official GitHub implementation and the scratch implementation up to epoch `105`.

Current epoch-105 comparison summary:
- Official top-1 recall: `0.6976`
- Scratch top-1 recall: `0.5677`
- Gap: `-0.1299`

More detailed summaries are available in the `results/` directory.
