# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project developing LLM-based agents for protein structure prediction. The approach uses transformers to search structure space guided by scoring functions (ProteinEBM, AF2Rank), enabling single-sequence structure prediction without MSAs.

## Commands

```bash
# Install dependencies
uv sync --all-extras

# Run all tests
uv run pytest

# Run tests for a specific experiment
uv run pytest experiments/exp1_distance_matrix/tests/ -v

# Run a single test
uv run pytest experiments/exp1_distance_matrix/tests/test_model.py::TestDistanceMatrixTransformer::test_output_shape -v

# Train Experiment 1 (LLM distance prediction)
uv run python -m experiments.exp1_distance_matrix.src.train --output-dir outputs/exp1

# Train with smaller dataset for testing
uv run python -m experiments.exp1_distance_matrix.src.train --train-samples 100 --val-samples 20 --eval-samples 10 --n-epochs 1 --no-wandb

# Train Experiment 2a (SS prediction from coordinates)
uv run python -m experiments.exp2a_secondary_structure.src.train --data-dir data/exp2a --output-dir outputs/exp2a

# Train Experiment 3 (SS prediction with bio2token)
uv run python -m experiments.exp3_ss_bio2token.src.train --data-dir data/exp3 --output-dir outputs/exp3

# Train Experiment 4 (contact prediction, 8 GPUs)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run torchrun --nproc_per_node=8 \
-m experiments.exp4_contact_prediction.src.train \
--n-epochs 10000 --batch-size 1 --gradient-accumulation-steps 1 --output-dir outputs/exp4

# Lint
uv run ruff check .

# Format
uv run ruff format .

# Type check
uv run mypy experiments/ shared/
```

## Architecture

```
experiments/
├── exp1_distance_matrix/           # ~20M param model, 501 distance tokens (<d0>-<d500>)
├── exp1b_distance_matrix_1b/       # ~1.2B param model, same tokenization as exp1
├── exp1c_distance_matrix_decimal/  # ~1B param model, decimal tokenization (26 distance tokens)
├── exp1d_distance_matrix_addition/ # ~1B param model, decimal tokenization + 10% addition problems
├── exp2a_secondary_structure/      # ~1B param model, 3-state SS prediction (H/E/C) from CA coords
├── exp2_residue_infilling/         # Predict missing residue coordinates (not yet implemented)
├── exp3_ss_bio2token/              # ~1B param model, SS prediction using bio2token FSQ coordinate tokens
├── exp3_structure_prediction/      # Full structure from sequence + constraints (not yet implemented)
├── exp4_contact_prediction/        # ~1.2B param model, contact prediction from HuggingFace protein-docs
shared/                             # Shared utilities across experiments
scripts/                            # Preprocessing scripts (e.g., bio2token_preprocess.py)
```

### Experiment Directory Pattern

Each experiment follows a consistent structure:
```
exp_name/
├── src/
│   ├── train.py      # Training entry point: create_tokenizer(), create_model(), train()
│   └── data.py       # Dataset classes + data generation/loading
└── tests/
    └── test_*.py     # pytest tests
```

Training scripts are run as modules: `uv run python -m experiments.<exp_name>.src.train`

## Key Design Decisions

- **Isolated experiments**: Each experiment is self-contained with its own data, model, and training code
- **On-the-fly data generation**: Datasets generate synthetic data dynamically rather than pre-computing
- **LLM-based approach**: Training small Llama models from scratch on text-formatted distance documents
- **Held-out evaluation**: Model trained on full documents, evaluated on predicting held-out pairs

### Tokenization & Coordinate Encoding

- **exp1/exp1b**: Single token per distance: `<d0>` through `<d500>` (501 tokens)
- **exp1c/exp1d**: Decimal format with 3 tokens per distance: `<d000> <d00> <d0>` through `<d500> <d90> <d9>` (26 tokens)
- **exp2a**: Naive coordinate binning at 1Å resolution: `<c-200>` to `<c200>` (401 tokens per axis)
- **exp3_ss_bio2token**: bio2token FSQ codebook (4096 tokens), one token per backbone atom — more compact than exp2a
- **exp4**: Pre-formatted protein documents from HuggingFace (`timodonnell/protein-docs`) with position + atom contact tokens

### Data Strategy

- **Synthetic experiments** (exp1 variants): Generate data on-the-fly with seed control for reproducibility
- **Real protein experiments** (exp2a, exp3): Lazy-loaded from preprocessed JSONL files (CIF → JSONL via BioPython/DSSP)
- **HuggingFace datasets** (exp4): Loaded via `datasets` library from `timodonnell/protein-docs`
- **Loss masking** (exp2a, exp3): Input tokens (sequence + coordinates) are masked; loss computed only on prediction targets (SS labels)
- **Full causal LM** (exp4): Standard next-token prediction on the entire document (no masking)
