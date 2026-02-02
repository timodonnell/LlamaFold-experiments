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
├── exp1_distance_matrix/         # ~20M param model, 501 distance tokens (<d0>-<d500>)
├── exp1b_distance_matrix_1b/     # ~1.2B param model, same tokenization as exp1
├── exp1c_distance_matrix_decimal/ # ~1B param model, decimal tokenization (26 distance tokens)
├── exp2_residue_infilling/       # Predict missing residue coordinates (not yet implemented)
├── exp3_structure_prediction/    # Full structure from sequence + constraints (not yet implemented)
shared/                           # Shared utilities across experiments
```

## Key Design Decisions

- **Isolated experiments**: Each experiment is self-contained with its own data, model, and training code
- **On-the-fly data generation**: Datasets generate synthetic data dynamically rather than pre-computing
- **LLM-based approach**: Training small Llama models from scratch on text-formatted distance documents
- **Held-out evaluation**: Model trained on full documents, evaluated on predicting held-out pairs

### Distance Tokenization

- **exp1/exp1b**: Single token per distance: `<d0>` through `<d500>` (501 tokens)
- **exp1c**: Decimal format with 3 tokens per distance: `<d000> <d00> <d0>` through `<d500> <d90> <d9>` (26 tokens)
