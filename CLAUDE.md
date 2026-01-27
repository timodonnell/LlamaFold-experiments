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

# Train Experiment 1 (distance matrix completion)
uv run python -m experiments.exp1_distance_matrix.src.train --output-dir outputs/exp1

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
├── exp1_distance_matrix/    # Distance matrix completion from partial observations
│   ├── src/
│   │   ├── data.py          # DistanceMatrixDataset: generates 3D point clouds + distance matrices
│   │   ├── model.py         # DistanceMatrixTransformer: predicts missing distances
│   │   ├── reconstruct.py   # MDS + Procrustes for coordinate reconstruction
│   │   └── train.py         # Training loop with CLI
│   └── tests/
├── exp2_residue_infilling/  # Predict missing residue coordinates (not yet implemented)
├── exp3_structure_prediction/ # Full structure from sequence + constraints (not yet implemented)
shared/                      # Shared utilities across experiments
```

## Key Design Decisions

- **Isolated experiments**: Each experiment is self-contained with its own data, model, and training code
- **On-the-fly data generation**: Datasets generate synthetic data dynamically rather than pre-computing
- **Upper triangle optimization**: Distance matrices are symmetric, so models operate on upper triangle only
- **Masked prediction**: Models see observed distances and predict masked positions (BERT-style)
