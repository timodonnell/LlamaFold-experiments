# LLM Protein Experiments

Experiments in training Llama models from scratch on text-encoded protein structures. Each experiment explores a different way of representing 3D structural information as token sequences for causal language modeling.

## Experiments

| Experiment | Model | Task | Data |
|-----------|-------|------|------|
| **exp1** | ~20M params | Distance matrix prediction | Synthetic, 501 distance tokens (`<d0>`-`<d500>`) |
| **exp1b** | ~1.2B params | Distance matrix prediction | Same as exp1, larger model |
| **exp1c** | ~1B params | Distance matrix prediction | Decimal tokenization (26 tokens, 3 per distance) |
| **exp1d** | ~1B params | Distance matrix prediction | Decimal tokenization + 10% addition problems |
| **exp2a** | ~1B params | Secondary structure prediction | Real proteins (CIF), coordinate binning at 1A resolution |
| **exp3** | ~1B params | Secondary structure prediction | Real proteins, bio2token FSQ codebook (4096 tokens) |
| **exp4** | ~1B params | Contact prediction | ~24M AlphaFold structures, deterministic-positives-only |
| **exp5** | ~1B / ~2B params | Contact prediction | ~1.6M-5.3M AlphaFold structures, random-3-bins |

### Exp4: Deterministic Contact Prediction

Trains on protein documents with amino acid sequences and atomic contacts. Each contact is 4 tokens: two residue positions + two atom names (closest heavy-atom pair within 4.0A). Contacts sorted by decreasing sequence separation. One contact per residue pair, only true positives.

```
<deterministic-positives-only>
<begin_sequence> <MET> <LYS> <PHE> ...
<begin_contacts>
<p1> <p8> <SD> <CD1>
<p1> <p7> <CG> <CA>
<end_contacts>
<end>
```

Data: [timodonnell/protein-docs](https://huggingface.co/datasets/timodonnell/protein-docs) (deterministic-positives-only scheme, ~24M documents).

### Exp5: Random-3-Bins Contact Prediction

Trains on protein documents with distance-binned contacts, false contact injection with corrections, and pLDDT confidence tokens. Each contact is 6 tokens: correction flag + two positions + two atoms + distance bin. Contacts in random order.

```
<random-3-bins>
<begin_sequence> <MET> <LYS> <PHE> ...
<begin_contacts>
<non-correction> <p1> <p5> <SD> <CD1> <bin_lt4>
<non-correction> <p3> <p7> <CA> <CB> <bin_4_12>
<correction> <p3> <p7> <CG> <CB> <bin_lt4>
<plddt_80_85>
<end_contacts>
<end>
```

Key differences from exp4:
- **3 distance bins**: `<bin_lt4>` (< 4A), `<bin_4_12>` (4-12A), `<bin_gt12>` (> 12A)
- **False contacts** injected with wrong bins, then **corrected** later in the document
- **Random contact order** (not sorted by sequence separation)
- **pLDDT confidence bin** token appears once per document
- **Long-range upsampling** when budget-constrained
- Supports FSDP for training larger (2B) models

Data: local parquet files from [timodonnell/protein-docs](https://huggingface.co/datasets/timodonnell/protein-docs) (random-3-bins scheme).

## Project Structure

```
experiments/
  exp1_distance_matrix/        # Distance matrix, small model
  exp1b_distance_matrix_1b/    # Distance matrix, 1B model
  exp1c_distance_matrix_decimal/
  exp1d_distance_matrix_addition/
  exp2a_secondary_structure/   # SS from coordinates
  exp3_ss_bio2token/           # SS with bio2token
  exp4_contact_prediction/     # Deterministic contacts
  exp5_contact_prediction/     # Random-3-bins contacts
shared/                        # Shared utilities
scripts/                       # Data analysis scripts
```

Each experiment follows the pattern:
```
exp_name/
  src/
    train.py    # Training entry point
    data.py     # Vocabulary, data loading
  tests/
```

## Quick Start

```bash
uv sync --all-extras

# Exp5 training (7 GPUs, FSDP, 2B model)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run torchrun --nproc_per_node=7 \
-m experiments.exp5_contact_prediction.src.train \
--data-dir /path/to/random-3-bins-5x \
--num-layers 32 --batch-size 2 --lr 1.5e-4 \
--fsdp "full_shard auto_wrap" \
--fsdp_config experiments/exp5_contact_prediction/fsdp_config.json \
--output-dir outputs/exp5

# Tests
uv run pytest

# Lint / format
uv run ruff check .
uv run ruff format .
```
