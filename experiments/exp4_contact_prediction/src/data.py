"""Dataset and vocabulary for protein contact prediction.

Documents come pre-formatted from HuggingFace dataset timodonnell/protein-docs.
Each document contains an amino acid sequence and atomic contacts in a
tokenized text format suitable for causal LM training.

Document format:
    <deterministic-positives-only>
    <begin_sequence>
    <MET> <GLY> <VAL> ...
    <begin_contacts>
    <p15> <p66> <CG2> <CE>
    <p15> <p58> <CD1> <CB>
    ...
    <end_contacts>
    <end>

Each contact consists of 4 tokens: two position tokens and two atom tokens.
"""

from __future__ import annotations

from datasets import load_dataset as _hf_load_dataset

# Control / structural tokens
CONTROL_TOKENS = [
    "<deterministic-positives-only>",
    "<begin_sequence>",
    "<begin_contacts>",
    "<end_contacts>",
    "<end>",
]

# Standard 20 amino acids (3-letter codes)
AMINO_ACIDS = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
]

# Atom names (backbone + side chain)
ATOM_NAMES = [
    "C",
    "CA",
    "CB",
    "CD",
    "CD1",
    "CD2",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "CG",
    "CG1",
    "CG2",
    "CH2",
    "CZ",
    "CZ2",
    "CZ3",
    "N",
    "ND1",
    "ND2",
    "NE",
    "NE1",
    "NE2",
    "NH1",
    "NH2",
    "NZ",
    "O",
    "OD1",
    "OD2",
    "OE1",
    "OE2",
    "OG",
    "OG1",
    "OH",
    "SD",
    "SG",
    "OXT",
]

# Maximum residue position index (generous upper bound)
MAX_POSITION = 2000


def get_all_tokens() -> list[str]:
    """Return all domain-specific vocabulary tokens.

    Does not include utility tokens (pad, eos, newline) which are added
    by the tokenizer setup.

    Returns:
        List of token strings.
    """
    tokens: list[str] = []
    tokens += CONTROL_TOKENS
    tokens += [f"<{aa}>" for aa in AMINO_ACIDS]
    tokens += [f"<{atom}>" for atom in ATOM_NAMES]
    tokens += [f"<p{i}>" for i in range(MAX_POSITION + 1)]
    return tokens


def load_hf_dataset(
    split: str,
    dataset_name: str = "timodonnell/protein-docs",
    config: str = "default",
):
    """Load protein docs dataset from HuggingFace.

    Args:
        split: Dataset split name (e.g., "train", "val").
        dataset_name: HuggingFace dataset identifier.
        config: Dataset configuration/subset name.

    Returns:
        HuggingFace Dataset object with a "document" column.
    """
    return _hf_load_dataset(dataset_name, config, split=split)
