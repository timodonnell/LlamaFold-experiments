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
Positions are 1-indexed (p1 = first residue in the sequence).
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

# Maximum residue position index (generous upper bound; max observed seq_len ~2041)
MAX_POSITION = 2700

# Backbone atoms valid for all amino acids
_BACKBONE = {"N", "CA", "C", "O", "OXT"}

# Valid heavy atoms per amino acid (backbone + side chain)
VALID_ATOMS: dict[str, set[str]] = {
    "ALA": _BACKBONE | {"CB"},
    "ARG": _BACKBONE | {"CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"},
    "ASN": _BACKBONE | {"CB", "CG", "OD1", "ND2"},
    "ASP": _BACKBONE | {"CB", "CG", "OD1", "OD2"},
    "CYS": _BACKBONE | {"CB", "SG"},
    "GLN": _BACKBONE | {"CB", "CG", "CD", "OE1", "NE2"},
    "GLU": _BACKBONE | {"CB", "CG", "CD", "OE1", "OE2"},
    "GLY": _BACKBONE,
    "HIS": _BACKBONE | {"CB", "CG", "ND1", "CD2", "CE1", "NE2"},
    "ILE": _BACKBONE | {"CB", "CG1", "CG2", "CD1"},
    "LEU": _BACKBONE | {"CB", "CG", "CD1", "CD2"},
    "LYS": _BACKBONE | {"CB", "CG", "CD", "CE", "NZ"},
    "MET": _BACKBONE | {"CB", "CG", "SD", "CE"},
    "PHE": _BACKBONE | {"CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "PRO": _BACKBONE | {"CB", "CG", "CD"},
    "SER": _BACKBONE | {"CB", "OG"},
    "THR": _BACKBONE | {"CB", "OG1", "CG2"},
    "TRP": _BACKBONE | {"CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
    "TYR": _BACKBONE | {"CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"},
    "VAL": _BACKBONE | {"CB", "CG1", "CG2"},
}


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
        split: Dataset split name (e.g., "train", "validation").
        dataset_name: HuggingFace dataset identifier.
        config: Dataset configuration/subset name.

    Returns:
        HuggingFace Dataset object with a "document" column.
    """
    return _hf_load_dataset(dataset_name, config, split=split)


def filter_by_cluster_limit(dataset, max_docs_per_cluster: int, seed: int = 42):
    """Limit the number of documents per struct_cluster_id.

    Randomly selects up to *max_docs_per_cluster* documents from each
    cluster, ensuring no cluster is over-represented during training.

    Args:
        dataset: HuggingFace Dataset with a ``struct_cluster_id`` column.
        max_docs_per_cluster: Maximum documents to keep per cluster.
        seed: Random seed for reproducible subsampling.

    Returns:
        Filtered HuggingFace Dataset.
    """
    from collections import defaultdict

    import numpy as np

    rng = np.random.RandomState(seed)

    # Group indices by cluster
    cluster_to_indices: dict[str, list[int]] = defaultdict(list)
    cluster_ids = dataset["struct_cluster_id"]
    for i, cid in enumerate(cluster_ids):
        cluster_to_indices[cid].append(i)

    # Subsample each cluster
    keep_indices: list[int] = []
    for indices in cluster_to_indices.values():
        if len(indices) <= max_docs_per_cluster:
            keep_indices.extend(indices)
        else:
            keep_indices.extend(rng.choice(indices, size=max_docs_per_cluster, replace=False).tolist())

    keep_indices.sort()
    return dataset.select(keep_indices)
