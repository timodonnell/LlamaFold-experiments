"""Dataset and vocabulary for contacts-and-distances-v1 contact prediction.

Documents contain two types of statements:
- Contact statements (3 tokens): <mode> <p_i> <p_j>
  where mode is long-range-contact, medium-range-contact, or short-range-contact
  (CB-CB <= 8A, CASP-standard separation ranges)
- Distance statements (6 tokens): <distance> <p_i> <p_j> <atom1> <atom2> <d_val>
  with fine-grained 0.5A resolution distance bins for randomly sampled atom pairs.

Statements are rank-ordered: contacts tend to appear earlier in the document.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

# Control tokens
CONTROL_TOKENS = [
    "<contacts-and-distances-v1>",
    "<begin_sequence>",
    "<begin_statements>",
    "<end>",
]

# Contact type tokens (3-token statements)
CONTACT_TYPE_TOKENS = [
    "<long-range-contact>",
    "<medium-range-contact>",
    "<short-range-contact>",
]

# Distance statement marker
DISTANCE_MARKER = ["<distance>"]

# Distance value tokens: 64 bins at 0.5A resolution
# <d0.5> (< 0.5A), <d1.0> (0.5-1.0A), ..., <d32.0> (31.5-32.0A)
DISTANCE_TOKENS = [f"<d{i * 0.5:.1f}>" for i in range(1, 65)]

# pLDDT bin tokens
PLDDT_TOKENS = [
    "<plddt_lt70>",
    "<plddt_70_75>",
    "<plddt_75_80>",
    "<plddt_80_85>",
    "<plddt_85_90>",
    "<plddt_90_95>",
    "<plddt_95_100>",
]

# Standard 20 amino acids
AMINO_ACIDS = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL",
]

# Atom names (backbone + side chain)
ATOM_NAMES = [
    "C", "CA", "CB", "CD", "CD1", "CD2", "CE", "CE1", "CE2", "CE3",
    "CG", "CG1", "CG2", "CH2", "CZ", "CZ2", "CZ3",
    "N", "ND1", "ND2", "NE", "NE1", "NE2", "NH1", "NH2", "NZ",
    "O", "OD1", "OD2", "OE1", "OE2", "OG", "OG1", "OH",
    "SD", "SG", "OXT",
]

# Non-canonical residue token
EXTRA_TOKENS = ["<UNK>"]

MAX_POSITION = 2700

_BACKBONE = {"N", "CA", "C", "O", "OXT"}

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
    """Return all domain-specific vocabulary tokens."""
    tokens: list[str] = []
    tokens += CONTROL_TOKENS
    tokens += CONTACT_TYPE_TOKENS
    tokens += DISTANCE_MARKER
    tokens += DISTANCE_TOKENS
    tokens += PLDDT_TOKENS
    tokens += EXTRA_TOKENS
    tokens += [f"<{aa}>" for aa in AMINO_ACIDS]
    tokens += [f"<{atom}>" for atom in ATOM_NAMES]
    tokens += [f"<p{i}>" for i in range(MAX_POSITION + 1)]
    return tokens


def load_parquet_dataset(data_dir: str, split: str):
    """Load protein docs from local parquet files."""
    import os

    from datasets import load_dataset

    split_dir = os.path.join(data_dir, split)
    data_files = os.path.join(split_dir, "*.parquet")
    return load_dataset("parquet", data_files=data_files, split="train")


def filter_by_cluster_limit(dataset, max_docs_per_cluster: int, seed: int = 42):
    """Limit the number of documents per struct_cluster_id."""
    rng = np.random.RandomState(seed)

    cluster_to_indices: dict[str, list[int]] = defaultdict(list)
    cluster_ids = dataset["struct_cluster_id"]
    for i, cid in enumerate(cluster_ids):
        cluster_to_indices[cid].append(i)

    keep_indices: list[int] = []
    for indices in cluster_to_indices.values():
        if len(indices) <= max_docs_per_cluster:
            keep_indices.extend(indices)
        else:
            keep_indices.extend(
                rng.choice(indices, size=max_docs_per_cluster, replace=False).tolist()
            )

    keep_indices.sort()
    return dataset.select(keep_indices)
