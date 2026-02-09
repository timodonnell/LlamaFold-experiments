"""Dataset classes and tokenization for SS prediction with bio2token coordinate encoding.

Document format:
    <start>
    <seq>
    <MET> <ALA> <GLY> ...
    <coords>
    <b102> <b3451> <b890> <b2345>
    ...
    <ss>
    <H> <H> <C> ...
    <end>

Each coordinate line has 4 bio2token tokens: one per backbone atom (N, CA, C, O).
Bio2token indices come from a pretrained FSQ autoencoder (4096-token codebook).
"""

from __future__ import annotations

import json
import re
from typing import Any

from torch.utils.data import Dataset

# Standard 20 amino acids (3-letter codes for tokens)
AMINO_ACIDS = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLU",
    "GLN",
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

# 1-letter to 3-letter mapping
AA_1_TO_3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "E": "GLU",
    "Q": "GLN",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}

# Secondary structure states
SS_STATES = ["H", "E", "C"]

# Bio2token codebook size (FSQ levels [4,4,4,4,4,4] = 4096)
BIO2TOKEN_CODEBOOK_SIZE = 4096


def get_special_tokens() -> list[str]:
    """Return all special tokens for the vocabulary.

    Returns:
        List of special tokens:
        - 5 control tokens: <start>, <end>, <seq>, <coords>, <ss>
        - 20 amino acid tokens: <ALA>, <ARG>, ..., <VAL>
        - 3 secondary structure tokens: <H>, <E>, <C>
        - 4096 bio2token tokens: <b0>, <b1>, ..., <b4095>
    """
    control = ["<start>", "<end>", "<seq>", "<coords>", "<ss>"]
    aa_tokens = [f"<{aa}>" for aa in AMINO_ACIDS]
    ss_tokens = [f"<{ss}>" for ss in SS_STATES]
    bio2token_tokens = [f"<b{i}>" for i in range(BIO2TOKEN_CODEBOOK_SIZE)]

    return control + aa_tokens + ss_tokens + bio2token_tokens


def format_document(sequence: str, bio2token_indices: list[list[int]], ss3: str) -> str:
    """Format sequence, bio2token indices, and secondary structure as a training document.

    Args:
        sequence: Amino acid sequence in 1-letter code (e.g., "MAGIVL").
        bio2token_indices: List of bio2token indices per residue.
                          Each residue has [N_tok, CA_tok, C_tok, O_tok]
                          where each value is in [0, 4095].
        ss3: 3-state secondary structure string (e.g., "HHHEEC").

    Returns:
        Formatted document string.

    Raises:
        ValueError: If lengths don't match or indices out of range.
    """
    if len(sequence) != len(ss3):
        raise ValueError(f"Sequence length ({len(sequence)}) != SS length ({len(ss3)})")
    if len(sequence) != len(bio2token_indices):
        raise ValueError(
            f"Sequence length ({len(sequence)}) != bio2token length ({len(bio2token_indices)})"
        )

    # Convert 1-letter AA codes to tokens
    aa_tokens = []
    for aa in sequence:
        if aa in AA_1_TO_3:
            aa_tokens.append(f"<{AA_1_TO_3[aa]}>")
        else:
            raise ValueError(f"Unknown amino acid: {aa}")

    # Convert bio2token indices to tokens (one line per residue: N CA C O, 4 tokens)
    coord_lines = []
    for residue_indices in bio2token_indices:
        tokens = []
        for idx in residue_indices:
            if not (0 <= idx < BIO2TOKEN_CODEBOOK_SIZE):
                raise ValueError(
                    f"Bio2token index {idx} out of range [0, {BIO2TOKEN_CODEBOOK_SIZE})"
                )
            tokens.append(f"<b{idx}>")
        coord_lines.append(" ".join(tokens))

    # Convert SS to tokens
    ss_tokens = []
    for ss in ss3:
        if ss in SS_STATES:
            ss_tokens.append(f"<{ss}>")
        else:
            raise ValueError(f"Unknown secondary structure state: {ss}")

    # Build document
    lines = [
        "<start>",
        "<seq>",
        " ".join(aa_tokens),
        "<coords>",
        "\n".join(coord_lines),
        "<ss>",
        " ".join(ss_tokens),
        "<end>",
    ]

    return "\n".join(lines)


def format_prompt(sequence: str, bio2token_indices: list[list[int]]) -> str:
    """Format a sequence and bio2token indices as a prompt for generation.

    Args:
        sequence: Amino acid sequence in 1-letter code.
        bio2token_indices: List of bio2token indices per residue.

    Returns:
        Prompt string ending with <ss> marker.
    """
    # Convert 1-letter AA codes to tokens
    aa_tokens = []
    for aa in sequence:
        if aa in AA_1_TO_3:
            aa_tokens.append(f"<{AA_1_TO_3[aa]}>")
        else:
            raise ValueError(f"Unknown amino acid: {aa}")

    # Convert bio2token indices to tokens
    coord_lines = []
    for residue_indices in bio2token_indices:
        tokens = []
        for idx in residue_indices:
            if not (0 <= idx < BIO2TOKEN_CODEBOOK_SIZE):
                raise ValueError(
                    f"Bio2token index {idx} out of range [0, {BIO2TOKEN_CODEBOOK_SIZE})"
                )
            tokens.append(f"<b{idx}>")
        coord_lines.append(" ".join(tokens))

    # Build prompt (no SS labels, ends with <ss> marker)
    lines = [
        "<start>",
        "<seq>",
        " ".join(aa_tokens),
        "<coords>",
        "\n".join(coord_lines),
        "<ss>",
    ]

    return "\n".join(lines)


def _build_line_index(jsonl_path: str, max_length: int | None = None) -> list[int]:
    """Build an index of byte offsets for valid lines in a JSONL file.

    Only stores byte offsets, not the actual records, to avoid loading
    the entire dataset into memory.

    Args:
        jsonl_path: Path to the JSONL file.
        max_length: Optional maximum sequence length to include.

    Returns:
        List of byte offsets for each valid record.
    """
    length_pattern = re.compile(rb'"length":\s*(\d+)')
    offsets = []
    with open(jsonl_path, "rb") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            if max_length is not None:
                # Extract length via regex to avoid full JSON parse of large records
                match = length_pattern.search(line)
                if match and int(match.group(1)) > max_length:
                    continue
            offsets.append(offset)
    return offsets


def _read_record(jsonl_path: str, offset: int) -> dict[str, Any]:
    """Read a single record from a JSONL file at the given byte offset."""
    with open(jsonl_path, "rb") as f:
        f.seek(offset)
        line = f.readline()
        return json.loads(line)


class SSDataset(Dataset):
    """Training dataset for SS prediction from bio2token-encoded coordinates.

    Uses a line-offset index to avoid loading the entire JSONL file into memory.
    Records are read from disk on demand.
    """

    def __init__(self, jsonl_path: str, max_length: int | None = None):
        """Initialize the dataset.

        Args:
            jsonl_path: Path to JSONL file with sequence/bio2token_indices/ss3 records.
            max_length: Optional maximum sequence length to include.
        """
        self.jsonl_path = jsonl_path
        self.offsets = _build_line_index(jsonl_path, max_length)

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx: int) -> dict[str, str]:
        record = _read_record(self.jsonl_path, self.offsets[idx])
        text = format_document(record["sequence"], record["bio2token_indices"], record["ss3"])
        return {"text": text}


class SSEvalDataset(Dataset):
    """Evaluation dataset for secondary structure prediction.

    Returns prompts and ground truth for generative evaluation.
    Uses a line-offset index to avoid loading the entire JSONL file into memory.
    """

    def __init__(self, jsonl_path: str, max_length: int | None = None):
        """Initialize the dataset.

        Args:
            jsonl_path: Path to JSONL file with sequence/bio2token_indices/ss3 records.
            max_length: Optional maximum sequence length to include.
        """
        self.jsonl_path = jsonl_path
        self.offsets = _build_line_index(jsonl_path, max_length)

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = _read_record(self.jsonl_path, self.offsets[idx])
        prompt = format_prompt(record["sequence"], record["bio2token_indices"])
        true_ss = list(record["ss3"])

        return {
            "prompt": prompt,
            "true_ss": true_ss,
            "sequence": record["sequence"],
            "length": record["length"],
            "id": record["id"],
        }
