"""Dataset classes and tokenization for secondary structure prediction from coordinates.

Document format:
    <start>
    <seq>
    <MET> <ALA> <GLY> ...
    <coords>
    <c-12> <c46> <c-8> <c-10> <c44> <c-9> <c5> <c50> <c-3> <c6> <c49> <c-2>
    ...
    <ss>
    <H> <H> <C> ...
    <end>

Each coordinate line has 12 tokens: N(x,y,z) CA(x,y,z) C(x,y,z) O(x,y,z) for one residue.
Coordinates are backbone atom positions binned to nearest Angstrom.
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

# Coordinate range for binning (Angstroms)
COORD_MIN = -200
COORD_MAX = 200


def get_special_tokens() -> list[str]:
    """Return all special tokens for the vocabulary.

    Returns:
        List of special tokens:
        - 5 control tokens: <start>, <end>, <seq>, <coords>, <ss>
        - 20 amino acid tokens: <ALA>, <ARG>, ..., <VAL>
        - 3 secondary structure tokens: <H>, <E>, <C>
        - 401 coordinate tokens: <c-200>, <c-199>, ..., <c0>, ..., <c199>, <c200>
    """
    control = ["<start>", "<end>", "<seq>", "<coords>", "<ss>"]
    aa_tokens = [f"<{aa}>" for aa in AMINO_ACIDS]
    ss_tokens = [f"<{ss}>" for ss in SS_STATES]
    coord_tokens = [f"<c{i}>" for i in range(COORD_MIN, COORD_MAX + 1)]

    return control + aa_tokens + ss_tokens + coord_tokens


def coord_to_token(value: float) -> str:
    """Convert a coordinate value to a token.

    Args:
        value: Coordinate value in Angstroms.

    Returns:
        Token string like <c-12> or <c45>.
    """
    # Round and clamp to valid range
    binned = int(round(value))
    binned = max(COORD_MIN, min(COORD_MAX, binned))
    return f"<c{binned}>"


def format_document(sequence: str, coords_backbone: list[list[list[float]]], ss3: str) -> str:
    """Format sequence, backbone coordinates, and secondary structure as a training document.

    Args:
        sequence: Amino acid sequence in 1-letter code (e.g., "MAGIVL").
        coords_backbone: List of backbone coordinates per residue.
                        Each residue has [[N_x,N_y,N_z], [CA_x,CA_y,CA_z],
                                          [C_x,C_y,C_z], [O_x,O_y,O_z]]
        ss3: 3-state secondary structure string (e.g., "HHHEEC").

    Returns:
        Formatted document string.

    Raises:
        ValueError: If lengths don't match.
    """
    if len(sequence) != len(ss3):
        raise ValueError(f"Sequence length ({len(sequence)}) != SS length ({len(ss3)})")
    if len(sequence) != len(coords_backbone):
        raise ValueError(
            f"Sequence length ({len(sequence)}) != coords length ({len(coords_backbone)})"
        )

    # Convert 1-letter AA codes to tokens
    aa_tokens = []
    for aa in sequence:
        if aa in AA_1_TO_3:
            aa_tokens.append(f"<{AA_1_TO_3[aa]}>")
        else:
            raise ValueError(f"Unknown amino acid: {aa}")

    # Convert coordinates to tokens (one line per residue: N CA C O, 12 values total)
    coord_lines = []
    for residue_coords in coords_backbone:
        # residue_coords = [[N_xyz], [CA_xyz], [C_xyz], [O_xyz]]
        tokens = []
        for atom_coords in residue_coords:
            tokens.append(coord_to_token(atom_coords[0]))
            tokens.append(coord_to_token(atom_coords[1]))
            tokens.append(coord_to_token(atom_coords[2]))
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


def format_prompt(sequence: str, coords_backbone: list[list[list[float]]]) -> str:
    """Format a sequence and coordinates as a prompt for generation.

    Args:
        sequence: Amino acid sequence in 1-letter code.
        coords_backbone: List of backbone coordinates per residue.

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

    # Convert coordinates to tokens
    coord_lines = []
    for residue_coords in coords_backbone:
        tokens = []
        for atom_coords in residue_coords:
            tokens.append(coord_to_token(atom_coords[0]))
            tokens.append(coord_to_token(atom_coords[1]))
            tokens.append(coord_to_token(atom_coords[2]))
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
    """Training dataset for secondary structure prediction from coordinates.

    Uses a line-offset index to avoid loading the entire JSONL file into memory.
    Records are read from disk on demand.
    """

    def __init__(self, jsonl_path: str, max_length: int | None = None):
        """Initialize the dataset.

        Args:
            jsonl_path: Path to JSONL file with sequence/coords/ss3 records.
            max_length: Optional maximum sequence length to include.
        """
        self.jsonl_path = jsonl_path
        self.offsets = _build_line_index(jsonl_path, max_length)

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx: int) -> dict[str, str]:
        record = _read_record(self.jsonl_path, self.offsets[idx])
        text = format_document(record["sequence"], record["coords_backbone"], record["ss3"])
        return {"text": text}


class SSEvalDataset(Dataset):
    """Evaluation dataset for secondary structure prediction.

    Returns prompts and ground truth for generative evaluation.
    Uses a line-offset index to avoid loading the entire JSONL file into memory.
    """

    def __init__(self, jsonl_path: str, max_length: int | None = None):
        """Initialize the dataset.

        Args:
            jsonl_path: Path to JSONL file with sequence/coords/ss3 records.
            max_length: Optional maximum sequence length to include.
        """
        self.jsonl_path = jsonl_path
        self.offsets = _build_line_index(jsonl_path, max_length)

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = _read_record(self.jsonl_path, self.offsets[idx])
        prompt = format_prompt(record["sequence"], record["coords_backbone"])
        true_ss = list(record["ss3"])

        return {
            "prompt": prompt,
            "true_ss": true_ss,
            "sequence": record["sequence"],
            "length": record["length"],
            "id": record["id"],
        }
