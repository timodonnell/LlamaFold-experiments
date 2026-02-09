#!/usr/bin/env python
"""Preprocess protein structures into bio2token FSQ indices for exp3.

This script runs in bio2token's Python 3.11 environment (NOT our project's env).
It reads existing exp2a JSONL data for metadata (id, sequence, ss3), finds
corresponding CIF files, encodes all-atom coordinates with bio2token's pretrained
prot2token encoder, extracts backbone atom (N, CA, C, O) token indices, and
writes new JSONL files to data/exp3/.

Requirements:
    - bio2token repo cloned and set up with `uv sync`
    - Pretrained checkpoint at checkpoints/prot2token_pretrained/last.ckpt
    - CIF files for the proteins in the exp2a data

Usage (from the bio2token repo directory):
    uv run python /path/to/bio2token_preprocess.py \
        --exp2a-dir /path/to/data/exp2a \
        --cif-dir /path/to/cif_files \
        --output-dir /path/to/data/exp3 \
        --bio2token-dir . \
        --checkpoint checkpoints/prot2token_pretrained/last.ckpt
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import tempfile
from pathlib import Path

import torch
from Bio.PDB import PDBIO, MMCIFParser


def load_bio2token_model(bio2token_dir: str, checkpoint_path: str, device: torch.device):
    """Load the pretrained bio2token prot2token model.

    Args:
        bio2token_dir: Path to the bio2token repo root.
        checkpoint_path: Path to the pretrained checkpoint (.ckpt).
        device: Device to load model on.

    Returns:
        Loaded and eval-mode bio2token Autoencoder model.
    """
    # Add bio2token src to path
    sys.path.insert(0, str(Path(bio2token_dir) / "src"))

    from bio2token.models.autoencoder import Autoencoder, AutoencoderConfig
    from bio2token.utils.configs import pi_instantiate, utilsyaml_to_dict

    # Load config
    config_path = str(Path(bio2token_dir) / "configs" / "test_pdb.yaml")
    global_configs = utilsyaml_to_dict(config_path)

    # Instantiate model
    model_config = pi_instantiate(AutoencoderConfig, yaml_dict=global_configs["model"])
    model = Autoencoder(model_config)

    # Load pretrained weights (Lightning checkpoint has "model." prefix)
    state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval().to(device)

    return model


def cif_to_pdb_dict(cif_path: Path) -> dict | None:
    """Parse a CIF file and extract all-atom info in bio2token's expected format.

    Args:
        cif_path: Path to .cif or .cif.gz file.

    Returns:
        Dictionary with keys: seq, res_types, coords_groundtruth, atom_names,
        res_atom_start, res_atom_end. Or None if parsing fails.
    """
    try:
        parser = MMCIFParser(QUIET=True)

        if str(cif_path).endswith(".gz"):
            with gzip.open(cif_path, "rt") as f:
                structure = parser.get_structure("protein", f)
        else:
            structure = parser.get_structure("protein", str(cif_path))

        # Write to temp PDB for bio2token's pdb_2_dict
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w") as tmp:
            tmp_path = tmp.name

        io = PDBIO()
        io.set_structure(structure)
        io.save(tmp_path)

        return tmp_path

    except Exception as e:
        print(f"Error parsing {cif_path}: {e}", file=sys.stderr)
        return None


def encode_structure(
    model,
    pdb_path: str,
    device: torch.device,
) -> list[list[int]] | None:
    """Encode a protein structure into per-residue backbone bio2token indices.

    Args:
        model: Loaded bio2token Autoencoder.
        pdb_path: Path to a PDB file.
        device: Device for inference.

    Returns:
        List of [N_tok, CA_tok, C_tok, O_tok] per residue (values in 0-4095),
        or None on failure.
    """
    from bio2token.data.utils.utils import compute_masks, pdb_2_dict, uniform_dataframe

    try:
        pdb_dict = pdb_2_dict(pdb_path)
    except Exception as e:
        print(f"Error reading PDB {pdb_path}: {e}", file=sys.stderr)
        return None

    try:
        structure, unknown_structure, residue_name, residue_ids, token_class, atom_names = (
            uniform_dataframe(
                pdb_dict["seq"],
                pdb_dict["res_types"],
                pdb_dict["coords_groundtruth"],
                pdb_dict["atom_names"],
                pdb_dict["res_atom_start"],
                pdb_dict["res_atom_end"],
            )
        )
    except Exception as e:
        print(f"Error in uniform_dataframe for {pdb_path}: {e}", file=sys.stderr)
        return None

    # Build batch
    batch = {
        "structure": torch.tensor(structure).float(),
        "unknown_structure": torch.tensor(unknown_structure).bool(),
        "residue_ids": torch.tensor(residue_ids).long(),
        "token_class": torch.tensor(token_class).long(),
    }

    # Remove unknown atoms
    known_mask = ~batch["unknown_structure"]
    batch = {k: v[known_mask] for k, v in batch.items()}

    if batch["structure"].shape[0] == 0:
        return None

    # Compute masks
    batch = compute_masks(batch, structure_track=True)

    # Add batch dimension and move to device
    batch = {k: v[None].to(device) for k, v in batch.items()}

    # Run inference
    with torch.no_grad():
        batch = model(batch)

    # Extract indices
    all_indices = batch["indices"][0].cpu()  # (N_atoms,)
    all_token_class = batch["token_class"][0].cpu()  # (N_atoms,)
    all_residue_ids = batch["residue_ids"][0].cpu()  # (N_atoms,)
    pad_mask = batch["eos_pad_mask"][0].cpu()

    # Remove padding
    valid = pad_mask == 0
    all_indices = all_indices[valid]
    all_token_class = all_token_class[valid]
    all_residue_ids = all_residue_ids[valid]

    # Extract backbone atoms only (token_class 0=BB(N,C,O), 1=CA_ref)
    # Within each residue, bio2token orders atoms as: N, CA, C, O, then sidechains
    # So backbone atoms are the first 4 per residue with token_class 0 or 1
    backbone_mask = (all_token_class == 0) | (all_token_class == 1)
    bb_indices = all_indices[backbone_mask]
    bb_residue_ids = all_residue_ids[backbone_mask]

    # Group by residue: expect exactly 4 backbone atoms per residue (N, CA, C, O)
    unique_residues = bb_residue_ids.unique(sorted=True)
    result = []
    for res_id in unique_residues:
        res_mask = bb_residue_ids == res_id
        res_indices = bb_indices[res_mask].tolist()
        if len(res_indices) != 4:
            # Skip residues without exactly 4 backbone atoms
            continue
        # Verify all indices are in valid range
        if any(idx < 0 or idx >= 4096 for idx in res_indices):
            continue
        result.append(res_indices)

    return result if result else None


def find_cif_file(protein_id: str, cif_dir: Path) -> Path | None:
    """Find the CIF file for a given protein ID.

    Tries common naming patterns for AlphaFold DB files.

    Args:
        protein_id: Protein identifier (e.g., "AF-A0A009IHW8-F1").
        cif_dir: Directory containing CIF files.

    Returns:
        Path to the CIF file, or None if not found.
    """
    # Try common patterns
    patterns = [
        f"{protein_id}-model_v4.cif.gz",
        f"{protein_id}-model_v4.cif",
        f"{protein_id}-model_v3.cif.gz",
        f"{protein_id}-model_v3.cif",
        f"{protein_id}.cif.gz",
        f"{protein_id}.cif",
    ]

    for pattern in patterns:
        path = cif_dir / pattern
        if path.exists():
            return path

    return None


def process_split(
    split_name: str,
    exp2a_path: Path,
    cif_dir: Path,
    output_path: Path,
    model,
    device: torch.device,
) -> dict[str, int]:
    """Process one data split (train/val/test).

    Args:
        split_name: Name of the split ("train", "val", "test").
        exp2a_path: Path to the exp2a JSONL file.
        cif_dir: Directory containing CIF files.
        output_path: Path to write the output JSONL.
        model: Loaded bio2token model.
        device: Device for inference.

    Returns:
        Dictionary with processing statistics.
    """
    stats = {"total": 0, "success": 0, "no_cif": 0, "encode_fail": 0, "length_mismatch": 0}

    with open(exp2a_path) as fin, open(output_path, "w") as fout:
        for line_num, line in enumerate(fin):
            record = json.loads(line)
            stats["total"] += 1

            protein_id = record["id"]

            # Find CIF file
            cif_path = find_cif_file(protein_id, cif_dir)
            if cif_path is None:
                stats["no_cif"] += 1
                if stats["no_cif"] <= 5:
                    print(f"  Warning: no CIF file for {protein_id}", file=sys.stderr)
                continue

            # Convert CIF to temp PDB
            tmp_pdb_path = cif_to_pdb_dict(cif_path)
            if tmp_pdb_path is None:
                stats["encode_fail"] += 1
                continue

            try:
                # Encode with bio2token
                bio2token_indices = encode_structure(model, tmp_pdb_path, device)
            finally:
                # Clean up temp PDB
                Path(tmp_pdb_path).unlink(missing_ok=True)

            if bio2token_indices is None:
                stats["encode_fail"] += 1
                continue

            # Verify length matches the sequence
            # Note: bio2token may drop residues with missing atoms, so we need to
            # check if the number of residues with 4 backbone tokens matches
            seq_len = len(record["sequence"])
            if len(bio2token_indices) != seq_len:
                stats["length_mismatch"] += 1
                if stats["length_mismatch"] <= 5:
                    print(
                        f"  Warning: length mismatch for {protein_id}: "
                        f"seq={seq_len}, bio2token={len(bio2token_indices)}",
                        file=sys.stderr,
                    )
                continue

            # Write output record
            out_record = {
                "id": record["id"],
                "sequence": record["sequence"],
                "length": record["length"],
                "bio2token_indices": bio2token_indices,
                "ss3": record["ss3"],
            }
            fout.write(json.dumps(out_record) + "\n")
            stats["success"] += 1

            # Progress
            if (stats["total"]) % 100 == 0:
                print(
                    f"  [{split_name}] {stats['total']} processed, "
                    f"{stats['success']} success, "
                    f"{stats['no_cif']} no CIF, "
                    f"{stats['encode_fail']} encode fail, "
                    f"{stats['length_mismatch']} length mismatch"
                )

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess protein structures into bio2token FSQ indices"
    )
    parser.add_argument(
        "--exp2a-dir",
        type=str,
        required=True,
        help="Directory containing exp2a JSONL files (train.jsonl, val.jsonl, test.jsonl)",
    )
    parser.add_argument(
        "--cif-dir",
        type=str,
        required=True,
        help="Directory containing .cif.gz files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for exp3 JSONL files",
    )
    parser.add_argument(
        "--bio2token-dir",
        type=str,
        required=True,
        help="Path to the bio2token repo root",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to pretrained checkpoint "
        "(default: <bio2token-dir>/checkpoints/prot2token_pretrained/last.ckpt)",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val", "test"],
        help="Which splits to process",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference (cuda or cpu)",
    )

    args = parser.parse_args()

    exp2a_dir = Path(args.exp2a_dir)
    cif_dir = Path(args.cif_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = str(
            Path(args.bio2token_dir) / "checkpoints" / "prot2token_pretrained" / "last.ckpt"
        )

    device = torch.device(args.device)

    print(f"Loading bio2token model from {checkpoint}...")
    model = load_bio2token_model(args.bio2token_dir, checkpoint, device)
    print("Model loaded.")

    all_stats = {}
    for split in args.splits:
        exp2a_path = exp2a_dir / f"{split}.jsonl"
        if not exp2a_path.exists():
            print(f"Skipping {split}: {exp2a_path} not found")
            continue

        output_path = output_dir / f"{split}.jsonl"
        print(f"\nProcessing {split}...")
        stats = process_split(split, exp2a_path, cif_dir, output_path, model, device)
        all_stats[split] = stats

        print(f"  {split} done: {stats['success']}/{stats['total']} success")
        print(f"    no CIF: {stats['no_cif']}")
        print(f"    encode fail: {stats['encode_fail']}")
        print(f"    length mismatch: {stats['length_mismatch']}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for split, stats in all_stats.items():
        print(f"  {split}: {stats['success']}/{stats['total']} success")
    print("=" * 60)


if __name__ == "__main__":
    main()
