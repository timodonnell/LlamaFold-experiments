#!/usr/bin/env python
"""Preprocess protein structures into bio2token FSQ indices for exp3.

This script runs in bio2token's Python 3.11 environment (NOT our project's env).
It reads existing exp2a JSONL data for metadata (id, sequence, ss3), finds the
corresponding CIF file, converts it to PDB, and feeds all atoms through bio2token's
pretrained prot2token FSQ autoencoder. The resulting per-atom discrete tokens for
backbone atoms (N, CA, C, O) are extracted and written to new JSONL files.

Requirements:
    - bio2token repo cloned and set up with `uv sync`
    - Pretrained checkpoint (auto-detected in checkpoints/ dir)
    - CIF files directory (AlphaFold DB structures)

Usage (from the bio2token repo directory):
    .venv/bin/python /path/to/bio2token_preprocess.py \
        --exp2a-dir /path/to/data/exp2a \
        --cif-dir /path/to/cif_files \
        --output-dir /path/to/data/exp3 \
        --bio2token-dir .
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
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

    # Load config (utilsyaml_to_dict prepends "configs/" relative to CWD)
    old_cwd = os.getcwd()
    os.chdir(bio2token_dir)
    global_configs = utilsyaml_to_dict("test_pdb.yaml")
    os.chdir(old_cwd)

    # Instantiate model
    model_config = pi_instantiate(AutoencoderConfig, yaml_dict=global_configs["model"])
    model = Autoencoder(model_config)

    # Load pretrained weights (Lightning checkpoint has "model." prefix)
    state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval().to(device)

    return model


def find_checkpoint(bio2token_dir: str) -> str:
    """Auto-detect the prot2token pretrained checkpoint.

    Args:
        bio2token_dir: Path to the bio2token repo root.

    Returns:
        Path to the checkpoint file.

    Raises:
        FileNotFoundError: If no checkpoint is found.
    """
    ckpt_dir = Path(bio2token_dir) / "checkpoints" / "bio2token" / "prot2token_pretrained"
    if ckpt_dir.is_dir():
        ckpts = list(ckpt_dir.glob("*.ckpt"))
        if ckpts:
            return str(ckpts[0])

    # Fallback: try legacy path
    legacy = Path(bio2token_dir) / "checkpoints" / "prot2token_pretrained" / "last.ckpt"
    if legacy.exists():
        return str(legacy)

    raise FileNotFoundError(
        f"No prot2token checkpoint found in {ckpt_dir} or {legacy.parent}"
    )


def find_cif_file(protein_id: str, cif_dir: Path) -> Path | None:
    """Find the CIF file for a given protein ID.

    Args:
        protein_id: Protein identifier (e.g., "AF-A0A009IHW8-F1").
        cif_dir: Directory containing CIF files.

    Returns:
        Path to the CIF file, or None if not found.
    """
    for version in ("v6", "v4", "v3"):
        for ext in (".cif.gz", ".cif"):
            path = cif_dir / f"{protein_id}-model_{version}{ext}"
            if path.exists():
                return path

    # Try without version suffix
    for ext in (".cif.gz", ".cif"):
        path = cif_dir / f"{protein_id}{ext}"
        if path.exists():
            return path

    return None


def cif_to_temp_pdb(cif_path: Path) -> str | None:
    """Convert a CIF file to a temporary PDB file.

    Args:
        cif_path: Path to .cif or .cif.gz file.

    Returns:
        Path to temporary PDB file, or None on failure.
    """
    try:
        parser = MMCIFParser(QUIET=True)
        if str(cif_path).endswith(".gz"):
            with gzip.open(cif_path, "rt") as f:
                structure = parser.get_structure("protein", f)
        else:
            structure = parser.get_structure("protein", str(cif_path))

        with tempfile.NamedTemporaryFile(
            suffix=".pdb", delete=False, mode="w"
        ) as tmp:
            tmp_path = tmp.name

        io = PDBIO()
        io.set_structure(structure)
        io.save(tmp_path)
        return tmp_path
    except Exception as e:
        print(f"Error converting {cif_path}: {e}", file=sys.stderr)
        return None


def encode_all_atom(
    model,
    pdb_path: str,
    device: torch.device,
) -> list[list[int]] | None:
    """Encode all-atom structure and extract backbone bio2token indices.

    Uses bio2token's native pdb_2_dict + uniform_dataframe pipeline to get
    all-atom input, then extracts backbone (N, CA, C, O) token indices.

    Args:
        model: Loaded bio2token Autoencoder.
        pdb_path: Path to a PDB file.
        device: Device for inference.

    Returns:
        List of [N_tok, CA_tok, C_tok, O_tok] per residue (values 0-4095),
        or None on failure.
    """
    from bio2token.data.utils.utils import compute_masks, pdb_2_dict, uniform_dataframe

    # Suppress bio2token's print statements
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        pdb_dict = pdb_2_dict(pdb_path)
    except Exception as e:
        sys.stdout = old_stdout
        print(f"Error in pdb_2_dict: {e}", file=sys.stderr)
        return None
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout

    try:
        (
            structure,
            unknown_structure,
            residue_name,
            residue_ids,
            token_class,
            atom_names,
        ) = uniform_dataframe(
            pdb_dict["seq"],
            pdb_dict["res_types"],
            pdb_dict["coords_groundtruth"],
            pdb_dict["atom_names"],
            pdb_dict["res_atom_start"],
            pdb_dict["res_atom_end"],
        )
    except Exception as e:
        print(f"Error in uniform_dataframe: {e}", file=sys.stderr)
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
    valid = ~pad_mask
    all_indices = all_indices[valid]
    all_token_class = all_token_class[valid]
    all_residue_ids = all_residue_ids[valid]

    # Extract backbone atoms only (token_class 0=BB(N,C,O), 1=CA_ref)
    backbone_mask = (all_token_class == 0) | (all_token_class == 1)
    bb_indices = all_indices[backbone_mask]
    bb_residue_ids = all_residue_ids[backbone_mask]

    # Group by residue: expect exactly 4 backbone atoms (N, CA, C, O)
    unique_residues = bb_residue_ids.unique(sorted=True)
    result = []
    for res_id in unique_residues:
        res_mask = bb_residue_ids == res_id
        res_indices = bb_indices[res_mask].tolist()
        if len(res_indices) != 4:
            continue
        if any(idx < 0 or idx >= 4096 for idx in res_indices):
            continue
        result.append(res_indices)

    return result if result else None


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
    stats = {
        "total": 0,
        "success": 0,
        "no_cif": 0,
        "convert_fail": 0,
        "encode_fail": 0,
        "length_mismatch": 0,
    }

    with open(exp2a_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            record = json.loads(line)
            stats["total"] += 1

            protein_id = record["id"]

            # Find CIF file
            cif_path = find_cif_file(protein_id, cif_dir)
            if cif_path is None:
                stats["no_cif"] += 1
                if stats["no_cif"] <= 5:
                    print(
                        f"  Warning: no CIF for {protein_id}",
                        file=sys.stderr,
                    )
                continue

            # Convert CIF to temp PDB
            tmp_pdb_path = cif_to_temp_pdb(cif_path)
            if tmp_pdb_path is None:
                stats["convert_fail"] += 1
                continue

            try:
                # Encode all-atom structure with bio2token
                bio2token_indices = encode_all_atom(model, tmp_pdb_path, device)
            finally:
                Path(tmp_pdb_path).unlink(missing_ok=True)

            if bio2token_indices is None:
                stats["encode_fail"] += 1
                if stats["encode_fail"] <= 5:
                    print(
                        f"  Warning: encode failed for {protein_id}",
                        file=sys.stderr,
                    )
                continue

            # Verify length matches sequence
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
            if stats["total"] % 1000 == 0:
                print(
                    f"  [{split_name}] {stats['total']} processed, "
                    f"{stats['success']} success"
                )

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess CIF structures into bio2token FSQ indices"
    )
    parser.add_argument(
        "--exp2a-dir",
        type=str,
        required=True,
        help="Directory with exp2a JSONL files (train/val/test.jsonl)",
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
        help="Path to pretrained checkpoint (auto-detected if not set)",
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
        checkpoint = find_checkpoint(args.bio2token_dir)
    print(f"Using checkpoint: {checkpoint}")

    device = torch.device(args.device)

    print("Loading bio2token model...")
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
        stats = process_split(
            split, exp2a_path, cif_dir, output_path, model, device
        )
        all_stats[split] = stats

        print(f"  {split} done: {stats['success']}/{stats['total']} success")
        for key in ("no_cif", "convert_fail", "encode_fail", "length_mismatch"):
            if stats[key] > 0:
                print(f"    {key}: {stats[key]}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for split, stats in all_stats.items():
        print(f"  {split}: {stats['success']}/{stats['total']} success")
    print("=" * 60)


if __name__ == "__main__":
    main()
