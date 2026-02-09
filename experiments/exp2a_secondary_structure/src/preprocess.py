"""Preprocess AlphaFold DB CIF files to extract sequences, coordinates, and secondary structure.

Converts .cif.gz files to JSONL format with sequence, backbone coordinates, secondary structure,
and additional DSSP annotations.

8-state to 3-state mapping:
    H, G, I -> H (helix)
    E, B -> E (strand)
    T, S, C, - -> C (coil)

Output fields per record:
    id: AlphaFold DB identifier
    sequence: amino acid sequence (1-letter codes)
    length: sequence length
    coords_backbone: backbone atom coordinates as [[N_xyz, CA_xyz, C_xyz, O_xyz], ...]
                     where each is [x, y, z] in Angstroms (rounded to 1 decimal)
    ss8: 8-state secondary structure (H, G, I, E, B, T, S, C, -)
    ss3: 3-state secondary structure (H, E, C)
    rsa: relative solvent accessibility per residue
    phi: phi dihedral angles per residue
    psi: psi dihedral angles per residue
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

from Bio.PDB import PDBIO, MMCIFParser
from Bio.PDB.DSSP import DSSP

# 8-state to 3-state mapping
SS8_TO_SS3 = {
    "H": "H",  # Alpha helix
    "G": "H",  # 3-10 helix
    "I": "H",  # Pi helix
    "E": "E",  # Extended strand
    "B": "E",  # Beta bridge
    "T": "C",  # Turn
    "S": "C",  # Bend
    "C": "C",  # Coil
    "-": "C",  # Not assigned
    " ": "C",  # Not assigned (space)
}


def process_cif_file(cif_path: Path) -> dict | None:
    """Process a single CIF file and extract sequence, coordinates, and DSSP annotations.

    Args:
        cif_path: Path to the .cif.gz file.

    Returns:
        Dictionary with coordinates and DSSP annotations, or None if processing fails.
    """
    try:
        # Parse CIF file
        parser = MMCIFParser(QUIET=True)

        # Handle both .cif.gz and .cif files
        if str(cif_path).endswith(".gz"):
            with gzip.open(cif_path, "rt") as f:
                structure = parser.get_structure("protein", f)
        else:
            structure = parser.get_structure("protein", str(cif_path))

        # Get the first model
        model = structure[0]

        # Save as temporary PDB for DSSP
        # DSSP requires a CRYST1 record, so we write the PDB and prepend one
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w") as tmp_pdb:
            tmp_pdb_path = tmp_pdb.name

        io = PDBIO()
        io.set_structure(structure)
        io.save(tmp_pdb_path)

        # Prepend CRYST1 record (dummy unit cell) since DSSP requires it
        with open(tmp_pdb_path) as f:
            pdb_content = f.read()
        with open(tmp_pdb_path, "w") as f:
            # Dummy CRYST1: 1x1x1 Angstrom cell, P1 space group
            f.write("CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1\n")
            f.write(pdb_content)

        try:
            # Run DSSP
            dssp = DSSP(model, tmp_pdb_path, dssp="mkdssp")
        finally:
            # Clean up temp file
            Path(tmp_pdb_path).unlink(missing_ok=True)

        # Extract all DSSP data and backbone coordinates
        sequence = []
        coords_backbone = []  # [[N_xyz, CA_xyz, C_xyz, O_xyz], ...]
        ss8_list = []
        rsa_list = []
        phi_list = []
        psi_list = []

        # Backbone atom names in order
        backbone_atoms = ["N", "CA", "C", "O"]

        for key in dssp.keys():
            residue_data = dssp[key]
            aa = residue_data[1]  # One-letter amino acid code

            # Skip non-standard amino acids (X)
            if aa == "X":
                continue

            # Get the residue from the model to extract backbone coordinates
            # DSSP key is (chain_id, res_id)
            chain_id, res_id = key
            try:
                residue = model[chain_id][res_id]
                # Check all backbone atoms are present
                if not all(atom_name in residue for atom_name in backbone_atoms):
                    continue

                # Extract coordinates for N, CA, C, O
                residue_coords = []
                for atom_name in backbone_atoms:
                    coord = residue[atom_name].get_coord()
                    residue_coords.append(
                        [
                            round(float(coord[0]), 1),
                            round(float(coord[1]), 1),
                            round(float(coord[2]), 1),
                        ]
                    )
                coords_backbone.append(residue_coords)
            except KeyError:
                # Residue not found, skip
                continue

            sequence.append(aa)
            ss8_list.append(residue_data[2])  # 8-state secondary structure
            rsa_list.append(round(residue_data[3], 3))  # Relative solvent accessibility
            phi_list.append(round(residue_data[4], 1))  # Phi angle
            psi_list.append(round(residue_data[5], 1))  # Psi angle

        # Check minimum length
        if len(sequence) < 10:
            return None

        # Convert 8-state to 3-state
        ss3_list = [SS8_TO_SS3.get(ss, "C") for ss in ss8_list]

        # Build result
        seq_str = "".join(sequence)
        ss8_str = "".join(ss8_list)
        ss3_str = "".join(ss3_list)

        # Extract ID from filename (e.g., AF-A0A009IHW8-F1-model_v4.cif.gz -> AF-A0A009IHW8-F1)
        filename = cif_path.name
        if filename.endswith(".cif.gz"):
            file_id = filename[:-7]  # Remove .cif.gz
        elif filename.endswith(".cif"):
            file_id = filename[:-4]  # Remove .cif
        else:
            file_id = filename

        # Remove -model_v* suffix if present
        if "-model_v" in file_id:
            file_id = file_id.split("-model_v")[0]

        return {
            "id": file_id,
            "sequence": seq_str,
            "length": len(seq_str),
            "coords_backbone": coords_backbone,
            "ss8": ss8_str,
            "ss3": ss3_str,
            "rsa": rsa_list,
            "phi": phi_list,
            "psi": psi_list,
        }

    except Exception as e:
        print(f"Error processing {cif_path}: {e}", file=sys.stderr)
        return None


def split_by_hash(record_id: str, train_ratio: float = 0.9, val_ratio: float = 0.05) -> str:
    """Deterministically split records by hashing the ID.

    Args:
        record_id: The record ID to hash.
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.

    Returns:
        "train", "val", or "test" based on hash.
    """
    # Hash the ID and take first 8 hex chars as integer
    hash_val = int(hashlib.md5(record_id.encode()).hexdigest()[:8], 16)
    normalized = hash_val / 0xFFFFFFFF

    if normalized < train_ratio:
        return "train"
    elif normalized < train_ratio + val_ratio:
        return "val"
    else:
        return "test"


def main():
    parser = argparse.ArgumentParser(description="Preprocess AlphaFold DB CIF files to JSONL")
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
        help="Output directory for JSONL files",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Maximum number of files to process (0 = all)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers",
    )

    args = parser.parse_args()

    cif_dir = Path(args.cif_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all CIF files
    cif_files = sorted(cif_dir.glob("*.cif.gz"))
    if not cif_files:
        # Try without .gz extension
        cif_files = sorted(cif_dir.glob("*.cif"))

    if not cif_files:
        print(f"No CIF files found in {cif_dir}", file=sys.stderr)
        sys.exit(1)

    if args.max_files > 0:
        cif_files = cif_files[: args.max_files]

    print(f"Processing {len(cif_files)} CIF files with {args.workers} workers...")

    # Process files in parallel
    n_processed = 0
    n_failed = 0
    n_skipped = 0

    # Open output files
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    test_path = output_dir / "test.jsonl"

    train_count = 0
    val_count = 0
    test_count = 0

    with (
        open(train_path, "w") as train_f,
        open(val_path, "w") as val_f,
        open(test_path, "w") as test_f,
    ):
        file_handles = {
            "train": train_f,
            "val": val_f,
            "test": test_f,
        }

        with Pool(processes=args.workers) as pool:
            for i, result in enumerate(pool.imap_unordered(process_cif_file, cif_files)):
                if result is None:
                    n_failed += 1
                elif result.get("length", 0) < 10:
                    n_skipped += 1
                else:
                    # Determine split
                    split = split_by_hash(result["id"])
                    file_handles[split].write(json.dumps(result) + "\n")

                    if split == "train":
                        train_count += 1
                    elif split == "val":
                        val_count += 1
                    else:
                        test_count += 1

                    n_processed += 1

                # Progress update every 1000 files
                if (i + 1) % 1000 == 0:
                    print(
                        f"Progress: {i + 1}/{len(cif_files)} "
                        f"(processed: {n_processed}, failed: {n_failed}, skipped: {n_skipped})"
                    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total files:     {len(cif_files)}")
    print(f"Processed:       {n_processed}")
    print(f"Failed:          {n_failed}")
    print(f"Skipped (short): {n_skipped}")
    print("\nOutput split:")
    print(f"  Train: {train_count}")
    print(f"  Val:   {val_count}")
    print(f"  Test:  {test_count}")
    print("\nOutput files:")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  {test_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
