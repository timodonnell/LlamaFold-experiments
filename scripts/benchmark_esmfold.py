"""Run ESMFold on the same proteins from an exp5 benchmark JSONL.

Reads the benchmark JSONL (which has sequences and GT contacts), predicts
structures with ESMFold, extracts contacts at 4A cutoff, and saves results.

Usage:
    uv run python scripts/benchmark_esmfold.py \
        --benchmark results/benchmark_exp5.jsonl \
        --output results/benchmark_esmfold.jsonl \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoTokenizer, EsmForProteinFolding

# 3-letter to 1-letter AA mapping
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "UNK": "X",
}

CONTACT_CUTOFF = 4.0  # Angstroms, matching exp5 bin_lt4


def extract_contacts_from_positions(positions, seq_len, cutoff=CONTACT_CUTOFF):
    """Extract residue-level contacts from CA atom positions.

    Args:
        positions: (seq_len, 37, 3) atom positions tensor (atom37 format).
        seq_len: number of residues.
        cutoff: distance cutoff in Angstroms.

    Returns:
        Set of (i, j) residue pairs (1-indexed) with CA distance < cutoff.
    """
    # CA is atom index 1 in atom37 format
    ca_positions = positions[:seq_len, 1, :]  # (seq_len, 3)

    # Compute pairwise distances
    diff = ca_positions.unsqueeze(0) - ca_positions.unsqueeze(1)  # (L, L, 3)
    dist = torch.sqrt((diff ** 2).sum(dim=-1))  # (L, L)

    contacts = set()
    for i in range(seq_len):
        for j in range(i + 2, seq_len):  # skip adjacent residues
            if dist[i, j].item() < cutoff:
                contacts.add((i + 1, j + 1))  # 1-indexed
    return contacts


def extract_contacts_allatom(positions, atom_mask, aatype, seq_len, cutoff=CONTACT_CUTOFF):
    """Extract contacts using all resolved atoms from ESMFold's output.

    ESMFold returns positions as (seq_len, n_atoms, 3) where n_atoms may be
    14 (backbone + CB representation). We check all atom pairs.
    Returns set of (i, j) 1-indexed pairs.
    """
    contacts = set()
    n_atoms = positions.shape[1]

    for i in range(seq_len):
        for j in range(i + 2, seq_len):
            # Use the atoms that exist (mask is atom37 but positions may be shorter)
            n_valid_i = min(n_atoms, int(atom_mask[i][:n_atoms].sum().item()))
            n_valid_j = min(n_atoms, int(atom_mask[j][:n_atoms].sum().item()))
            if n_valid_i == 0 or n_valid_j == 0:
                continue

            pos_i = positions[i][:n_valid_i]
            pos_j = positions[j][:n_valid_j]

            diff = pos_i.unsqueeze(1) - pos_j.unsqueeze(0)
            dist = torch.sqrt((diff ** 2).sum(dim=-1))
            min_dist = dist.min().item()

            if min_dist < cutoff:
                contacts.add((i + 1, j + 1))

    return contacts


def run_esmfold_benchmark(
    benchmark_path: str,
    output_path: str,
    device: str,
    contact_method: str = "allatom",
):
    # Load benchmark data
    print(f"Loading benchmark from {benchmark_path}...")
    records = []
    with open(benchmark_path) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} proteins")

    # Load ESMFold
    print("Loading ESMFold model...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
    model = model.to(device).eval()
    # Reduce memory usage
    model.esm = model.esm.half()
    print("ESMFold loaded")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w") as f:
        for pi, rec in enumerate(records):
            entry_id = rec["entry_id"]
            seq_len = rec["seq_len"]
            sequence_3letter = rec["sequence"]

            # Convert to 1-letter sequence
            sequence_1letter = "".join(THREE_TO_ONE.get(aa, "X") for aa in sequence_3letter)

            # Skip if sequence has unknown residues
            if "X" in sequence_1letter:
                print(f"[{pi+1}/{len(records)}] {entry_id}: skipping (unknown residues)")
                continue

            # Predict structure
            t0 = time.time()
            with torch.no_grad():
                inputs = tokenizer(sequence_1letter, return_tensors="pt", add_special_tokens=False)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
            elapsed = time.time() - t0

            # Extract contacts
            positions = outputs.positions[-1][0]  # last layer, first (only) batch
            atom_mask = outputs.atom37_atom_exists[0]

            if contact_method == "allatom":
                pred_contacts = extract_contacts_allatom(
                    positions, atom_mask, None, seq_len, CONTACT_CUTOFF
                )
            else:
                pred_contacts = extract_contacts_from_positions(
                    positions, seq_len, CONTACT_CUTOFF
                )

            # Get pLDDT (ESMFold returns 0-1, convert to 0-100)
            plddt_raw = outputs.plddt[0, :seq_len].mean().item()
            plddt = plddt_raw * 100 if plddt_raw <= 1.0 else plddt_raw

            result = {
                "entry_id": entry_id,
                "seq_len": seq_len,
                "predicted_contacts": sorted([list(p) for p in pred_contacts]),
                "n_predicted": len(pred_contacts),
                "mean_plddt": round(plddt, 2),
                "elapsed": round(elapsed, 2),
                "contact_method": contact_method,
                "contact_cutoff": CONTACT_CUTOFF,
            }

            f.write(json.dumps(result) + "\n")
            f.flush()

            print(
                f"[{pi+1}/{len(records)}] {entry_id} ({seq_len} res) | "
                f"{len(pred_contacts)} contacts, pLDDT={plddt:.1f}, {elapsed:.1f}s"
            )
            sys.stdout.flush()

    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run ESMFold on benchmark proteins")
    parser.add_argument("--benchmark", type=str, required=True,
                        help="Path to exp5 benchmark JSONL (for protein sequences)")
    parser.add_argument("--output", type=str, default="results/benchmark_esmfold.jsonl")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--contact-method", type=str, default="allatom",
                        choices=["allatom", "ca_only"],
                        help="Contact extraction method")
    args = parser.parse_args()

    run_esmfold_benchmark(
        benchmark_path=args.benchmark,
        output_path=args.output,
        device=args.device,
        contact_method=args.contact_method,
    )


if __name__ == "__main__":
    main()
