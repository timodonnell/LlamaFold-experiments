"""Benchmark exp5 contact prediction on the test set.

Generates rollouts for a subsample of test proteins and saves raw results.
All evaluation and metrics are computed in the analysis notebook.

Saves one JSONL record per protein with:
- Protein metadata (entry_id, seq_len, sequence)
- Ground truth contacts (parsed from training data)
- Raw rollouts for each condition (seq_only, with_longest)

Usage:
    uv run python scripts/benchmark_exp5.py \
        --checkpoint outputs/exp5.ethereal-galaxy-3/checkpoint-125500 \
        --data-dir /home/ubuntu/protein-docs/random-3-bins-5x \
        --n-proteins 100 \
        --n-rollouts 10 \
        --output results/benchmark_exp5.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from transformers import LlamaForCausalLM

from experiments.exp5_contact_prediction.src.data import load_parquet_dataset
from experiments.exp5_contact_prediction.src.train import (
    create_tokenizer,
    parse_document,
    parse_generated_contacts,
)


def run_benchmark(
    checkpoint: str,
    data_dir: str,
    n_proteins: int,
    n_rollouts: int,
    max_new_tokens: int,
    output_path: str,
    device: str,
    seed: int,
):
    rng = np.random.RandomState(seed)

    print(f"Loading checkpoint: {checkpoint}")
    tokenizer = create_tokenizer()
    model = LlamaForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16)
    model = model.to(device).eval()
    end_token_id = tokenizer.convert_tokens_to_ids("<end>")

    print(f"Loading test set from {data_dir}...")
    ds = load_parquet_dataset(data_dir, "test")
    print(f"Test set: {len(ds)} documents")

    indices = rng.choice(len(ds), size=min(n_proteins, len(ds)), replace=False).tolist()
    print(f"Benchmarking {len(indices)} proteins, {n_rollouts} rollouts each")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    def generate_rollout(prompt):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=1.0,
                top_k=0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=end_token_id,
            )
        elapsed = time.time() - t0
        gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
        contacts, valid, plddt = parse_generated_contacts(gen_text.split())
        return contacts, valid, plddt, elapsed

    def serialize_contacts(contacts):
        """Convert contact tuples to JSON-serializable lists."""
        return [[c[0], c[1], c[2], c[3], c[4], c[5]] for c in contacts]

    with open(output_path, "w") as f:
        for pi, idx in enumerate(indices):
            doc = ds[idx]["document"]
            entry_id = ds[idx].get("entry_id", f"idx_{idx}")

            # Parse ground truth
            sequence, gt_contacts, base_prompt = parse_document(doc)
            seq_len = len(sequence)

            gt_lt4 = [c for c in gt_contacts if c[5] == "bin_lt4"]
            if not gt_lt4:
                continue

            # Find the longest-range GT contact (among bin_lt4)
            longest_contact = max(gt_lt4, key=lambda c: abs(c[1] - c[2]))
            longest_sep = abs(longest_contact[1] - longest_contact[2])

            # Build prompts
            prompt_seq_only = base_prompt

            is_corr, p1, p2, a1, a2, bin_tok = longest_contact
            corr_tok = "<correction>" if is_corr else "<non-correction>"
            prefix_tokens = f"{corr_tok} <p{p1}> <p{p2}> <{a1}> <{a2}> <{bin_tok}>"
            prompt_with_longest = base_prompt + " " + prefix_tokens

            record = {
                "entry_id": entry_id,
                "seq_len": seq_len,
                "sequence": sequence,
                "gt_contacts": serialize_contacts(gt_contacts),
                "longest_contact": [is_corr, p1, p2, a1, a2, bin_tok],
                "longest_contact_sep": longest_sep,
                "config": {
                    "checkpoint": checkpoint,
                    "n_rollouts": n_rollouts,
                    "max_new_tokens": max_new_tokens,
                    "seed": seed,
                },
            }

            for condition, prompt in [("seq_only", prompt_seq_only),
                                       ("with_longest", prompt_with_longest)]:
                rollouts_data = []
                for r in range(n_rollouts):
                    contacts, valid, plddt, elapsed = generate_rollout(prompt)
                    rollouts_data.append({
                        "contacts": serialize_contacts(contacts),
                        "valid_grammar": valid,
                        "plddt": plddt,
                        "elapsed": round(elapsed, 2),
                    })
                record[condition] = rollouts_data

            f.write(json.dumps(record) + "\n")
            f.flush()

            # Quick progress summary
            n_gt = sum(1 for c in gt_contacts if c[5] == "bin_lt4")
            avg_gen = np.mean([len(r["contacts"]) for r in record["seq_only"]])
            avg_time = np.mean([r["elapsed"] for r in record["seq_only"]])
            print(
                f"[{pi+1}/{len(indices)}] {entry_id} "
                f"({seq_len} res, {n_gt} GT lt4, longest_sep={longest_sep}) | "
                f"avg {avg_gen:.0f} contacts/rollout, {avg_time:.1f}s/rollout"
            )
            sys.stdout.flush()

    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark exp5 contact prediction")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--n-proteins", type=int, default=100)
    parser.add_argument("--n-rollouts", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=3440)
    parser.add_argument("--output", type=str, default="results/benchmark_exp5.jsonl")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_benchmark(
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        n_proteins=args.n_proteins,
        n_rollouts=args.n_rollouts,
        max_new_tokens=args.max_new_tokens,
        output_path=args.output,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
