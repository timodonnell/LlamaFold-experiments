"""Training script for protein contact prediction (random-3-bins).

Trains a ~1B parameter Llama 3.2 architecture from scratch on protein structure
documents containing amino acid sequences and distance-binned atomic contacts
with correction tokens, false contact injection, and pLDDT bin tokens.
Documents are loaded from local parquet files and trained with standard causal
language modeling (next-token prediction on the full document).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.nn.functional as functional  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from tokenizers.models import WordLevel  # noqa: E402
from tokenizers.pre_tokenizers import WhitespaceSplit  # noqa: E402
from transformers import (  # noqa: E402
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

import wandb  # noqa: E402

from .data import (  # noqa: E402
    ATOM_NAMES,
    CORRECTION_TOKENS,
    DISTANCE_BIN_TOKENS,
    PLDDT_TOKENS,
    VALID_ATOMS,
    filter_by_cluster_limit,
    get_all_tokens,
    load_parquet_dataset,
)

# Index lookup for atom diversity tracking
_ATOM_NAME_TO_IDX = {name: i for i, name in enumerate(ATOM_NAMES)}
_N_ATOM_NAMES = len(ATOM_NAMES)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

from contextlib import nullcontext

# Precomputed sets for fast lookup during parsing
_ATOM_TOKEN_SET = {f"<{a}>" for a in ATOM_NAMES}
_POS_PATTERN = re.compile(r"^<p(\d+)>$")
_END_MARKERS = {"<end_contacts>", "<end>", "<eos>", "<pad>"}
_CORRECTION_TOKEN_SET = set(CORRECTION_TOKENS)
_BIN_TOKEN_SET = set(DISTANCE_BIN_TOKENS)
_PLDDT_TOKEN_SET = set(PLDDT_TOKENS)


# ---------------------------------------------------------------------------
# Tokenizer & model creation
# ---------------------------------------------------------------------------


def _is_main_process() -> bool:
    """Check if this is the main process in distributed training."""
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    return local_rank in (-1, 0)


def _get_dist_info() -> tuple[int, int]:
    """Return (rank, world_size). Falls back to (0, 1) for single-GPU."""
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def create_tokenizer() -> PreTrainedTokenizerFast:
    """Create a tokenizer from scratch with all protein document tokens."""
    special_tokens = get_all_tokens()
    all_tokens = ["<pad>", "<eos>", "\n"] + special_tokens
    vocab = {token: idx for idx, token in enumerate(all_tokens)}

    tokenizer_model = WordLevel(vocab=vocab, unk_token="<pad>")
    tokenizer = Tokenizer(tokenizer_model)
    tokenizer.pre_tokenizer = WhitespaceSplit()

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<pad>",
        pad_token="<pad>",
        eos_token="<eos>",
    )
    return hf_tokenizer


def create_model(vocab_size: int, **kwargs) -> LlamaForCausalLM:
    """Create a Llama 3.2 1B architecture model from scratch."""
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=kwargs.get("hidden_size", 2048),
        intermediate_size=kwargs.get("intermediate_size", 8192),
        num_hidden_layers=kwargs.get("num_layers", 16),
        num_attention_heads=kwargs.get("num_heads", 32),
        num_key_value_heads=kwargs.get("num_kv_heads", 8),
        max_position_embeddings=kwargs.get("max_seq_len", 8192),
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
        pad_token_id=0,
        eos_token_id=1,
    )
    return LlamaForCausalLM(config)


# ---------------------------------------------------------------------------
# Dataset & collator
# ---------------------------------------------------------------------------


class TextDataset(torch.utils.data.Dataset):
    """Wraps a HuggingFace dataset, tokenizes documents for causal LM training."""

    def __init__(self, hf_dataset, tokenizer: PreTrainedTokenizer, max_length: int = 8192):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        text = self.hf_dataset[idx]["document"]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        input_ids = encoding["input_ids"]

        if 0 in input_ids:
            tokens = text.split()
            unk_tokens = [t for t in tokens if self.tokenizer.convert_tokens_to_ids(t) == 0]
            raise ValueError(
                f"Unknown tokens found: {unk_tokens[:10]}... "
                f"Text preview: {text[:200]}"
            )

        return {
            "input_ids": input_ids,
            "attention_mask": encoding["attention_mask"],
            "labels": list(input_ids),
        }


class DataCollator:
    """Pads input_ids, attention_mask, and labels to equal length within a batch."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids = []
        attention_mask = []
        labels = []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),
        }


# ---------------------------------------------------------------------------
# Document parsing helpers
# ---------------------------------------------------------------------------

# A contact is (is_correction, pos1, pos2, atom1, atom2, distance_bin)
Contact = tuple[bool, int, int, str, str, str]


def parse_document(text: str) -> tuple[list[str], list[Contact], str]:
    """Parse a random-3-bins protein document into sequence, contacts, and prompt.

    Returns:
        sequence: List of 3-letter amino acid codes.
        contacts: List of (is_correction, pos1, pos2, atom1, atom2, bin) tuples.
        prompt: Text up to and including ``<begin_contacts>``.
    """
    tokens = text.split()

    begin_seq_idx = tokens.index("<begin_sequence>")
    begin_cont_idx = tokens.index("<begin_contacts>")

    seq = [t.strip("<>") for t in tokens[begin_seq_idx + 1 : begin_cont_idx]]

    # Find end of contacts
    end_cont_idx = len(tokens)
    for i in range(begin_cont_idx + 1, len(tokens)):
        if tokens[i] in ("<end_contacts>", "<end>"):
            end_cont_idx = i
            break

    contact_tokens = tokens[begin_cont_idx + 1 : end_cont_idx]
    contacts, _, _ = parse_generated_contacts(contact_tokens)

    prompt = " ".join(tokens[: begin_cont_idx + 1])
    return seq, contacts, prompt


def parse_generated_contacts(
    tokens: list[str],
) -> tuple[list[Contact], bool, str | None]:
    """Parse 6-token contact groups, tolerating a pLDDT token inline.

    Each contact: <correction|non-correction> <pos1> <pos2> <atom1> <atom2> <bin>
    A pLDDT token can appear between any two contacts.

    Returns:
        contacts: Parsed contacts (may be partial if grammar breaks).
        is_valid_grammar: True if all groups matched the expected pattern.
        plddt_token: The pLDDT bin token if found, else None.
    """
    contacts: list[Contact] = []
    is_valid = True
    plddt: str | None = None
    i = 0

    while i < len(tokens):
        tok = tokens[i]

        if tok in _END_MARKERS:
            break

        # pLDDT token can appear between contacts or after end_contacts
        if tok in _PLDDT_TOKEN_SET:
            plddt = tok.strip("<>")
            i += 1
            continue

        # Must be a correction/non-correction token starting a 6-token group
        if tok not in _CORRECTION_TOKEN_SET:
            is_valid = False
            break

        if i + 6 > len(tokens):
            is_valid = False
            break

        corr, t1, t2, t3, t4, t5 = tokens[i : i + 6]

        if any(t in _END_MARKERS for t in (t1, t2, t3, t4, t5)):
            is_valid = False
            break

        m1 = _POS_PATTERN.match(t1)
        m2 = _POS_PATTERN.match(t2)
        if m1 and m2 and t3 in _ATOM_TOKEN_SET and t4 in _ATOM_TOKEN_SET and t5 in _BIN_TOKEN_SET:
            is_correction = corr == "<correction>"
            contacts.append((
                is_correction,
                int(m1.group(1)),
                int(m2.group(1)),
                t3.strip("<>"),
                t4.strip("<>"),
                t5.strip("<>"),
            ))
            i += 6
        else:
            is_valid = False
            break

    return contacts, is_valid, plddt


def check_atom_validity(
    contacts: list[Contact],
    sequence: list[str],
) -> tuple[int, int]:
    """Count how many atom references are valid for the residue's amino acid.

    Returns:
        (valid_count, total_count)
    """
    valid = 0
    total = 0
    for _, pos1, pos2, atom1, atom2, _ in contacts:
        idx1 = pos1 - 1
        idx2 = pos2 - 1

        total += 1
        if 0 <= idx1 < len(sequence):
            aa = sequence[idx1]
            if aa in VALID_ATOMS and atom1 in VALID_ATOMS[aa]:
                valid += 1

        total += 1
        if 0 <= idx2 < len(sequence):
            aa = sequence[idx2]
            if aa in VALID_ATOMS and atom2 in VALID_ATOMS[aa]:
                valid += 1

    return valid, total


# ---------------------------------------------------------------------------
# Generation-based evaluation
# ---------------------------------------------------------------------------


def evaluate_generation(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    hf_dataset,
    device: torch.device,
    n_samples: int = 50,
    max_new_tokens: int = 8000,
    prefix_sizes: tuple[int, ...] = (0, 5, 10, 20),
    recall_cutoffs: tuple[int, ...] = (1, 10, 100),
) -> dict[str, dict[str, float]]:
    """Evaluate model generation on contact prediction.

    Returns:
        Nested dict ``{prefix_label: {metric_name: value}}``.
    """
    from tqdm import tqdm

    model.eval()
    end_token_id = tokenizer.convert_tokens_to_ids("<end>")
    rank, world_size = _get_dist_info()
    is_main = rank == 0

    n_samples = min(n_samples, len(hf_dataset))

    if world_size > 1:
        indices_tensor = torch.zeros(n_samples, dtype=torch.long, device=device)
        if is_main:
            indices_tensor[:] = torch.tensor(
                np.random.choice(len(hf_dataset), size=n_samples, replace=False),
                dtype=torch.long,
            )
        dist.broadcast(indices_tensor, src=0)
        indices = indices_tensor.cpu().tolist()
    else:
        indices = np.random.choice(len(hf_dataset), size=n_samples, replace=False).tolist()

    parsed: list[tuple[list[str], list[Contact], str]] = []
    for idx in indices:
        parsed.append(parse_document(hf_dataset[idx]["document"]))

    my_parsed = parsed[rank::world_size]

    # Accumulator layout per prefix:
    #   [0] n_valid_grammar
    #   [1] n_has_plddt
    #   [2] total_valid_atoms
    #   [3] total_atom_checks
    #   [4] bin_correct (generated contacts matching GT where bin also matches)
    #   [5] bin_total (generated contacts matching GT on pos+atoms)
    #   For each cutoff i:
    #     [6 + i*4 + 0] contact_recall_found
    #     [6 + i*4 + 1] contact_recall_total
    #     [6 + i*4 + 2] pos_recall_found
    #     [6 + i*4 + 3] pos_recall_total
    n_accum = 6 + 4 * len(recall_cutoffs)

    results: dict[str, dict[str, float]] = {}
    pbar = None
    if is_main:
        pbar = tqdm(
            total=len(my_parsed) * len(prefix_sizes),
            desc=f"Gen eval (×{world_size} GPUs)" if world_size > 1 else "Gen eval",
            unit="gen",
        )

    for n_prefix in prefix_sizes:
        accum = torch.zeros(n_accum, dtype=torch.float64, device=device)
        atom1_counts = torch.zeros(_N_ATOM_NAMES, dtype=torch.float64, device=device)
        atom2_counts = torch.zeros(_N_ATOM_NAMES, dtype=torch.float64, device=device)

        for sequence, gt_contacts, base_prompt in my_parsed:
            if pbar is not None:
                pbar.set_postfix_str(f"prefix={n_prefix}")

            # Build prompt with 6-token prefix contacts
            if n_prefix > 0 and gt_contacts:
                prefix_toks = []
                for is_corr, p1, p2, a1, a2, bin_tok in gt_contacts[:n_prefix]:
                    corr_tok = "<correction>" if is_corr else "<non-correction>"
                    prefix_toks.extend([corr_tok, f"<p{p1}>", f"<p{p2}>", f"<{a1}>", f"<{a2}>", f"<{bin_tok}>"])
                prompt = base_prompt + " " + " ".join(prefix_toks)
            else:
                prompt = base_prompt

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=end_token_id,
                    )
            except Exception as e:
                if is_main:
                    print(f"  Generation failed: {e}")
                accum[3] += len(gt_contacts) * 2
                if pbar is not None:
                    pbar.update(1)
                continue

            gen_ids = outputs[0][inputs["input_ids"].shape[1] :]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
            gen_contacts, valid_grammar, gen_plddt = parse_generated_contacts(gen_text.split())

            if valid_grammar and len(gen_contacts) > 0:
                accum[0] += 1  # n_valid_grammar
                if gen_plddt is not None:
                    accum[1] += 1  # n_has_plddt

                v, t = check_atom_validity(gen_contacts, sequence)
                accum[2] += v
                accum[3] += t

                # Track atom name diversity
                for _, _, _, a1, a2, _ in gen_contacts:
                    idx1 = _ATOM_NAME_TO_IDX.get(a1)
                    idx2 = _ATOM_NAME_TO_IDX.get(a2)
                    if idx1 is not None:
                        atom1_counts[idx1] += 1
                    if idx2 is not None:
                        atom2_counts[idx2] += 1

                # Bin accuracy: for generated contacts matching GT on (pos1,pos2,atom1,atom2)
                gt_contact_to_bin = {
                    (c[1], c[2], c[3], c[4]): c[5] for c in gt_contacts
                }
                for _, p1, p2, a1, a2, gen_bin in gen_contacts:
                    gt_bin = gt_contact_to_bin.get((p1, p2, a1, a2))
                    if gt_bin is not None:
                        accum[5] += 1  # bin_total
                        if gen_bin == gt_bin:
                            accum[4] += 1  # bin_correct
            else:
                accum[3] += len(gt_contacts) * 2

            # Recall: match on (pos1, pos2, atom1, atom2) ignoring correction and bin
            gen_contact_set = {(c[1], c[2], c[3], c[4]) for c in gen_contacts}
            gen_position_set = {(c[1], c[2]) for c in gen_contacts}
            for ci, k in enumerate(recall_cutoffs):
                gt_subset = gt_contacts[n_prefix : n_prefix + k]
                base = 6 + ci * 4
                accum[base] += sum(1 for c in gt_subset if (c[1], c[2], c[3], c[4]) in gen_contact_set)
                accum[base + 1] += len(gt_subset)
                accum[base + 2] += sum(1 for c in gt_subset if (c[1], c[2]) in gen_position_set)
                accum[base + 3] += len(gt_subset)

            if pbar is not None:
                pbar.update(1)

        if world_size > 1:
            dist.all_reduce(accum, op=dist.ReduceOp.SUM)
            dist.all_reduce(atom1_counts, op=dist.ReduceOp.SUM)
            dist.all_reduce(atom2_counts, op=dist.ReduceOp.SUM)

        a = accum.cpu().tolist()
        label = f"prefix_{n_prefix}"
        metrics: dict[str, float] = {
            "pct_valid_grammar": 100 * a[0] / n_samples if n_samples else 0,
            "pct_has_plddt": 100 * a[1] / max(a[0], 1),
            "pct_valid_atoms": 100 * a[2] / a[3] if a[3] else 0,
            "pct_bin_accuracy": 100 * a[4] / a[5] if a[5] else 0,
        }
        for ci, k in enumerate(recall_cutoffs):
            base = 6 + ci * 4
            metrics[f"contact_recall_top_{k}"] = 100 * a[base] / a[base + 1] if a[base + 1] else 0
            metrics[f"position_recall_top_{k}"] = (
                100 * a[base + 2] / a[base + 3] if a[base + 3] else 0
            )

        for slot_name, counts in [("atom1", atom1_counts.cpu()), ("atom2", atom2_counts.cpu())]:
            nonzero = counts[counts > 0]
            metrics[f"{slot_name}_n_unique"] = float(len(nonzero))
            if len(nonzero) > 0:
                probs = nonzero / nonzero.sum()
                metrics[f"{slot_name}_entropy"] = float(-(probs * probs.log()).sum().item())
            else:
                metrics[f"{slot_name}_entropy"] = 0.0

        results[label] = metrics

    if pbar is not None:
        pbar.close()
    return results


# ---------------------------------------------------------------------------
# Per-contact-position perplexity
# ---------------------------------------------------------------------------


_PPL_MAX_POSITIONS = 3000


def compute_contact_position_perplexity(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    hf_dataset,
    device: torch.device,
    n_samples: int = 200,
    max_length: int = 8192,
) -> dict[int, tuple[float, int]]:
    """Compute average perplexity at each contact position across documents.

    Walks through tokens to find 6-token contact groups, skipping inline
    pLDDT tokens, so positions are accurate regardless of pLDDT placement.

    Returns:
        Dict mapping contact position index to (perplexity, doc_count).
    """
    model.eval()
    begin_contacts_id = tokenizer.convert_tokens_to_ids("<begin_contacts>")
    end_contacts_id = tokenizer.convert_tokens_to_ids("<end_contacts>")
    end_id = tokenizer.convert_tokens_to_ids("<end>")
    correction_ids = {
        tokenizer.convert_tokens_to_ids("<correction>"),
        tokenizer.convert_tokens_to_ids("<non-correction>"),
    }
    plddt_ids = {tokenizer.convert_tokens_to_ids(t) for t in
                 ["<plddt_lt70>", "<plddt_70_75>", "<plddt_75_80>",
                  "<plddt_80_85>", "<plddt_85_90>", "<plddt_90_95>", "<plddt_95_100>"]}

    rank, world_size = _get_dist_info()

    n_samples = min(n_samples, len(hf_dataset))
    all_indices = list(range(n_samples))
    my_indices = all_indices[rank::world_size]

    loss_sum = torch.zeros(_PPL_MAX_POSITIONS, dtype=torch.float64, device=device)
    loss_count = torch.zeros(_PPL_MAX_POSITIONS, dtype=torch.float64, device=device)

    for i in my_indices:
        text = hf_dataset[i]["document"]
        encoding = tokenizer(
            text, truncation=True, max_length=max_length, padding=False, return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(device)

        with torch.no_grad():
            logits = model(input_ids=input_ids).logits

        shift_logits = logits[0, :-1]
        shift_labels = input_ids[0, 1:]
        per_token_loss = functional.cross_entropy(shift_logits, shift_labels, reduction="none")

        ids = input_ids[0].tolist()
        begin_pos = None
        end_pos = None
        for j, tid in enumerate(ids):
            if tid == begin_contacts_id:
                begin_pos = j
            elif tid in (end_contacts_id, end_id):
                end_pos = j
                break

        if begin_pos is None:
            continue
        if end_pos is None:
            end_pos = len(ids)

        # Walk through contact section, identifying 6-token groups
        contact_starts: list[int] = []
        j = begin_pos + 1
        while j < end_pos:
            tid = ids[j]
            if tid in plddt_ids:
                j += 1  # skip inline pLDDT token
                continue
            if tid in correction_ids:
                contact_starts.append(j)
                j += 6
            else:
                break  # unexpected token

        for c, start in enumerate(contact_starts):
            if c >= _PPL_MAX_POSITIONS:
                break
            loss_start = start - 1  # shift for causal LM offset
            loss_end = loss_start + 6
            if loss_end > len(per_token_loss):
                break
            loss_sum[c] += per_token_loss[loss_start:loss_end].mean().item()
            loss_count[c] += 1

    if world_size > 1:
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(loss_count, op=dist.ReduceOp.SUM)

    result: dict[int, tuple[float, int]] = {}
    for pos in range(_PPL_MAX_POSITIONS):
        count = int(loss_count[pos].item())
        if count > 0:
            result[pos] = (math.exp(loss_sum[pos].item() / count), count)
    return result


# ---------------------------------------------------------------------------
# Subsampled Trainer
# ---------------------------------------------------------------------------


class SubsampledTrainer(Trainer):
    """Trainer that randomly subsamples the eval dataset at each evaluation step."""

    def __init__(
        self,
        *args,
        full_eval_dataset=None,
        eval_subsample_size: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._full_eval_dataset = full_eval_dataset
        self._eval_subsample_size = eval_subsample_size

    def evaluate(self, eval_dataset=None, **kwargs):
        if (
            eval_dataset is None
            and self._full_eval_dataset is not None
            and self._eval_subsample_size is not None
        ):
            n = min(self._eval_subsample_size, len(self._full_eval_dataset))
            rank, world_size = _get_dist_info()
            if world_size > 1:
                idx_tensor = torch.zeros(n, dtype=torch.long, device=self.args.device)
                if rank == 0:
                    idx_tensor[:] = torch.tensor(
                        np.random.choice(len(self._full_eval_dataset), size=n, replace=False),
                        dtype=torch.long,
                    )
                dist.broadcast(idx_tensor, src=0)
                indices = idx_tensor.cpu().tolist()
            else:
                indices = np.random.choice(
                    len(self._full_eval_dataset), size=n, replace=False
                ).tolist()
            eval_dataset = torch.utils.data.Subset(self._full_eval_dataset, indices)
        return super().evaluate(eval_dataset=eval_dataset, **kwargs)


# ---------------------------------------------------------------------------
# Evaluation callback
# ---------------------------------------------------------------------------


class EvalCallback(TrainerCallback):
    """Callback to generate examples, compute contact perplexity, and run generation eval."""

    def __init__(
        self,
        val_hf_dataset,
        tokenizer: PreTrainedTokenizer,
        use_wandb: bool = True,
        perplexity_samples: int = 200,
        gen_eval_samples: int = 50,
        gen_max_new_tokens: int = 8000,
        max_length: int = 8192,
    ):
        self.val_hf_dataset = val_hf_dataset
        self.tokenizer = tokenizer
        self.use_wandb = use_wandb
        self.perplexity_samples = perplexity_samples
        self.gen_eval_samples = gen_eval_samples
        self.gen_max_new_tokens = gen_max_new_tokens
        self.max_length = max_length
        self._running = False

    def on_evaluate(self, args, state, control, model, **kwargs):
        if self._running:
            return
        self._running = True
        try:
            self._do_evaluate(args, state, control, model, **kwargs)
        finally:
            self._running = False

    def _do_evaluate(self, args, state, control, model, **kwargs):
        is_main = _is_main_process()

        raw_model = model.module if hasattr(model, "module") else model
        device = next(raw_model.parameters()).device

        # For FSDP: summon full parameters so generate() and forward() work
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        fsdp_model = None
        if isinstance(model, FSDP):
            fsdp_model = model
        elif isinstance(raw_model, FSDP):
            fsdp_model = raw_model
        ctx = FSDP.summon_full_params(fsdp_model) if fsdp_model is not None else nullcontext()

        with ctx:
            example_doc = None
            if is_main:
                example_doc = self._generate_example(raw_model, device)

            ppl_data = compute_contact_position_perplexity(
                model=raw_model,
                tokenizer=self.tokenizer,
                hf_dataset=self.val_hf_dataset,
                device=device,
                n_samples=self.perplexity_samples,
                max_length=self.max_length,
            )

            gen_results = evaluate_generation(
                model=raw_model,
                tokenizer=self.tokenizer,
                hf_dataset=self.val_hf_dataset,
                device=device,
                n_samples=self.gen_eval_samples,
                max_new_tokens=self.gen_max_new_tokens,
            )

        if not is_main:
            return

        print(f"\n[Step {state.global_step}] Contact Prediction Evaluation (random-3-bins):")
        if ppl_data:
            positions = sorted(ppl_data.keys())
            ppls = [ppl_data[p][0] for p in positions]
            print(f"  Contact positions tracked: {len(positions)}")
            print(f"  First contact perplexity: {ppl_data[positions[0]][0]:.2f}")
            print(f"  Median contact perplexity: {float(np.median(ppls)):.2f}")
            print(f"  Last contact perplexity: {ppl_data[positions[-1]][0]:.2f}")

        for label, metrics in gen_results.items():
            print(f"\n  [{label}]")
            for k, v in metrics.items():
                print(f"    {k}: {v:.2f}")

        if example_doc:
            print("\n  Generated document:")
            print(example_doc)

        if not self.use_wandb or wandb.run is None:
            return

        step = state.global_step

        if example_doc:
            wandb.log(
                {
                    "eval_examples/generated_document": wandb.Html(
                        f"<pre>{example_doc[:10000]}</pre>"
                    ),
                    "global_step": step,
                }
            )

        if ppl_data:
            positions = sorted(ppl_data.keys())
            perplexities = [ppl_data[p][0] for p in positions]
            counts = [ppl_data[p][1] for p in positions]

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            ax1.plot(positions, perplexities, linewidth=0.5)
            ax1.set_xlabel("Contact Position")
            ax1.set_ylabel("Perplexity")
            ax1.set_title(f"Per-Contact-Position Perplexity (Step {step})")
            ax2.plot(positions, counts, linewidth=0.5, color="orange")
            ax2.set_xlabel("Contact Position")
            ax2.set_ylabel("# Documents")
            ax2.set_title("Documents Contributing per Position")
            plt.tight_layout()
            wandb.log({
                "eval_examples/contact_perplexity_plot": wandb.Image(fig),
                "global_step": step,
            })
            plt.close(fig)

            wandb.log(
                {
                    "eval_examples/first_contact_ppl": perplexities[0],
                    "eval_examples/median_contact_ppl": float(np.median(perplexities)),
                    "eval_examples/last_contact_ppl": perplexities[-1],
                    "eval_examples/n_contact_positions": len(positions),
                    "global_step": step,
                }
            )

        for label, metrics in gen_results.items():
            for metric_name, value in metrics.items():
                wandb.log({f"gen_eval/{label}/{metric_name}": value, "global_step": step})

    def _generate_example(self, model: torch.nn.Module, device: torch.device) -> str | None:
        """Generate contacts for a random validation protein sequence."""
        model.eval()
        idx = np.random.randint(len(self.val_hf_dataset))
        doc = self.val_hf_dataset[int(idx)]["document"]
        _, _, prompt = parse_document(doc)

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        end_token_id = self.tokenizer.convert_tokens_to_ids("<end>")
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=2000,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=end_token_id,
                )
            return str(self.tokenizer.decode(outputs[0], skip_special_tokens=False))
        except Exception as e:
            print(f"  Generation failed: {e}")
            return None


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    data_dir: str,
    train_split: str = "train",
    val_split: str = "val",
    max_token_length: int = 8192,
    train_samples: int | None = None,
    eval_samples: int | None = None,
    gen_eval_samples: int = 50,
    gen_max_new_tokens: int = 8000,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    lr: float = 2e-4,
    n_epochs: int = 3,
    warmup_ratio: float = 0.1,
    warmup_steps: int | None = None,
    seed: int = 42,
    output_dir: str = "outputs/exp5",
    use_wandb: bool = True,
    wandb_project: str = "exp5",
    wandb_entity: str | None = "timodonnell",
    wandb_run_name: str | None = None,
    perplexity_samples: int = 200,
    eval_steps: int = 500,
    save_steps: int = 500,
    resume_from_checkpoint: str | bool | None = None,
    load_weights_only: str | None = None,
    max_docs_per_cluster: int | None = None,
    hidden_size: int = 2048,
    intermediate_size: int = 8192,
    num_layers: int = 16,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    fsdp: str = "",
    fsdp_config: str | None = None,
) -> dict[str, Any]:
    """Train the contact prediction LLM on random-3-bins documents."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    is_main = _is_main_process()

    torch.manual_seed(seed)
    np.random.seed(seed)

    if is_main:
        print("Setting up tokenizer and model...")
    tokenizer = create_tokenizer()
    model = create_model(
        vocab_size=len(tokenizer),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )

    if load_weights_only:
        ckpt_path = Path(load_weights_only)
        if is_main:
            print(f"Loading model weights from {ckpt_path} (optimizer/scheduler reset)...")
        state_dict = LlamaForCausalLM.from_pretrained(str(ckpt_path)).state_dict()
        model.load_state_dict(state_dict)
        del state_dict

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_config = model.config

    if is_main:
        print("\n" + "=" * 60)
        print("MODEL SUMMARY")
        print("=" * 60)
        print(f"  Total parameters:     {n_params:,}")
        print(f"  Trainable parameters: {n_trainable:,}")
        print(f"  Vocabulary size:      {len(tokenizer)}")
        print(f"  Hidden size:          {model_config.hidden_size}")
        print(f"  Intermediate size:    {model_config.intermediate_size}")
        print(f"  Num layers:           {model_config.num_hidden_layers}")
        print(f"  Num attention heads:  {model_config.num_attention_heads}")
        print(f"  Num KV heads:         {model_config.num_key_value_heads}")
        print(f"  Max sequence length:  {model_config.max_position_embeddings}")
        print("=" * 60 + "\n")

    if is_main:
        print(f"Loading datasets from {data_dir}...")
    hf_train = load_parquet_dataset(data_dir, train_split)
    hf_val = load_parquet_dataset(data_dir, val_split)

    if max_docs_per_cluster is not None:
        if is_main:
            print(f"  Filtering to max {max_docs_per_cluster} docs per struct_cluster_id...")
        hf_train = filter_by_cluster_limit(hf_train, max_docs_per_cluster, seed=seed)
        if is_main:
            print(f"  Train after cluster limit: {len(hf_train)} documents")

    if is_main:
        print(f"  Train: {len(hf_train)} documents")
        print(f"  Val:   {len(hf_val)} documents")

    if train_samples is not None and train_samples < len(hf_train):
        hf_train = hf_train.select(range(train_samples))
        if is_main:
            print(f"  Limited train to: {len(hf_train)} documents")

    train_dataset = TextDataset(hf_train, tokenizer, max_length=max_token_length)
    val_dataset = TextDataset(hf_val, tokenizer, max_length=max_token_length)

    config = {
        "experiment": "exp5",
        "task": "contact_prediction_random_3_bins",
        "data_dir": data_dir,
        "max_docs_per_cluster": max_docs_per_cluster,
        "train_samples": len(hf_train),
        "val_samples": len(hf_val),
        "eval_samples": eval_samples,
        "gen_eval_samples": gen_eval_samples,
        "gen_max_new_tokens": gen_max_new_tokens,
        "max_token_length": max_token_length,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "lr": lr,
        "n_epochs": n_epochs,
        "warmup_ratio": warmup_ratio,
        "warmup_steps": warmup_steps,
        "seed": seed,
        "n_params": n_params,
        "hidden_size": model_config.hidden_size,
        "num_layers": model_config.num_hidden_layers,
        "num_heads": model_config.num_attention_heads,
        "vocab_size": len(tokenizer),
    }

    data_collator = DataCollator(pad_token_id=tokenizer.pad_token_id)

    if warmup_steps is None:
        n_devices = max(int(os.environ.get("WORLD_SIZE", 1)), 1)
        steps_per_epoch = math.ceil(len(train_dataset) / (batch_size * n_devices))
        total_steps = steps_per_epoch * n_epochs
        warmup_steps = int(total_steps * warmup_ratio)

    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=n_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=lr,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        dataloader_num_workers=4,
        report_to="wandb" if use_wandb else "none",
        run_name=wandb_run_name,
        seed=seed,
        fsdp=fsdp if fsdp else None,
        fsdp_config=fsdp_config,
    )

    if use_wandb and is_main:
        wandb.init(project=wandb_project, entity=wandb_entity, name=wandb_run_name, config=config)
        wandb.config.update({"command": " ".join(sys.argv)})

    eval_callback = EvalCallback(
        val_hf_dataset=hf_val,
        tokenizer=tokenizer,
        use_wandb=use_wandb,
        perplexity_samples=perplexity_samples,
        gen_eval_samples=gen_eval_samples,
        gen_max_new_tokens=gen_max_new_tokens,
        max_length=max_token_length,
    )

    trainer = SubsampledTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[eval_callback],
        full_eval_dataset=val_dataset if eval_samples else None,
        eval_subsample_size=eval_samples,
    )

    if is_main:
        print("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    trainer.save_model(str(output_path / "final_model"))
    if is_main:
        tokenizer.save_pretrained(str(output_path / "final_model"))

    results: dict[str, Any] = {"config": config, "train_loss": train_result.training_loss}
    if is_main:
        with open(output_path / "results.json", "w") as f:
            json.dump(results, f, indent=2)

    if use_wandb and is_main:
        if wandb.run is not None:
            wandb.run.summary["train_loss"] = train_result.training_loss
        wandb.finish()

    return results


def main():
    parser = argparse.ArgumentParser(description="Train protein contact prediction LLM (random-3-bins)")
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Root directory containing train/val/test parquet subdirectories",
    )
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--max-token-length", type=int, default=8192)
    parser.add_argument("--train-samples", type=int, default=None, help="Limit training samples")
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=None,
        help="Subsample eval set to this many docs per eval step (random, different each step)",
    )
    parser.add_argument(
        "--gen-eval-samples",
        type=int,
        default=50,
        help="Number of docs for generation-based eval metrics",
    )
    parser.add_argument(
        "--gen-max-new-tokens",
        type=int,
        default=8000,
        help="Max tokens to generate per doc during gen eval (default 8000)",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--n-epochs", type=int, default=3)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument(
        "--warmup-steps", type=int, default=None, help="Warmup steps (overrides --warmup-ratio)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/exp5")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="exp5")
    parser.add_argument("--wandb-entity", type=str, default="timodonnell")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--perplexity-samples", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory, or 'latest' to resume from latest",
    )
    parser.add_argument(
        "--load-weights-only",
        type=str,
        default=None,
        help="Load model weights from checkpoint without restoring optimizer/scheduler",
    )
    parser.add_argument(
        "--max-docs-per-cluster",
        type=int,
        default=None,
        help="Limit training docs per struct_cluster_id (prevents overfitting to large clusters)",
    )
    parser.add_argument("--hidden-size", type=int, default=2048, help="Model hidden size")
    parser.add_argument("--intermediate-size", type=int, default=8192, help="MLP intermediate size")
    parser.add_argument("--num-layers", type=int, default=16, help="Number of transformer layers")
    parser.add_argument("--num-heads", type=int, default=32, help="Number of attention heads")
    parser.add_argument("--num-kv-heads", type=int, default=8, help="Number of KV heads (GQA)")
    parser.add_argument("--fsdp", type=str, default="", help="FSDP strategy (e.g. 'full_shard auto_wrap')")
    parser.add_argument("--fsdp_config", type=str, default=None, help="Path to FSDP config JSON file")

    args = parser.parse_args()

    resume = args.resume_from_checkpoint
    if resume == "latest":
        resume = True

    train(
        data_dir=args.data_dir,
        train_split=args.train_split,
        val_split=args.val_split,
        max_token_length=args.max_token_length,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        gen_eval_samples=args.gen_eval_samples,
        gen_max_new_tokens=args.gen_max_new_tokens,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr=args.lr,
        n_epochs=args.n_epochs,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        output_dir=args.output_dir,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        perplexity_samples=args.perplexity_samples,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        resume_from_checkpoint=resume,
        load_weights_only=args.load_weights_only,
        max_docs_per_cluster=args.max_docs_per_cluster,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        fsdp=args.fsdp,
        fsdp_config=args.fsdp_config,
    )


if __name__ == "__main__":
    main()
