"""Training script for protein contact prediction.

Trains a ~1B parameter Llama 3.2 architecture from scratch on protein structure
documents containing amino acid sequences and atomic contacts. Documents are
loaded from the HuggingFace dataset timodonnell/protein-docs and trained with
standard causal language modeling (next-token prediction on the full document).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
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

from .data import ATOM_NAMES, VALID_ATOMS, get_all_tokens, load_hf_dataset  # noqa: E402

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

# Precomputed sets for fast lookup during parsing
_ATOM_TOKEN_SET = {f"<{a}>" for a in ATOM_NAMES}
_POS_PATTERN = re.compile(r"^<p(\d+)>$")
_END_MARKERS = {"<end_contacts>", "<end>", "<eos>", "<pad>"}


# ---------------------------------------------------------------------------
# Tokenizer & model creation
# ---------------------------------------------------------------------------


def _is_main_process() -> bool:
    """Check if this is the main process in distributed training."""
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    return local_rank in (-1, 0)


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
    """Create a Llama 3.2 1B architecture model from scratch.

    Returns:
        Randomly initialized LlamaForCausalLM (~1.2B parameters).
    """
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

# A contact is (pos1, pos2, atom1, atom2) where positions are ints and atoms are strings.
Contact = tuple[int, int, str, str]


def parse_document(text: str) -> tuple[list[str], list[Contact], str]:
    """Parse a protein document into sequence, contacts, and prompt.

    Args:
        text: Full document text.

    Returns:
        sequence: List of 3-letter amino acid codes (1-indexed: sequence[0] = residue at p1).
        contacts: List of (pos1, pos2, atom1, atom2) tuples.
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
    contacts, _ = parse_generated_contacts(contact_tokens)

    prompt = " ".join(tokens[: begin_cont_idx + 1])
    return seq, contacts, prompt


def parse_generated_contacts(
    tokens: list[str],
) -> tuple[list[Contact], bool]:
    """Parse a flat token list into contacts and validate grammar.

    Grammar is valid if every token group is (position, position, atom, atom)
    with no leftover or misplaced tokens before an end marker.

    Args:
        tokens: Token strings (after ``<begin_contacts>``).

    Returns:
        contacts: Parsed contacts (may be partial if grammar breaks).
        is_valid_grammar: ``True`` if all groups matched the expected pattern.
    """
    contacts: list[Contact] = []
    is_valid = True
    i = 0

    while i < len(tokens):
        if tokens[i] in _END_MARKERS:
            break

        if i + 4 > len(tokens):
            is_valid = False
            break

        t1, t2, t3, t4 = tokens[i : i + 4]

        if any(t in _END_MARKERS for t in (t1, t2, t3, t4)):
            if t1 in _END_MARKERS:
                break
            is_valid = False
            break

        m1 = _POS_PATTERN.match(t1)
        m2 = _POS_PATTERN.match(t2)
        if m1 and m2 and t3 in _ATOM_TOKEN_SET and t4 in _ATOM_TOKEN_SET:
            contacts.append((int(m1.group(1)), int(m2.group(1)), t3.strip("<>"), t4.strip("<>")))
            i += 4
        else:
            is_valid = False
            break

    return contacts, is_valid


def check_contact_ordering(contacts: list[Contact]) -> bool:
    """Check that contacts have non-increasing sequence separation."""
    for i in range(1, len(contacts)):
        prev_sep = abs(contacts[i - 1][0] - contacts[i - 1][1])
        curr_sep = abs(contacts[i][0] - contacts[i][1])
        if curr_sep > prev_sep:
            return False
    return True


def check_atom_validity(
    contacts: list[Contact],
    sequence: list[str],
) -> tuple[int, int]:
    """Count how many atom references are valid for the residue's amino acid.

    Positions are 1-indexed: position ``p`` maps to ``sequence[p - 1]``.

    Returns:
        (valid_count, total_count)
    """
    valid = 0
    total = 0
    for pos1, pos2, atom1, atom2 in contacts:
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

    For each *prefix_size*, prompts the model with the sequence (and optionally
    the first *prefix_size* ground-truth contacts) and generates up to
    *max_new_tokens*.  Computes:

    1. % valid high-level grammar (repeating pos-pos-atom-atom groups)
    2. % valid grammar AND contacts in decreasing sequence-separation order
    3. % of atom tokens that are valid for the residue's amino acid
       (invalid-grammar documents contribute 0 valid / estimated total)
    4. Contact recall at each cutoff (fraction of top-K ground-truth contacts
       found anywhere in the output)

    Returns:
        Nested dict ``{prefix_label: {metric_name: value}}``.
    """
    model.eval()
    end_token_id = tokenizer.convert_tokens_to_ids("<end>")

    n_samples = min(n_samples, len(hf_dataset))
    indices = np.random.choice(len(hf_dataset), size=n_samples, replace=False)

    # Pre-parse all documents once
    parsed: list[tuple[list[str], list[Contact], str]] = []
    for idx in indices:
        doc = hf_dataset[int(idx)]["document"]
        parsed.append(parse_document(doc))

    results: dict[str, dict[str, float]] = {}

    from tqdm import tqdm

    total_generations = len(parsed) * len(prefix_sizes)
    pbar = tqdm(total=total_generations, desc="Gen eval", unit="gen")

    for n_prefix in prefix_sizes:
        n_valid_grammar = 0
        n_valid_grammar_and_order = 0
        total_valid_atoms = 0
        total_atom_checks = 0
        recall_found: dict[int, int] = {k: 0 for k in recall_cutoffs}
        recall_total: dict[int, int] = {k: 0 for k in recall_cutoffs}
        pos_recall_found: dict[int, int] = {k: 0 for k in recall_cutoffs}
        pos_recall_total: dict[int, int] = {k: 0 for k in recall_cutoffs}

        for sequence, gt_contacts, base_prompt in parsed:
            pbar.set_postfix_str(f"prefix={n_prefix}")
            # Build prompt with optional prefix contacts
            if n_prefix > 0 and gt_contacts:
                prefix = gt_contacts[:n_prefix]
                prefix_tokens = []
                for p1, p2, a1, a2 in prefix:
                    prefix_tokens.extend([f"<p{p1}>", f"<p{p2}>", f"<{a1}>", f"<{a2}>"])
                prompt = base_prompt + " " + " ".join(prefix_tokens)
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
                print(f"  Generation failed: {e}")
                continue

            # Decode only the generated tokens
            gen_ids = outputs[0][inputs["input_ids"].shape[1] :]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
            gen_tokens = gen_text.split()

            gen_contacts, valid_grammar = parse_generated_contacts(gen_tokens)

            # (1) Grammar
            if valid_grammar and len(gen_contacts) > 0:
                n_valid_grammar += 1

                # (2) Ordering — check full list (prefix + generated)
                if n_prefix > 0 and gt_contacts:
                    full_contacts = list(gt_contacts[:n_prefix]) + list(gen_contacts)
                else:
                    full_contacts = list(gen_contacts)
                if check_contact_ordering(full_contacts):
                    n_valid_grammar_and_order += 1

                # (3) Atom validity — only generated contacts
                v, t = check_atom_validity(gen_contacts, sequence)
                total_valid_atoms += v
                total_atom_checks += t
            else:
                # Invalid grammar: estimate atom positions, all invalid
                n_gen_before_end = len(gen_tokens)
                for ei, et in enumerate(gen_tokens):
                    if et in _END_MARKERS:
                        n_gen_before_end = ei
                        break
                estimated_contacts = n_gen_before_end // 4
                total_atom_checks += estimated_contacts * 2

            # (4) Contact recall — include prefix contacts in the output set
            if n_prefix > 0 and gt_contacts:
                all_output_contacts = set(gt_contacts[:n_prefix]) | set(gen_contacts)
            else:
                all_output_contacts = set(gen_contacts)

            # Position-only recall: match on (pos1, pos2) ignoring atoms
            all_output_positions = {(c[0], c[1]) for c in all_output_contacts}

            for k in recall_cutoffs:
                gt_subset = gt_contacts[:k]
                found = sum(1 for c in gt_subset if c in all_output_contacts)
                recall_found[k] += found
                recall_total[k] += len(gt_subset)

                pos_found = sum(1 for c in gt_subset if (c[0], c[1]) in all_output_positions)
                pos_recall_found[k] += pos_found
                pos_recall_total[k] += len(gt_subset)

            pbar.update(1)

        label = f"prefix_{n_prefix}"
        metrics: dict[str, float] = {
            "pct_valid_grammar": 100 * n_valid_grammar / n_samples if n_samples else 0,
            "pct_valid_grammar_and_order": (
                100 * n_valid_grammar_and_order / n_samples if n_samples else 0
            ),
            "pct_valid_atoms": (
                100 * total_valid_atoms / total_atom_checks if total_atom_checks else 0
            ),
        }
        for k in recall_cutoffs:
            metrics[f"contact_recall_top_{k}"] = (
                100 * recall_found[k] / recall_total[k] if recall_total[k] else 0
            )
            metrics[f"position_recall_top_{k}"] = (
                100 * pos_recall_found[k] / pos_recall_total[k] if pos_recall_total[k] else 0
            )
        results[label] = metrics

    pbar.close()
    return results


# ---------------------------------------------------------------------------
# Per-contact-position perplexity
# ---------------------------------------------------------------------------


def compute_contact_position_perplexity(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    hf_dataset,
    device: torch.device,
    n_samples: int = 200,
    max_length: int = 8192,
) -> dict[int, tuple[float, int]]:
    """Compute average perplexity at each contact position across documents.

    Returns:
        Dict mapping contact position index to (perplexity, doc_count).
    """
    model.eval()
    begin_contacts_id = tokenizer.convert_tokens_to_ids("<begin_contacts>")
    end_contacts_id = tokenizer.convert_tokens_to_ids("<end_contacts>")

    position_losses: dict[int, list[float]] = defaultdict(list)
    n_samples = min(n_samples, len(hf_dataset))

    for i in range(n_samples):
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
            elif tid == end_contacts_id:
                end_pos = j

        if begin_pos is None:
            continue
        if end_pos is None:
            end_pos = len(ids)

        contact_start = begin_pos + 1
        n_contacts = (end_pos - contact_start) // 4

        for c in range(n_contacts):
            loss_start = contact_start + c * 4 - 1
            loss_end = loss_start + 4
            if loss_end > len(per_token_loss):
                break
            position_losses[c].append(per_token_loss[loss_start:loss_end].mean().item())

    result: dict[int, tuple[float, int]] = {}
    for pos in sorted(position_losses.keys()):
        losses = position_losses[pos]
        result[pos] = (math.exp(sum(losses) / len(losses)), len(losses))
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
            indices = np.random.choice(len(self._full_eval_dataset), size=n, replace=False)
            eval_dataset = torch.utils.data.Subset(self._full_eval_dataset, indices.tolist())
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
        max_length: int = 8192,
    ):
        self.val_hf_dataset = val_hf_dataset
        self.tokenizer = tokenizer
        self.use_wandb = use_wandb
        self.perplexity_samples = perplexity_samples
        self.gen_eval_samples = gen_eval_samples
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
        if not _is_main_process():
            return

        raw_model = model.module if hasattr(model, "module") else model
        device = next(raw_model.parameters()).device

        # 1. Generate an example document
        example_doc = self._generate_example(raw_model, device)

        # 2. Per-contact-position perplexity
        ppl_data = compute_contact_position_perplexity(
            model=raw_model,
            tokenizer=self.tokenizer,
            hf_dataset=self.val_hf_dataset,
            device=device,
            n_samples=self.perplexity_samples,
            max_length=self.max_length,
        )

        # 3. Generation evaluation
        gen_results = evaluate_generation(
            model=raw_model,
            tokenizer=self.tokenizer,
            hf_dataset=self.val_hf_dataset,
            device=device,
            n_samples=self.gen_eval_samples,
        )

        # --- Terminal output ---
        print(f"\n[Step {state.global_step}] Contact Prediction Evaluation:")
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
                print(f"    {k}: {v:.2f}%")

        if example_doc:
            print("\n  Generated document preview:")
            print(f"  {example_doc[:500]}")
            if len(example_doc) > 500:
                print(f"  ... ({len(example_doc)} chars total)")

        # --- Wandb logging ---
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

        # Log generation eval metrics
        for label, metrics in gen_results.items():
            for metric_name, value in metrics.items():
                wandb.log({f"gen_eval/{label}/{metric_name}": value, "global_step": step})

    def _generate_example(self, model: torch.nn.Module, device: torch.device) -> str | None:
        """Generate an example document from scratch."""
        model.eval()
        prompt = "<deterministic-positives-only>"
        inputs = self.tokenizer(prompt, return_tensors="pt")
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
    dataset_name: str = "timodonnell/protein-docs",
    dataset_config: str = "default",
    train_split: str = "train",
    val_split: str = "validation",
    max_token_length: int = 8192,
    train_samples: int | None = None,
    eval_samples: int | None = None,
    gen_eval_samples: int = 50,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    lr: float = 2e-4,
    n_epochs: int = 3,
    warmup_ratio: float = 0.1,
    warmup_steps: int | None = None,
    seed: int = 42,
    output_dir: str = "outputs/exp4",
    use_wandb: bool = True,
    wandb_project: str = "exp4",
    wandb_entity: str | None = "timodonnell",
    wandb_run_name: str | None = None,
    perplexity_samples: int = 200,
    eval_steps: int = 500,
    save_steps: int = 500,
    resume_from_checkpoint: str | bool | None = None,
) -> dict[str, Any]:
    """Train the contact prediction LLM."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    is_main = _is_main_process()

    torch.manual_seed(seed)
    np.random.seed(seed)

    if is_main:
        print("Setting up tokenizer and model...")
    tokenizer = create_tokenizer()
    model = create_model(vocab_size=len(tokenizer))

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
        print("Loading datasets...")
    hf_train = load_hf_dataset(train_split, dataset_name, dataset_config)
    hf_val = load_hf_dataset(val_split, dataset_name, dataset_config)

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
        "experiment": "exp4",
        "task": "contact_prediction",
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "train_samples": len(hf_train),
        "val_samples": len(hf_val),
        "eval_samples": eval_samples,
        "gen_eval_samples": gen_eval_samples,
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

    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=n_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=lr,
        warmup_steps=warmup_steps if warmup_steps is not None else 0,
        warmup_ratio=0.0 if warmup_steps is not None else warmup_ratio,
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
        max_length=max_token_length,
    )

    trainer = SubsampledTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,  # fallback for non-subsampled eval
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
    parser = argparse.ArgumentParser(description="Train protein contact prediction LLM")
    parser.add_argument("--dataset", type=str, default="timodonnell/protein-docs")
    parser.add_argument("--dataset-config", type=str, default="default")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="validation")
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
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--n-epochs", type=int, default=3)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument(
        "--warmup-steps", type=int, default=None, help="Warmup steps (overrides --warmup-ratio)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/exp4")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="exp4")
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

    args = parser.parse_args()

    resume = args.resume_from_checkpoint
    if resume == "latest":
        resume = True

    train(
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        train_split=args.train_split,
        val_split=args.val_split,
        max_token_length=args.max_token_length,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        gen_eval_samples=args.gen_eval_samples,
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
    )


if __name__ == "__main__":
    main()
