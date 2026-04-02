"""Training script for contacts-and-distances-v1 protein contact prediction.

Trains a Llama model from scratch on documents containing both contact
statements (3 tokens: mode + two positions) and distance statements
(6 tokens: marker + two positions + two atoms + distance bin).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from contextlib import nullcontext
from dataclasses import dataclass
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
    CONTACT_TYPE_TOKENS,
    DISTANCE_MARKER,
    DISTANCE_TOKENS,
    PLDDT_TOKENS,
    VALID_ATOMS,
    filter_by_cluster_limit,
    get_all_tokens,
    load_parquet_dataset,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

# Precomputed sets for fast parsing
_ATOM_TOKEN_SET = {f"<{a}>" for a in ATOM_NAMES}
_POS_PATTERN = re.compile(r"^<p(\d+)>$")
_END_MARKERS = {"<end>", "<eos>", "<pad>"}
_CONTACT_TYPE_SET = set(CONTACT_TYPE_TOKENS)
_DISTANCE_MARKER_SET = set(DISTANCE_MARKER)
_DISTANCE_TOKEN_SET = set(DISTANCE_TOKENS)
_PLDDT_TOKEN_SET = set(PLDDT_TOKENS)


# ---------------------------------------------------------------------------
# Statement types
# ---------------------------------------------------------------------------

@dataclass
class ContactStatement:
    contact_type: str  # "long-range-contact", "medium-range-contact", "short-range-contact"
    pos1: int
    pos2: int


@dataclass
class DistanceStatement:
    pos1: int
    pos2: int
    atom1: str
    atom2: str
    distance_token: str  # e.g. "d4.5"


Statement = ContactStatement | DistanceStatement


# ---------------------------------------------------------------------------
# Tokenizer & model creation
# ---------------------------------------------------------------------------


def _is_main_process() -> bool:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    return local_rank in (-1, 0)


def _get_dist_info() -> tuple[int, int]:
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def create_tokenizer() -> PreTrainedTokenizerFast:
    special_tokens = get_all_tokens()
    all_tokens = ["<pad>", "<eos>", "\n"] + special_tokens
    vocab = {token: idx for idx, token in enumerate(all_tokens)}

    tokenizer_model = WordLevel(vocab=vocab, unk_token="<pad>")
    tokenizer = Tokenizer(tokenizer_model)
    tokenizer.pre_tokenizer = WhitespaceSplit()

    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<pad>",
        pad_token="<pad>",
        eos_token="<eos>",
    )


def create_model(vocab_size: int, **kwargs) -> LlamaForCausalLM:
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
    def __init__(self, hf_dataset, tokenizer: PreTrainedTokenizer, max_length: int = 8192):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        text = self.hf_dataset[idx]["document"]
        encoding = self.tokenizer(text, truncation=True, max_length=self.max_length,
                                  padding=False, return_tensors=None)
        input_ids = encoding["input_ids"]
        if 0 in input_ids:
            tokens = text.split()
            unk = [t for t in tokens if self.tokenizer.convert_tokens_to_ids(t) == 0]
            raise ValueError(f"Unknown tokens: {unk[:10]}... Text: {text[:200]}")
        return {"input_ids": input_ids, "attention_mask": encoding["attention_mask"],
                "labels": list(input_ids)}


class DataCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)
        return {"input_ids": torch.tensor(input_ids),
                "attention_mask": torch.tensor(attention_mask),
                "labels": torch.tensor(labels)}


# ---------------------------------------------------------------------------
# Document parsing
# ---------------------------------------------------------------------------


def parse_document(text: str) -> tuple[list[str], list[Statement], str]:
    """Parse a contacts-and-distances-v1 document.

    Returns:
        sequence: List of amino acid codes.
        statements: List of ContactStatement and DistanceStatement.
        prompt: Text up to and including <begin_statements>.
    """
    tokens = text.split()
    begin_seq_idx = tokens.index("<begin_sequence>")
    begin_stmt_idx = tokens.index("<begin_statements>")
    seq = [t.strip("<>") for t in tokens[begin_seq_idx + 1:begin_stmt_idx]]

    end_idx = len(tokens)
    for i in range(begin_stmt_idx + 1, len(tokens)):
        if tokens[i] == "<end>":
            end_idx = i
            break

    stmt_tokens = tokens[begin_stmt_idx + 1:end_idx]
    statements, _, _ = parse_generated_statements(stmt_tokens)
    prompt = " ".join(tokens[:begin_stmt_idx + 1])
    return seq, statements, prompt


def parse_generated_statements(
    tokens: list[str],
) -> tuple[list[Statement], bool, str | None]:
    """Parse a mixed stream of contact and distance statements.

    Returns:
        statements: Parsed statements.
        is_valid_grammar: True if parsing completed without errors.
        plddt_token: The pLDDT token if found, else None.
    """
    statements: list[Statement] = []
    is_valid = True
    plddt: str | None = None
    i = 0

    while i < len(tokens):
        tok = tokens[i]

        if tok in _END_MARKERS:
            break

        # pLDDT token (standalone, between statements)
        if tok in _PLDDT_TOKEN_SET:
            plddt = tok.strip("<>")
            i += 1
            continue

        # Contact statement: <mode> <p_i> <p_j> (3 tokens)
        if tok in _CONTACT_TYPE_SET:
            if i + 3 > len(tokens):
                is_valid = False
                break
            t1, t2 = tokens[i + 1], tokens[i + 2]
            if any(t in _END_MARKERS for t in (t1, t2)):
                is_valid = False
                break
            m1 = _POS_PATTERN.match(t1)
            m2 = _POS_PATTERN.match(t2)
            if m1 and m2:
                contact_type = tok.strip("<>")
                statements.append(ContactStatement(
                    contact_type=contact_type,
                    pos1=int(m1.group(1)),
                    pos2=int(m2.group(1)),
                ))
                i += 3
            else:
                is_valid = False
                break
            continue

        # Distance statement: <distance> <p_i> <p_j> <atom1> <atom2> <d_val> (6 tokens)
        if tok in _DISTANCE_MARKER_SET:
            if i + 6 > len(tokens):
                is_valid = False
                break
            t1, t2, t3, t4, t5 = tokens[i + 1:i + 6]
            if any(t in _END_MARKERS for t in (t1, t2, t3, t4, t5)):
                is_valid = False
                break
            m1 = _POS_PATTERN.match(t1)
            m2 = _POS_PATTERN.match(t2)
            if m1 and m2 and t3 in _ATOM_TOKEN_SET and t4 in _ATOM_TOKEN_SET and t5 in _DISTANCE_TOKEN_SET:
                statements.append(DistanceStatement(
                    pos1=int(m1.group(1)),
                    pos2=int(m2.group(1)),
                    atom1=t3.strip("<>"),
                    atom2=t4.strip("<>"),
                    distance_token=t5.strip("<>"),
                ))
                i += 6
            else:
                is_valid = False
                break
            continue

        # Unknown token
        is_valid = False
        break

    return statements, is_valid, plddt


def check_atom_validity(statements: list[Statement], sequence: list[str]) -> tuple[int, int]:
    """Check atom validity for distance statements only."""
    valid = 0
    total = 0
    for stmt in statements:
        if not isinstance(stmt, DistanceStatement):
            continue
        for pos, atom in [(stmt.pos1, stmt.atom1), (stmt.pos2, stmt.atom2)]:
            idx = pos - 1
            total += 1
            if 0 <= idx < len(sequence):
                aa = sequence[idx]
                if aa in VALID_ATOMS and atom in VALID_ATOMS[aa]:
                    valid += 1
    return valid, total


# ---------------------------------------------------------------------------
# Generation evaluation
# ---------------------------------------------------------------------------


def evaluate_generation(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    hf_dataset,
    device: torch.device,
    n_samples: int = 50,
    max_new_tokens: int = 8000,
    prefix_sizes: tuple[int, ...] = (0,),
) -> dict[str, dict[str, float]]:
    """Evaluate generation quality."""
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
                np.random.choice(len(hf_dataset), size=n_samples, replace=False), dtype=torch.long)
        dist.broadcast(indices_tensor, src=0)
        indices = indices_tensor.cpu().tolist()
    else:
        indices = np.random.choice(len(hf_dataset), size=n_samples, replace=False).tolist()

    parsed = []
    for idx in indices:
        parsed.append(parse_document(hf_dataset[idx]["document"]))
    my_parsed = parsed[rank::world_size]

    # Accumulators:
    # [0] n_valid_grammar
    # [1] n_has_plddt
    # [2] total_valid_atoms
    # [3] total_atom_checks
    # [4] n_contact_stmts_generated
    # [5] n_distance_stmts_generated
    # [6] contact_correct (position pairs matching GT)
    # [7] contact_total_pred
    # [8] contact_total_gt
    # [9] distance_correct_within_1A
    # [10] distance_total
    n_accum = 11

    results: dict[str, dict[str, float]] = {}
    pbar = tqdm(total=len(my_parsed), desc="Gen eval", unit="gen") if is_main else None

    for n_prefix in prefix_sizes:
        accum = torch.zeros(n_accum, dtype=torch.float64, device=device)

        for sequence, gt_stmts, base_prompt in my_parsed:
            if pbar:
                pbar.set_postfix_str(f"prefix={n_prefix}")

            # Build prompt with prefix statements
            if n_prefix > 0 and gt_stmts:
                prefix_toks = []
                for stmt in gt_stmts[:n_prefix]:
                    if isinstance(stmt, ContactStatement):
                        prefix_toks.extend([f"<{stmt.contact_type}>", f"<p{stmt.pos1}>", f"<p{stmt.pos2}>"])
                    else:
                        prefix_toks.extend([f"<distance>", f"<p{stmt.pos1}>", f"<p{stmt.pos2}>",
                                            f"<{stmt.atom1}>", f"<{stmt.atom2}>", f"<{stmt.distance_token}>"])
                prompt = base_prompt + " " + " ".join(prefix_toks)
            else:
                prompt = base_prompt

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            try:
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                             do_sample=False, pad_token_id=tokenizer.pad_token_id,
                                             eos_token_id=end_token_id)
            except Exception as e:
                if is_main:
                    print(f"  Generation failed: {e}")
                if pbar:
                    pbar.update(1)
                continue

            gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
            gen_stmts, valid, gen_plddt = parse_generated_statements(gen_text.split())

            if valid and len(gen_stmts) > 0:
                accum[0] += 1
                if gen_plddt is not None:
                    accum[1] += 1

                v, t = check_atom_validity(gen_stmts, sequence)
                accum[2] += v
                accum[3] += t

                # Count statement types
                gen_contacts = [s for s in gen_stmts if isinstance(s, ContactStatement)]
                gen_distances = [s for s in gen_stmts if isinstance(s, DistanceStatement)]
                accum[4] += len(gen_contacts)
                accum[5] += len(gen_distances)

                # Contact accuracy (position-pair level)
                gt_contact_pairs = {(min(s.pos1, s.pos2), max(s.pos1, s.pos2))
                                    for s in gt_stmts if isinstance(s, ContactStatement)}
                gen_contact_pairs = {(min(s.pos1, s.pos2), max(s.pos1, s.pos2))
                                     for s in gen_contacts}
                accum[6] += len(gen_contact_pairs & gt_contact_pairs)
                accum[7] += len(gen_contact_pairs)
                accum[8] += len(gt_contact_pairs)

                # Distance accuracy
                gt_dist_map = {}
                for s in gt_stmts:
                    if isinstance(s, DistanceStatement):
                        key = (s.pos1, s.pos2, s.atom1, s.atom2)
                        gt_dist_map[key] = float(s.distance_token[1:])  # "d4.5" -> 4.5
                for s in gen_distances:
                    try:
                        gen_d = float(s.distance_token[1:])
                    except (ValueError, IndexError):
                        continue
                    gt_key = (s.pos1, s.pos2, s.atom1, s.atom2)
                    if gt_key in gt_dist_map:
                        accum[10] += 1
                        if abs(gen_d - gt_dist_map[gt_key]) <= 1.0:
                            accum[9] += 1

            if pbar:
                pbar.update(1)

        if world_size > 1:
            dist.all_reduce(accum, op=dist.ReduceOp.SUM)

        a = accum.cpu().tolist()
        label = f"prefix_{n_prefix}"
        results[label] = {
            "pct_valid_grammar": 100 * a[0] / n_samples if n_samples else 0,
            "pct_has_plddt": 100 * a[1] / max(a[0], 1),
            "pct_valid_atoms": 100 * a[2] / a[3] if a[3] else 0,
            "avg_contact_stmts": a[4] / max(a[0], 1),
            "avg_distance_stmts": a[5] / max(a[0], 1),
            "contact_precision": 100 * a[6] / a[7] if a[7] else 0,
            "contact_recall": 100 * a[6] / a[8] if a[8] else 0,
            "distance_accuracy_1A": 100 * a[9] / a[10] if a[10] else 0,
        }

    if pbar:
        pbar.close()
    return results


# ---------------------------------------------------------------------------
# Per-statement-position perplexity
# ---------------------------------------------------------------------------

_PPL_MAX_POSITIONS = 3000


def compute_statement_position_perplexity(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    hf_dataset,
    device: torch.device,
    n_samples: int = 200,
    max_length: int = 8192,
) -> dict[int, tuple[float, int]]:
    """Compute average perplexity at each statement position."""
    model.eval()
    begin_stmt_id = tokenizer.convert_tokens_to_ids("<begin_statements>")
    end_id = tokenizer.convert_tokens_to_ids("<end>")
    contact_type_ids = {tokenizer.convert_tokens_to_ids(t) for t in CONTACT_TYPE_TOKENS}
    distance_id = tokenizer.convert_tokens_to_ids("<distance>")
    plddt_ids = {tokenizer.convert_tokens_to_ids(t) for t in PLDDT_TOKENS}

    rank, world_size = _get_dist_info()
    n_samples = min(n_samples, len(hf_dataset))
    my_indices = list(range(n_samples))[rank::world_size]

    loss_sum = torch.zeros(_PPL_MAX_POSITIONS, dtype=torch.float64, device=device)
    loss_count = torch.zeros(_PPL_MAX_POSITIONS, dtype=torch.float64, device=device)

    for i in my_indices:
        text = hf_dataset[i]["document"]
        enc = tokenizer(text, truncation=True, max_length=max_length, padding=False, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)

        with torch.no_grad():
            logits = model(input_ids=input_ids).logits
        per_token_loss = functional.cross_entropy(logits[0, :-1], input_ids[0, 1:], reduction="none")

        ids = input_ids[0].tolist()
        begin_pos = next((j for j, t in enumerate(ids) if t == begin_stmt_id), None)
        if begin_pos is None:
            continue

        # Walk through statements of variable length
        j = begin_pos + 1
        stmt_idx = 0
        while j < len(ids) and stmt_idx < _PPL_MAX_POSITIONS:
            tid = ids[j]
            if tid == end_id:
                break
            if tid in plddt_ids:
                j += 1
                continue
            if tid in contact_type_ids:
                stmt_len = 3
            elif tid == distance_id:
                stmt_len = 6
            else:
                break

            loss_start = j - 1
            loss_end = loss_start + stmt_len
            if loss_end > len(per_token_loss):
                break
            loss_sum[stmt_idx] += per_token_loss[loss_start:loss_end].mean().item()
            loss_count[stmt_idx] += 1
            stmt_idx += 1
            j += stmt_len

    if world_size > 1:
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(loss_count, op=dist.ReduceOp.SUM)

    result = {}
    for pos in range(_PPL_MAX_POSITIONS):
        count = int(loss_count[pos].item())
        if count > 0:
            result[pos] = (math.exp(loss_sum[pos].item() / count), count)
    return result


# ---------------------------------------------------------------------------
# Subsampled Trainer
# ---------------------------------------------------------------------------


class SubsampledTrainer(Trainer):
    def __init__(self, *args, full_eval_dataset=None, eval_subsample_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._full_eval_dataset = full_eval_dataset
        self._eval_subsample_size = eval_subsample_size

    def evaluate(self, eval_dataset=None, **kwargs):
        if (eval_dataset is None and self._full_eval_dataset is not None
                and self._eval_subsample_size is not None):
            n = min(self._eval_subsample_size, len(self._full_eval_dataset))
            rank, world_size = _get_dist_info()
            if world_size > 1:
                idx_tensor = torch.zeros(n, dtype=torch.long, device=self.args.device)
                if rank == 0:
                    idx_tensor[:] = torch.tensor(
                        np.random.choice(len(self._full_eval_dataset), size=n, replace=False), dtype=torch.long)
                dist.broadcast(idx_tensor, src=0)
                indices = idx_tensor.cpu().tolist()
            else:
                indices = np.random.choice(len(self._full_eval_dataset), size=n, replace=False).tolist()
            eval_dataset = torch.utils.data.Subset(self._full_eval_dataset, indices)
        return super().evaluate(eval_dataset=eval_dataset, **kwargs)


# ---------------------------------------------------------------------------
# Eval callback
# ---------------------------------------------------------------------------


class EvalCallback(TrainerCallback):
    def __init__(self, val_hf_dataset, tokenizer, use_wandb=True, perplexity_samples=200,
                 gen_eval_samples=50, gen_max_new_tokens=8000, max_length=8192):
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

        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        is_fsdp = isinstance(model, FSDP) or isinstance(raw_model, FSDP)
        if is_fsdp:
            if is_main:
                print(f"\n[Step {state.global_step}] Skipping generation eval under FSDP")
            return

        device = next(raw_model.parameters()).device

        example_doc = None
        if is_main:
            example_doc = self._generate_example(raw_model, device)

        ppl_data = compute_statement_position_perplexity(
            model=raw_model, tokenizer=self.tokenizer, hf_dataset=self.val_hf_dataset,
            device=device, n_samples=self.perplexity_samples, max_length=self.max_length)

        gen_results = evaluate_generation(
            model=raw_model, tokenizer=self.tokenizer, hf_dataset=self.val_hf_dataset,
            device=device, n_samples=self.gen_eval_samples, max_new_tokens=self.gen_max_new_tokens)

        if not is_main:
            return

        print(f"\n[Step {state.global_step}] Contacts-and-Distances Evaluation:")
        if ppl_data:
            positions = sorted(ppl_data.keys())
            ppls = [ppl_data[p][0] for p in positions]
            print(f"  Statement positions: {len(positions)}, median PPL: {float(np.median(ppls)):.2f}")

        for label, metrics in gen_results.items():
            print(f"\n  [{label}]")
            for k, v in metrics.items():
                print(f"    {k}: {v:.2f}")

        if example_doc:
            print(f"\n  Generated document:\n{example_doc[:2000]}")

        if not self.use_wandb or wandb.run is None:
            return

        step = state.global_step
        if example_doc:
            wandb.log({"eval_examples/generated_document": wandb.Html(f"<pre>{example_doc[:10000]}</pre>"),
                       "global_step": step})
        if ppl_data:
            positions = sorted(ppl_data.keys())
            ppls = [ppl_data[p][0] for p in positions]
            wandb.log({"eval_examples/median_stmt_ppl": float(np.median(ppls)),
                       "eval_examples/n_stmt_positions": len(positions), "global_step": step})
        for label, metrics in gen_results.items():
            for k, v in metrics.items():
                wandb.log({f"gen_eval/{label}/{k}": v, "global_step": step})

    def _generate_example(self, model, device):
        model.eval()
        idx = np.random.randint(len(self.val_hf_dataset))
        doc = self.val_hf_dataset[int(idx)]["document"]
        _, _, prompt = parse_document(doc)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        end_token_id = self.tokenizer.convert_tokens_to_ids("<end>")
        try:
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=2000, do_sample=False,
                                         pad_token_id=self.tokenizer.pad_token_id,
                                         eos_token_id=end_token_id)
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
    output_dir: str = "outputs/exp6",
    use_wandb: bool = True,
    wandb_project: str = "exp6",
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
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    is_main = _is_main_process()

    torch.manual_seed(seed)
    np.random.seed(seed)

    if is_main:
        print("Setting up tokenizer and model...")
    tokenizer = create_tokenizer()
    model = create_model(vocab_size=len(tokenizer), hidden_size=hidden_size,
                         intermediate_size=intermediate_size, num_layers=num_layers,
                         num_heads=num_heads, num_kv_heads=num_kv_heads)

    if load_weights_only:
        if is_main:
            print(f"Loading weights from {load_weights_only}...")
        state_dict = LlamaForCausalLM.from_pretrained(str(load_weights_only)).state_dict()
        model.load_state_dict(state_dict)
        del state_dict

    n_params = sum(p.numel() for p in model.parameters())
    if is_main:
        print(f"  Parameters: {n_params:,}, Vocab: {len(tokenizer)}")

    if is_main:
        print(f"Loading datasets from {data_dir}...")
    hf_train = load_parquet_dataset(data_dir, train_split)
    hf_val = load_parquet_dataset(data_dir, val_split)

    if max_docs_per_cluster is not None:
        if is_main:
            print(f"  Filtering to max {max_docs_per_cluster} docs per cluster...")
        hf_train = filter_by_cluster_limit(hf_train, max_docs_per_cluster, seed=seed)

    if is_main:
        print(f"  Train: {len(hf_train)}, Val: {len(hf_val)}")

    if train_samples is not None and train_samples < len(hf_train):
        hf_train = hf_train.select(range(train_samples))

    train_dataset = TextDataset(hf_train, tokenizer, max_length=max_token_length)
    val_dataset = TextDataset(hf_val, tokenizer, max_length=max_token_length)

    config = {
        "experiment": "exp6", "task": "contacts_and_distances_v1",
        "data_dir": data_dir, "max_docs_per_cluster": max_docs_per_cluster,
        "train_samples": len(hf_train), "val_samples": len(hf_val),
        "n_params": n_params, "vocab_size": len(tokenizer),
        "hidden_size": model.config.hidden_size, "num_layers": model.config.num_hidden_layers,
    }

    data_collator = DataCollator(pad_token_id=tokenizer.pad_token_id)

    if warmup_steps is None:
        n_devices = max(int(os.environ.get("WORLD_SIZE", 1)), 1)
        steps_per_epoch = math.ceil(len(train_dataset) / (batch_size * n_devices))
        warmup_steps = int(steps_per_epoch * n_epochs * warmup_ratio)

    training_args = TrainingArguments(
        output_dir=str(output_path), num_train_epochs=n_epochs,
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps, learning_rate=lr,
        warmup_steps=warmup_steps, weight_decay=0.01, logging_steps=10,
        eval_strategy="steps", eval_steps=eval_steps, save_strategy="steps",
        save_steps=save_steps, save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False, bf16=True,
        dataloader_num_workers=4, report_to="wandb" if use_wandb else "none",
        run_name=wandb_run_name, seed=seed,
        fsdp=fsdp if fsdp else None, fsdp_config=fsdp_config,
    )

    if use_wandb and is_main:
        wandb.init(project=wandb_project, entity=wandb_entity, name=wandb_run_name, config=config)
        wandb.config.update({"command": " ".join(sys.argv)})

    eval_callback = EvalCallback(
        val_hf_dataset=hf_val, tokenizer=tokenizer, use_wandb=use_wandb,
        perplexity_samples=perplexity_samples, gen_eval_samples=gen_eval_samples,
        gen_max_new_tokens=gen_max_new_tokens, max_length=max_token_length)

    trainer = SubsampledTrainer(
        model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=val_dataset, data_collator=data_collator,
        callbacks=[eval_callback], full_eval_dataset=val_dataset if eval_samples else None,
        eval_subsample_size=eval_samples)

    if is_main:
        print("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    trainer.save_model(str(output_path / "final_model"))
    if is_main:
        tokenizer.save_pretrained(str(output_path / "final_model"))
        with open(output_path / "results.json", "w") as f:
            json.dump({"config": config, "train_loss": train_result.training_loss}, f, indent=2)

    if use_wandb and is_main:
        if wandb.run:
            wandb.run.summary["train_loss"] = train_result.training_loss
        wandb.finish()

    return {"config": config, "train_loss": train_result.training_loss}


def main():
    parser = argparse.ArgumentParser(description="Train contacts-and-distances-v1 LLM")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--max-token-length", type=int, default=8192)
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--eval-samples", type=int, default=None)
    parser.add_argument("--gen-eval-samples", type=int, default=50)
    parser.add_argument("--gen-max-new-tokens", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--n-epochs", type=int, default=3)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/exp6")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="exp6")
    parser.add_argument("--wandb-entity", type=str, default="timodonnell")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--perplexity-samples", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--load-weights-only", type=str, default=None)
    parser.add_argument("--max-docs-per-cluster", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=8192)
    parser.add_argument("--num-layers", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--fsdp", type=str, default="")
    parser.add_argument("--fsdp_config", type=str, default=None)

    args = parser.parse_args()
    resume = args.resume_from_checkpoint
    if resume == "latest":
        resume = True

    train(
        data_dir=args.data_dir, train_split=args.train_split, val_split=args.val_split,
        max_token_length=args.max_token_length, train_samples=args.train_samples,
        eval_samples=args.eval_samples, gen_eval_samples=args.gen_eval_samples,
        gen_max_new_tokens=args.gen_max_new_tokens, batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps, lr=args.lr,
        n_epochs=args.n_epochs, warmup_ratio=args.warmup_ratio, warmup_steps=args.warmup_steps,
        seed=args.seed, output_dir=args.output_dir, use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project, wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name, perplexity_samples=args.perplexity_samples,
        eval_steps=args.eval_steps, save_steps=args.save_steps,
        resume_from_checkpoint=resume, load_weights_only=args.load_weights_only,
        max_docs_per_cluster=args.max_docs_per_cluster,
        hidden_size=args.hidden_size, intermediate_size=args.intermediate_size,
        num_layers=args.num_layers, num_heads=args.num_heads, num_kv_heads=args.num_kv_heads,
        fsdp=args.fsdp, fsdp_config=args.fsdp_config,
    )


if __name__ == "__main__":
    main()
