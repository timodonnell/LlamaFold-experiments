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
import sys
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib
import numpy as np
import torch
import torch.nn.functional as functional
import wandb
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from .data import get_all_tokens, load_hf_dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


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


class TextDataset(torch.utils.data.Dataset):
    """Wraps a HuggingFace dataset, tokenizes documents for causal LM training.

    Standard causal LM: labels = input_ids (the model shifts internally).
    Padding positions get label -100 so they don't contribute to loss.
    """

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

        # Crash on unknown tokens (pad token id 0 is also the unk token)
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


def compute_contact_position_perplexity(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    hf_dataset,
    device: torch.device,
    n_samples: int = 200,
    max_length: int = 8192,
) -> dict[int, tuple[float, int]]:
    """Compute average perplexity at each contact position across documents.

    Each contact consists of 4 tokens (pos1, pos2, atom1, atom2). For each
    contact position index (0 = first contact, 1 = second, ...), computes the
    average cross-entropy over those 4 tokens and converts to perplexity.

    Args:
        model: The model in eval mode.
        tokenizer: The tokenizer.
        hf_dataset: HuggingFace dataset with "document" column.
        device: Device to run on.
        n_samples: Number of documents to sample.
        max_length: Max token length for tokenization.

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
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits  # (1, seq_len, vocab_size)

        # Per-token cross-entropy: loss[i] predicts token i+1 from context 0..i
        shift_logits = logits[0, :-1]
        shift_labels = input_ids[0, 1:]
        per_token_loss = functional.cross_entropy(shift_logits, shift_labels, reduction="none")

        # Find contact region boundaries
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
            end_pos = len(ids)  # Document was truncated, no end marker

        # Contact tokens start right after <begin_contacts>
        contact_start = begin_pos + 1
        n_contact_tokens = end_pos - contact_start
        n_contacts = n_contact_tokens // 4

        for c in range(n_contacts):
            # Loss indices for this contact's 4 tokens
            # Token at position p has its prediction loss at per_token_loss[p-1]
            loss_start = contact_start + c * 4 - 1
            loss_end = loss_start + 4
            if loss_end > len(per_token_loss):
                break
            avg_loss = per_token_loss[loss_start:loss_end].mean().item()
            position_losses[c].append(avg_loss)

    # Convert to perplexity
    result: dict[int, tuple[float, int]] = {}
    for pos in sorted(position_losses.keys()):
        losses = position_losses[pos]
        avg_loss = sum(losses) / len(losses)
        result[pos] = (math.exp(avg_loss), len(losses))

    return result


class EvalCallback(TrainerCallback):
    """Callback to generate example documents and compute contact perplexity."""

    def __init__(
        self,
        val_hf_dataset,
        tokenizer: PreTrainedTokenizer,
        use_wandb: bool = True,
        perplexity_samples: int = 200,
        max_length: int = 8192,
    ):
        self.val_hf_dataset = val_hf_dataset
        self.tokenizer = tokenizer
        self.use_wandb = use_wandb
        self.perplexity_samples = perplexity_samples
        self.max_length = max_length
        self._running = False

    def on_evaluate(self, args, state, control, model, **kwargs):
        """Run custom evaluation after each eval step."""
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

        # 2. Compute per-contact-position perplexity
        ppl_data = compute_contact_position_perplexity(
            model=raw_model,
            tokenizer=self.tokenizer,
            hf_dataset=self.val_hf_dataset,
            device=device,
            n_samples=self.perplexity_samples,
            max_length=self.max_length,
        )

        # Print summary to terminal
        print(f"\n[Step {state.global_step}] Contact Prediction Evaluation:")
        if ppl_data:
            positions = sorted(ppl_data.keys())
            first_ppl = ppl_data[positions[0]][0]
            last_ppl = ppl_data[positions[-1]][0]
            ppls = [ppl_data[p][0] for p in positions]
            print(f"  Contact positions tracked: {len(positions)}")
            print(f"  First contact perplexity: {first_ppl:.2f}")
            print(f"  Median contact perplexity: {float(np.median(ppls)):.2f}")
            print(f"  Last contact perplexity: {last_ppl:.2f}")

        if example_doc:
            print("\n  Generated document preview:")
            print(f"  {example_doc[:500]}")
            if len(example_doc) > 500:
                print(f"  ... ({len(example_doc)} chars total)")

        # Log to wandb
        if not self.use_wandb or wandb.run is None:
            return

        # Log example document as HTML for readability
        if example_doc:
            wandb.log(
                {
                    "eval_examples/generated_document": wandb.Html(
                        f"<pre>{example_doc[:10000]}</pre>"
                    ),
                    "global_step": state.global_step,
                }
            )

        # Log per-contact-position perplexity
        if ppl_data:
            positions = sorted(ppl_data.keys())
            perplexities = [ppl_data[p][0] for p in positions]
            counts = [ppl_data[p][1] for p in positions]

            # Create plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            ax1.plot(positions, perplexities, linewidth=0.5)
            ax1.set_xlabel("Contact Position")
            ax1.set_ylabel("Perplexity")
            ax1.set_title(f"Per-Contact-Position Perplexity (Step {state.global_step})")

            ax2.plot(positions, counts, linewidth=0.5, color="orange")
            ax2.set_xlabel("Contact Position")
            ax2.set_ylabel("# Documents")
            ax2.set_title("Documents Contributing per Position")

            plt.tight_layout()
            wandb.log(
                {
                    "eval_examples/contact_perplexity_plot": wandb.Image(fig),
                    "global_step": state.global_step,
                }
            )
            plt.close(fig)

            # Log summary statistics
            wandb.log(
                {
                    "eval_examples/first_contact_ppl": perplexities[0],
                    "eval_examples/median_contact_ppl": float(np.median(perplexities)),
                    "eval_examples/last_contact_ppl": perplexities[-1],
                    "eval_examples/n_contact_positions": len(positions),
                    "global_step": state.global_step,
                }
            )

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


def train(
    dataset_name: str = "timodonnell/protein-docs",
    dataset_config: str = "default",
    train_split: str = "train",
    val_split: str = "validation",
    max_token_length: int = 8192,
    train_samples: int | None = None,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    lr: float = 2e-4,
    n_epochs: int = 3,
    warmup_ratio: float = 0.1,
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
    """Train the contact prediction LLM.

    Args:
        dataset_name: HuggingFace dataset name.
        dataset_config: Dataset configuration/subset name.
        train_split: Name of the training split.
        val_split: Name of the validation split.
        max_token_length: Maximum token sequence length.
        train_samples: Limit training samples (None = all).
        batch_size: Per-device training batch size.
        gradient_accumulation_steps: Gradient accumulation steps.
        lr: Learning rate.
        n_epochs: Number of training epochs.
        warmup_ratio: Warmup ratio.
        seed: Random seed.
        output_dir: Output directory.
        use_wandb: Whether to use wandb logging.
        wandb_project: Wandb project name.
        wandb_entity: Wandb entity/team name.
        wandb_run_name: Wandb run name.
        perplexity_samples: Number of samples for contact perplexity computation.
        eval_steps: Evaluate every N steps.
        save_steps: Save checkpoint every N steps.
        resume_from_checkpoint: Path to checkpoint or True for latest.

    Returns:
        Dictionary with training results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    is_main = _is_main_process()

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Tokenizer and model
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

    # Load datasets
    if is_main:
        print("Loading datasets...")
    hf_train = load_hf_dataset(train_split, dataset_name, dataset_config)
    hf_val = load_hf_dataset(val_split, dataset_name, dataset_config)

    if is_main:
        print(f"  Train: {len(hf_train)} documents")
        print(f"  Val:   {len(hf_val)} documents")

    # Limit training samples if specified
    if train_samples is not None and train_samples < len(hf_train):
        hf_train = hf_train.select(range(train_samples))
        if is_main:
            print(f"  Limited train to: {len(hf_train)} documents")

    # Wrap with tokenization
    train_dataset = TextDataset(hf_train, tokenizer, max_length=max_token_length)
    val_dataset = TextDataset(hf_val, tokenizer, max_length=max_token_length)

    # Config for logging
    config = {
        "experiment": "exp4",
        "task": "contact_prediction",
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "train_samples": len(hf_train),
        "val_samples": len(hf_val),
        "max_token_length": max_token_length,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "lr": lr,
        "n_epochs": n_epochs,
        "warmup_ratio": warmup_ratio,
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
        warmup_ratio=warmup_ratio,
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

    # Initialize wandb (main process only)
    if use_wandb and is_main:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run_name,
            config=config,
        )
        command = " ".join(sys.argv)
        wandb.config.update({"command": command})

    # Create callback for example generation and contact perplexity
    eval_callback = EvalCallback(
        val_hf_dataset=hf_val,
        tokenizer=tokenizer,
        use_wandb=use_wandb,
        perplexity_samples=perplexity_samples,
        max_length=max_token_length,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[eval_callback],
    )

    # Train
    if is_main:
        print("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save model
    trainer.save_model(str(output_path / "final_model"))
    if is_main:
        tokenizer.save_pretrained(str(output_path / "final_model"))

    # Save results
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
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--n-epochs", type=int, default=3)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/exp4")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="exp4")
    parser.add_argument("--wandb-entity", type=str, default="timodonnell")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument(
        "--perplexity-samples",
        type=int,
        default=200,
        help="Number of val samples for contact position perplexity",
    )
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
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr=args.lr,
        n_epochs=args.n_epochs,
        warmup_ratio=args.warmup_ratio,
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
