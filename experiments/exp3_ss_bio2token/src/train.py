"""Training script for SS prediction with bio2token coordinate encoding.

Trains a ~1B parameter Llama 3.2 architecture from scratch to predict
3-state secondary structure (H/E/C) from bio2token-encoded backbone coordinates.

Bio2token compresses each backbone atom's 3D position into a single discrete
token from a 4096-token codebook, giving 4 tokens per residue (N, CA, C, O)
instead of 12 tokens with naive 1-Angstrom binning.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
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

import wandb

from .data import SSDataset, SSEvalDataset, get_special_tokens

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


def _is_main_process() -> bool:
    """Check if this is the main process in distributed training."""
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    return local_rank in (-1, 0)


def create_tokenizer() -> PreTrainedTokenizerFast:
    """Create a tokenizer from scratch with only our custom tokens."""
    # Get all special tokens
    special_tokens = get_special_tokens()

    # Add newline token and pad/eos tokens
    all_tokens = ["<pad>", "<eos>", "\n"] + special_tokens

    # Create vocabulary mapping
    vocab = {token: idx for idx, token in enumerate(all_tokens)}

    # Create a WordLevel tokenizer
    tokenizer_model = WordLevel(vocab=vocab, unk_token="<pad>")
    tokenizer = Tokenizer(tokenizer_model)

    # Use whitespace + newline splitting as pre-tokenizer
    tokenizer.pre_tokenizer = WhitespaceSplit()

    # Wrap in HuggingFace PreTrainedTokenizerFast
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<pad>",
        pad_token="<pad>",
        eos_token="<eos>",
        bos_token="<start>",
    )

    return hf_tokenizer


def create_model(vocab_size: int, **kwargs) -> LlamaForCausalLM:
    """Create a Llama 3.2 1B architecture model from scratch.

    Args:
        vocab_size: Size of the vocabulary.
        **kwargs: Override default config values.

    Returns:
        Randomly initialized LlamaForCausalLM (~1B parameters).
    """
    # Llama 3.2 1B architecture - approximately 1.2B parameters
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
        bos_token_id=2,  # <start> token
    )

    # Initialize model with random weights
    model = LlamaForCausalLM(config)

    return model


class TextDataset(torch.utils.data.Dataset):
    """Wrapper dataset that returns tokenized text.

    Masks the loss on everything before the <ss> marker so the model
    only trains on predicting secondary structure labels, not coordinates.
    """

    def __init__(self, dataset: SSDataset, tokenizer: PreTrainedTokenizer, max_length: int = 2048):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Get the token ID for <ss> marker
        self.ss_token_id = tokenizer.convert_tokens_to_ids("<ss>")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.tokenizer(
            item["text"],
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        # Crash if any unknown tokens (pad token = 0) are in the input
        input_ids = encoding["input_ids"]
        if 0 in input_ids:  # 0 is the pad/unk token
            # Find which tokens are unknown
            tokens = item["text"].split()
            unk_tokens = [t for t in tokens if self.tokenizer.convert_tokens_to_ids(t) == 0]
            raise ValueError(
                f"Unknown tokens found in input: {unk_tokens[:10]}... "
                f"Text preview: {item['text'][:200]}"
            )

        # Create labels: mask everything before and including <ss> with -100
        # so loss is only computed on the SS prediction tokens
        labels = list(input_ids)
        ss_pos = None
        for i, token_id in enumerate(input_ids):
            if token_id == self.ss_token_id:
                ss_pos = i
        if ss_pos is not None:
            for i in range(ss_pos + 1):
                labels[i] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": encoding["attention_mask"],
            "labels": labels,
        }


def parse_ss_output(generated_text: str) -> list[str] | None:
    """Parse model output to extract predicted secondary structure labels.

    Args:
        generated_text: Full generated text including prompt.

    Returns:
        List of SS labels ["H", "H", "C", "E", ...] or None if malformed.
    """
    # Find the <ss> marker
    ss_idx = generated_text.find("<ss>")
    if ss_idx == -1:
        return None

    # Get content after <ss>
    after_ss = generated_text[ss_idx + len("<ss>") :]

    # Find <end> if present
    end_idx = after_ss.find("<end>")
    if end_idx != -1:
        after_ss = after_ss[:end_idx]

    # Extract SS tokens
    ss_pattern = r"<([HEC])>"
    matches = re.findall(ss_pattern, after_ss)

    if not matches:
        return None

    return matches


def evaluate_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    eval_dataset: SSEvalDataset,
    device: torch.device,
    n_examples: int = 100,
    log_examples: int = 5,
) -> dict[str, Any]:
    """Evaluate model on secondary structure prediction.

    Args:
        model: The trained model.
        tokenizer: The tokenizer.
        eval_dataset: Evaluation dataset.
        device: Device to run on.
        n_examples: Number of examples to evaluate.
        log_examples: Number of full examples to log.

    Returns:
        Dictionary with evaluation metrics.
    """
    model.eval()

    # Counters for Q3 accuracy
    total_residues = 0
    correct_residues = 0

    # Per-class counters
    class_correct = Counter()
    class_total = Counter()

    # Structure validity
    n_valid_syntax = 0
    n_length_match = 0

    logged_examples = []

    n_examples = min(n_examples, len(eval_dataset))

    for idx in range(n_examples):
        example = eval_dataset[idx]
        prompt = example["prompt"]
        true_ss = example["true_ss"]
        seq_length = example["length"]

        # Tokenize and generate
        inputs = tokenizer(prompt, return_tensors="pt")
        prompt_length = inputs["input_ids"].shape[1]
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=seq_length + 5,  # SS tokens + end + buffer
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode
        generated_text = str(tokenizer.decode(outputs[0], skip_special_tokens=False))
        new_tokens = outputs[0][prompt_length:]
        new_text = str(tokenizer.decode(new_tokens, skip_special_tokens=False))

        # Parse predictions
        pred_ss = parse_ss_output(generated_text)

        # Check validity
        is_valid = pred_ss is not None
        length_matches = is_valid and len(pred_ss) == seq_length

        if is_valid:
            n_valid_syntax += 1
        if length_matches:
            n_length_match += 1

            # Compute Q3 accuracy
            for true_label, pred_label in zip(true_ss, pred_ss):
                class_total[true_label] += 1
                total_residues += 1
                if true_label == pred_label:
                    correct_residues += 1
                    class_correct[true_label] += 1

        # Log examples
        if idx < log_examples:
            # Truncate for display
            seq_display = example["sequence"][:50]
            if len(example["sequence"]) > 50:
                seq_display += "..."

            true_ss_display = "".join(true_ss[:50])
            if len(true_ss) > 50:
                true_ss_display += "..."

            if pred_ss:
                pred_ss_display = "".join(pred_ss[:50])
                if len(pred_ss) > 50:
                    pred_ss_display += "..."
            else:
                pred_ss_display = "None"

            logged_examples.append(
                {
                    "id": example["id"],
                    "sequence": seq_display,
                    "true_ss": true_ss_display,
                    "pred_ss": pred_ss_display,
                    "new_text": new_text[:200],
                    "length": seq_length,
                    "pred_length": len(pred_ss) if pred_ss else 0,
                    "is_valid": is_valid,
                    "length_matches": length_matches,
                }
            )

    # Compute metrics
    q3_accuracy = 100 * correct_residues / total_residues if total_residues > 0 else 0.0

    # Per-class accuracy
    class_accuracy = {}
    for cls in ["H", "E", "C"]:
        if class_total[cls] > 0:
            class_accuracy[cls] = 100 * class_correct[cls] / class_total[cls]
        else:
            class_accuracy[cls] = 0.0

    return {
        "q3_accuracy": q3_accuracy,
        "class_accuracy_H": class_accuracy["H"],
        "class_accuracy_E": class_accuracy["E"],
        "class_accuracy_C": class_accuracy["C"],
        "pct_valid_syntax": 100 * n_valid_syntax / n_examples if n_examples > 0 else 0,
        "pct_length_match": 100 * n_length_match / n_examples if n_examples > 0 else 0,
        "total_residues": total_residues,
        "n_examples": n_examples,
        "logged_examples": logged_examples,
    }


class ExampleLoggingCallback(TrainerCallback):
    """Callback to log example predictions during evaluation."""

    def __init__(
        self,
        eval_dataset: SSEvalDataset,
        tokenizer: PreTrainedTokenizer,
        log_examples: int = 5,
        use_wandb: bool = True,
    ):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.log_examples = log_examples
        self.use_wandb = use_wandb
        self._running = False

    def on_evaluate(self, args, state, control, model, **kwargs):
        """Run example evaluation after each eval step."""
        if self._running:
            return
        self._running = True
        try:
            self._do_evaluate(args, state, control, model, **kwargs)
        finally:
            self._running = False

    def _do_evaluate(self, args, state, control, model, **kwargs):
        # Only run on main process
        if not _is_main_process():
            return

        # Unwrap DDP if needed
        raw_model = model.module if hasattr(model, "module") else model
        device = next(raw_model.parameters()).device

        # Run evaluation with examples
        eval_results = evaluate_model(
            model=raw_model,
            tokenizer=self.tokenizer,
            eval_dataset=self.eval_dataset,
            device=device,
            n_examples=min(50, len(self.eval_dataset)),
            log_examples=self.log_examples,
        )

        # Print to terminal
        print(f"\n[Step {state.global_step}] Secondary Structure Evaluation:")
        print(f"  Q3 Accuracy: {eval_results['q3_accuracy']:.2f}%")
        print(f"  Class H: {eval_results['class_accuracy_H']:.2f}%")
        print(f"  Class E: {eval_results['class_accuracy_E']:.2f}%")
        print(f"  Class C: {eval_results['class_accuracy_C']:.2f}%")
        print(f"  % valid syntax: {eval_results['pct_valid_syntax']:.1f}%")
        print(f"  % length match: {eval_results['pct_length_match']:.1f}%")

        # Log one example to terminal
        if eval_results["logged_examples"]:
            ex = eval_results["logged_examples"][0]
            print("\n  Sample prediction:")
            print(f"  ID: {ex['id']}")
            print(f"  Sequence: {ex['sequence']}")
            print(f"  True SS:  {ex['true_ss']}")
            print(f"  Pred SS:  {ex['pred_ss']}")
            print(f"  Raw gen:  {ex['new_text']}")
            print(f"  Length: {ex['length']}, Pred length: {ex['pred_length']}")

        # Log to wandb
        if self.use_wandb and wandb.run is not None:
            wandb.log(
                {
                    "eval_examples/q3_accuracy": eval_results["q3_accuracy"],
                    "eval_examples/class_accuracy_H": eval_results["class_accuracy_H"],
                    "eval_examples/class_accuracy_E": eval_results["class_accuracy_E"],
                    "eval_examples/class_accuracy_C": eval_results["class_accuracy_C"],
                    "eval_examples/pct_valid_syntax": eval_results["pct_valid_syntax"],
                    "eval_examples/pct_length_match": eval_results["pct_length_match"],
                    "global_step": state.global_step,
                }
            )

            # Log examples as a table
            example_table = wandb.Table(
                columns=[
                    "step",
                    "id",
                    "sequence",
                    "true_ss",
                    "pred_ss",
                    "length",
                    "pred_length",
                    "is_valid",
                    "length_matches",
                ]
            )
            for ex in eval_results["logged_examples"]:
                example_table.add_data(
                    state.global_step,
                    ex["id"],
                    ex["sequence"],
                    ex["true_ss"],
                    ex["pred_ss"],
                    ex["length"],
                    ex["pred_length"],
                    ex["is_valid"],
                    ex["length_matches"],
                )
            wandb.log({f"eval_examples/examples_step_{state.global_step}": example_table})


class SSDataCollator:
    """Data collator that pads input_ids, attention_mask, and custom labels."""

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


def train(
    train_data: str,
    val_data: str,
    test_data: str,
    max_seq_length: int = 1300,
    train_samples: int | None = None,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    lr: float = 2e-5,
    n_epochs: int = 3,
    warmup_ratio: float = 0.1,
    seed: int = 42,
    output_dir: str = "outputs/exp3",
    use_wandb: bool = True,
    wandb_project: str = "ss-prediction-bio2token",
    wandb_run_name: str | None = None,
    log_examples: int = 5,
    resume_from_checkpoint: str | bool | None = None,
) -> dict[str, Any]:
    """Train the secondary structure prediction LLM with bio2token coordinates.

    Args:
        train_data: Path to training JSONL file.
        val_data: Path to validation JSONL file.
        test_data: Path to test JSONL file.
        max_seq_length: Maximum sequence length in residues.
        train_samples: Limit training samples (None = all).
        batch_size: Training batch size per device.
        gradient_accumulation_steps: Gradient accumulation steps.
        lr: Learning rate.
        n_epochs: Number of training epochs.
        warmup_ratio: Warmup ratio.
        seed: Random seed.
        output_dir: Output directory.
        use_wandb: Whether to use wandb logging.
        wandb_project: Wandb project name.
        wandb_run_name: Wandb run name.
        log_examples: Number of examples to log in full.
        resume_from_checkpoint: Path to checkpoint or True to resume from latest.

    Returns:
        Dictionary with training results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    is_main = _is_main_process()

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    if is_main:
        print("Setting up tokenizer and model...")
    tokenizer = create_tokenizer()
    model = create_model(vocab_size=len(tokenizer))

    # Don't manually move to device — Trainer handles this for DDP

    # Log model summary
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
    # Load datasets
    train_dataset = SSDataset(train_data, max_length=max_seq_length)
    val_dataset = SSDataset(val_data, max_length=max_seq_length)
    eval_dataset = SSEvalDataset(test_data, max_length=max_seq_length)

    if is_main:
        print(f"  Train: {len(train_dataset)} sequences")
        print(f"  Val:   {len(val_dataset)} sequences")
        print(f"  Test:  {len(eval_dataset)} sequences")

    # Limit training samples if specified
    if train_samples is not None and train_samples < len(train_dataset):
        train_dataset.offsets = train_dataset.offsets[:train_samples]
        if is_main:
            print(f"  Limited train to: {len(train_dataset)} sequences")

    # Wrap with tokenization
    # Token budget: ~6N tokens for N residues:
    #   N AA tokens + 4N bio2token tokens (4 backbone atoms × 1 token each) + N SS tokens + markers
    max_token_length = max_seq_length * 6 + 20  # Add buffer for markers/newlines
    train_text_dataset = TextDataset(train_dataset, tokenizer, max_length=max_token_length)
    val_text_dataset = TextDataset(val_dataset, tokenizer, max_length=max_token_length)

    # Config for logging
    config = {
        "experiment": "exp3",
        "task": "secondary_structure_prediction_bio2token",
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data,
        "max_seq_length": max_seq_length,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(eval_dataset),
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

    # Data collator that pads and preserves our custom labels
    data_collator = SSDataCollator(pad_token_id=tokenizer.pad_token_id)

    # Training arguments
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
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
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
            name=wandb_run_name,
            config=config,
        )

        # Log training command
        command = " ".join(sys.argv)
        wandb.config.update({"command": command})

    # Create example logging callback
    example_callback = ExampleLoggingCallback(
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        log_examples=log_examples,
        use_wandb=use_wandb,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_text_dataset,
        eval_dataset=val_text_dataset,
        data_collator=data_collator,
        callbacks=[example_callback],
    )

    # Train
    if is_main:
        print("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save model (Trainer handles main-process-only saving)
    trainer.save_model(str(output_path / "final_model"))
    if is_main:
        tokenizer.save_pretrained(str(output_path / "final_model"))

    # Final evaluation (main process only)
    eval_results = None
    if is_main:
        print("Evaluating model on test set...")
        raw_model = model.module if hasattr(model, "module") else model
        device = next(raw_model.parameters()).device
        eval_results = evaluate_model(
            model=raw_model,
            tokenizer=tokenizer,
            eval_dataset=eval_dataset,
            device=device,
            n_examples=len(eval_dataset),
            log_examples=log_examples,
        )

        print("\nFinal Evaluation Results:")
        print(f"  Q3 Accuracy: {eval_results['q3_accuracy']:.2f}%")
        print(f"  Class H: {eval_results['class_accuracy_H']:.2f}%")
        print(f"  Class E: {eval_results['class_accuracy_E']:.2f}%")
        print(f"  Class C: {eval_results['class_accuracy_C']:.2f}%")
        print(f"  % valid syntax: {eval_results['pct_valid_syntax']:.1f}%")
        print(f"  % length match: {eval_results['pct_length_match']:.1f}%")
        print(f"  Total residues evaluated: {eval_results['total_residues']}")

        # Log examples to terminal
        print(f"\n{'=' * 80}")
        print("LOGGED EXAMPLES")
        print("=" * 80)
        for i, example in enumerate(eval_results["logged_examples"]):
            print(f"\n--- Example {i + 1} ---")
            print(f"ID: {example['id']}")
            print(f"Sequence: {example['sequence']}")
            print(f"True SS:  {example['true_ss']}")
            print(f"Pred SS:  {example['pred_ss']}")
            print(f"Length: {example['length']}, Pred length: {example['pred_length']}")
            print(f"Valid: {example['is_valid']}, Length match: {example['length_matches']}")
            print("-" * 40)

    # Log to wandb (main process only)
    if use_wandb and is_main and eval_results is not None:
        wandb.log(
            {
                "eval/q3_accuracy": eval_results["q3_accuracy"],
                "eval/class_accuracy_H": eval_results["class_accuracy_H"],
                "eval/class_accuracy_E": eval_results["class_accuracy_E"],
                "eval/class_accuracy_C": eval_results["class_accuracy_C"],
                "eval/pct_valid_syntax": eval_results["pct_valid_syntax"],
                "eval/pct_length_match": eval_results["pct_length_match"],
                "eval/total_residues": eval_results["total_residues"],
            }
        )

        # Log examples as a table
        example_table = wandb.Table(
            columns=[
                "id",
                "sequence",
                "true_ss",
                "pred_ss",
                "length",
                "pred_length",
                "is_valid",
                "length_matches",
            ]
        )
        for ex in eval_results["logged_examples"]:
            example_table.add_data(
                ex["id"],
                ex["sequence"],
                ex["true_ss"],
                ex["pred_ss"],
                ex["length"],
                ex["pred_length"],
                ex["is_valid"],
                ex["length_matches"],
            )
        wandb.log({"eval/examples": example_table})

        if wandb.run is not None:
            wandb.run.summary["eval_q3_accuracy"] = eval_results["q3_accuracy"]
            wandb.run.summary["eval_class_accuracy_H"] = eval_results["class_accuracy_H"]
            wandb.run.summary["eval_class_accuracy_E"] = eval_results["class_accuracy_E"]
            wandb.run.summary["eval_class_accuracy_C"] = eval_results["class_accuracy_C"]
            wandb.run.summary["eval_pct_valid_syntax"] = eval_results["pct_valid_syntax"]
            wandb.run.summary["eval_pct_length_match"] = eval_results["pct_length_match"]
        wandb.finish()

    # Save results (main process only)
    results = {"config": config, "train_loss": train_result.training_loss}
    if is_main and eval_results is not None:
        results.update(
            {
                "eval_q3_accuracy": eval_results["q3_accuracy"],
                "eval_class_accuracy_H": eval_results["class_accuracy_H"],
                "eval_class_accuracy_E": eval_results["class_accuracy_E"],
                "eval_class_accuracy_C": eval_results["class_accuracy_C"],
                "eval_pct_valid_syntax": eval_results["pct_valid_syntax"],
                "eval_pct_length_match": eval_results["pct_length_match"],
            }
        )
        with open(output_path / "results.json", "w") as f:
            json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train SS prediction LLM with bio2token coordinates"
    )
    parser.add_argument("--train-data", type=str, required=True, help="Path to training JSONL file")
    parser.add_argument("--val-data", type=str, required=True, help="Path to validation JSONL file")
    parser.add_argument("--test-data", type=str, required=True, help="Path to test JSONL file")
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=1300,
        help="Maximum sequence length in residues",
    )
    parser.add_argument("--train-samples", type=int, default=None, help="Limit training samples")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--n-epochs", type=int, default=3)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/exp3")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="ss-prediction-bio2token")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--log-examples", type=int, default=5)
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory, or 'latest' to resume from the latest checkpoint",
    )

    args = parser.parse_args()

    resume = args.resume_from_checkpoint
    if resume == "latest":
        resume = True

    train(
        train_data=args.train_data,
        val_data=args.val_data,
        test_data=args.test_data,
        max_seq_length=args.max_seq_length,
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
        wandb_run_name=args.wandb_run_name,
        log_examples=args.log_examples,
        resume_from_checkpoint=resume,
    )


if __name__ == "__main__":
    main()
