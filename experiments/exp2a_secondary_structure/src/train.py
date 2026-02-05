"""Training script for secondary structure prediction with 1B Llama.

Trains a ~1B parameter Llama 3.2 architecture from scratch to predict
3-state secondary structure (H/E/C) from CA coordinates.

The model receives sequence + 3D coordinates and learns to output SS labels,
essentially learning what DSSP does algorithmically.
"""

from __future__ import annotations

import argparse
import json
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
    DataCollatorForLanguageModeling,
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
    """Wrapper dataset that returns tokenized text."""

    def __init__(self, dataset: SSDataset, tokenizer: PreTrainedTokenizer, max_length: int = 2048):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

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

        return {
            "input_ids": input_ids,
            "attention_mask": encoding["attention_mask"],
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
        device = next(model.parameters()).device

        # Run evaluation with examples
        eval_results = evaluate_model(
            model=model,
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


def train(
    train_data: str,
    val_data: str,
    test_data: str,
    max_seq_length: int = 500,
    train_samples: int | None = None,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    lr: float = 2e-5,
    n_epochs: int = 3,
    warmup_ratio: float = 0.1,
    seed: int = 42,
    output_dir: str = "outputs/exp2a",
    use_wandb: bool = True,
    wandb_project: str = "ss-prediction",
    wandb_run_name: str | None = None,
    log_examples: int = 5,
    resume_from_checkpoint: str | bool | None = None,
) -> dict[str, Any]:
    """Train the secondary structure prediction LLM.

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

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("Setting up tokenizer and model...")
    tokenizer = create_tokenizer()
    model = create_model(vocab_size=len(tokenizer))

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Log model summary
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_config = model.config

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
    print(f"  Device:               {device}")
    print("=" * 60 + "\n")

    print("Loading datasets...")
    # Load datasets
    train_dataset = SSDataset(train_data, max_length=max_seq_length)
    val_dataset = SSDataset(val_data, max_length=max_seq_length)
    eval_dataset = SSEvalDataset(test_data, max_length=max_seq_length)

    print(f"  Train: {len(train_dataset)} sequences")
    print(f"  Val:   {len(val_dataset)} sequences")
    print(f"  Test:  {len(eval_dataset)} sequences")

    # Limit training samples if specified
    if train_samples is not None and train_samples < len(train_dataset):
        train_dataset.records = train_dataset.records[:train_samples]
        print(f"  Limited train to: {len(train_dataset)} sequences")

    # Wrap with tokenization
    # Token budget: ~14N tokens for N residues:
    #   N AA tokens + 12N coord tokens (4 backbone atoms × 3 coords) + N SS tokens + markers
    max_token_length = max_seq_length * 14 + 20  # Add buffer for markers/newlines
    train_text_dataset = TextDataset(train_dataset, tokenizer, max_length=max_token_length)
    val_text_dataset = TextDataset(val_dataset, tokenizer, max_length=max_token_length)

    # Config for logging
    config = {
        "experiment": "exp2a",
        "task": "secondary_structure_prediction",
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

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

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

    # Initialize wandb
    if use_wandb:
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
    print("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save model
    trainer.save_model(str(output_path / "final_model"))
    tokenizer.save_pretrained(str(output_path / "final_model"))

    # Final evaluation
    print("Evaluating model on test set...")
    device = next(model.parameters()).device
    eval_results = evaluate_model(
        model=model,
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

    # Log to wandb
    if use_wandb:
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

    # Save results
    results = {
        "config": config,
        "train_loss": train_result.training_loss,
        "eval_q3_accuracy": eval_results["q3_accuracy"],
        "eval_class_accuracy_H": eval_results["class_accuracy_H"],
        "eval_class_accuracy_E": eval_results["class_accuracy_E"],
        "eval_class_accuracy_C": eval_results["class_accuracy_C"],
        "eval_pct_valid_syntax": eval_results["pct_valid_syntax"],
        "eval_pct_length_match": eval_results["pct_length_match"],
    }

    with open(output_path / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Train secondary structure prediction LLM")
    parser.add_argument("--train-data", type=str, required=True, help="Path to training JSONL file")
    parser.add_argument("--val-data", type=str, required=True, help="Path to validation JSONL file")
    parser.add_argument("--test-data", type=str, required=True, help="Path to test JSONL file")
    parser.add_argument(
        "--max-seq-length", type=int, default=500, help="Maximum sequence length in residues"
    )
    parser.add_argument("--train-samples", type=int, default=None, help="Limit training samples")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--n-epochs", type=int, default=3)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/exp2a")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="ss-prediction")
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
