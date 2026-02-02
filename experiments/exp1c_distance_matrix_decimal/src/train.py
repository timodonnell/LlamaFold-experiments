"""Training script for distance matrix LLM experiment with decimal tokenization.

Trains a ~1B parameter Llama 3.2 architecture from scratch on distance document data.
Distances are represented using three decimal tokens (hundreds, tens, ones).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from scipy import stats
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

from .data import DistanceDataset, EvalDataset, get_special_tokens

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
    """Create a Llama 3.2 1B architecture model from scratch (randomly initialized).

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
        max_position_embeddings=kwargs.get("max_seq_len", 2048),
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

    def __init__(
        self, dataset: DistanceDataset, tokenizer: PreTrainedTokenizer, max_length: int = 2048
    ):
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


def parse_model_output(output: str) -> dict[tuple[int, int], int]:
    """Parse model output to extract predicted distances.

    Handles decimal tokenization where distances are 3 tokens:
    <pX> <pY> <dHHH> <dTT> <dO>

    Args:
        output: Model generated text.

    Returns:
        Dictionary mapping (point_i, point_j) pairs to predicted distances.
    """
    predictions = {}

    # Pattern to match: <pX> <pY> <dHHH> <dTT> <dO>
    # Hundreds: 000, 100, 200, 300, 400, 500
    # Tens: 00, 10, 20, ..., 90
    # Ones: 0, 1, 2, ..., 9
    pattern = r"<p(\d+)>\s*<p(\d+)>\s*<d(\d{3})>\s*<d(\d{2})>\s*<d(\d)>"

    for match in re.finditer(pattern, output):
        i = int(match.group(1))
        j = int(match.group(2))
        hundreds = int(match.group(3))
        tens = int(match.group(4))
        ones = int(match.group(5))
        dist = hundreds + tens + ones

        # Store with canonical key (smaller index first)
        key = (min(i, j), max(i, j))
        predictions[key] = dist

    return predictions


def check_output_structure(generated_text: str, prompt: str) -> dict[str, bool]:
    """Check if the generated output has correct structure.

    Args:
        generated_text: Full generated text (includes prompt echo).
        prompt: The original prompt.

    Returns:
        Dictionary with structure check results.
    """
    # Get only the newly generated part (after the prompt)
    # The prompt ends without <end>, so we look for content after the prompt
    if prompt in generated_text:
        new_content = generated_text[generated_text.find(prompt) + len(prompt) :]
    else:
        new_content = generated_text

    # Check if it ends with <end>
    ends_with_end = "<end>" in new_content

    # Check syntax: everything between <start> and <end> should be valid pairs
    # Valid format: lines of <pX> <pY> <dHHH> <dTT> <dO>
    syntax_valid = True

    # Extract content between markers
    full_content = generated_text
    if "<start>" in full_content:
        start_idx = full_content.find("<start>") + len("<start>")
        end_idx = full_content.find("<end>") if "<end>" in full_content else len(full_content)
        content = full_content[start_idx:end_idx]

        # Each non-empty line should match the pattern
        pair_pattern = r"^<p\d+>\s*<p\d+>\s*<d\d{3}>\s*<d\d{2}>\s*<d\d>$"
        for line in content.strip().split("\n"):
            line = line.strip()
            if line and not re.match(pair_pattern, line):
                syntax_valid = False
                break

    return {
        "ends_with_end": ends_with_end,
        "syntax_valid": syntax_valid,
        "structure_correct": ends_with_end and syntax_valid,
    }


def check_correct_pairs(
    predictions: dict[tuple[int, int], int],
    observed_pairs: list[tuple[int, int]],
    held_out_pairs: list[tuple[int, int]],
) -> dict[str, Any]:
    """Check if the model predicted exactly the right pairs.

    Args:
        predictions: Dictionary of predicted (pair -> distance).
        observed_pairs: List of pairs that were in the prompt.
        held_out_pairs: List of pairs that should be predicted.

    Returns:
        Dictionary with pair accuracy metrics.
    """
    # Get canonical versions of all pairs
    observed_canonical = {(min(i, j), max(i, j)) for i, j in observed_pairs}
    held_out_canonical = {(min(i, j), max(i, j)) for i, j in held_out_pairs}
    predicted_pairs = set(predictions.keys())

    # New pairs are those not in observed
    new_predicted = predicted_pairs - observed_canonical

    # Check if new predictions exactly match held-out pairs
    correct_new_pairs = new_predicted == held_out_canonical

    # How many of the held-out pairs were predicted?
    held_out_predicted = held_out_canonical & predicted_pairs
    held_out_recall = len(held_out_predicted) / len(held_out_canonical) if held_out_canonical else 0

    # How many of the new predictions were actually held-out pairs?
    held_out_precision = (
        len(held_out_predicted & new_predicted) / len(new_predicted) if new_predicted else 0
    )

    return {
        "correct_new_pairs": correct_new_pairs,
        "held_out_recall": held_out_recall,
        "held_out_precision": held_out_precision,
        "n_new_predicted": len(new_predicted),
        "n_held_out": len(held_out_canonical),
        "n_held_out_found": len(held_out_predicted),
    }


def evaluate_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    eval_dataset: EvalDataset,
    device: torch.device,
    n_examples: int = 100,
    log_examples: int = 5,
) -> dict[str, Any]:
    """Evaluate model on held-out distance prediction.

    Args:
        model: The fine-tuned model.
        tokenizer: The tokenizer.
        eval_dataset: Evaluation dataset with observed/held-out splits.
        device: Device to run on.
        n_examples: Number of examples to evaluate.
        log_examples: Number of full examples to log.

    Returns:
        Dictionary with evaluation metrics.
    """
    model.eval()

    all_pred_distances = []
    all_true_distances = []
    logged_examples = []

    # New metrics
    n_correct_pairs = 0
    n_correct_structure = 0
    total_held_out_recall = 0.0
    total_held_out_precision = 0.0

    n_examples = min(n_examples, len(eval_dataset))

    for idx in range(n_examples):
        example = eval_dataset[idx]
        prompt = example["prompt"]

        held_out_pairs = example["held_out_pairs"]
        observed_pairs = example["observed_pairs"]

        # Tokenize and generate
        inputs = tokenizer(prompt, return_tensors="pt")  # type: ignore[misc]
        prompt_length = inputs["input_ids"].shape[1]
        inputs = {k: v.to(device) for k, v in inputs.items()}  # type: ignore[union-attr]

        with torch.no_grad():
            outputs = model.generate(  # type: ignore[operator]
                **inputs,
                max_new_tokens=300,  # More tokens needed for decimal format
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode full output and just the new tokens
        generated_text = str(tokenizer.decode(outputs[0], skip_special_tokens=False))
        new_tokens = outputs[0][prompt_length:]
        new_text = str(tokenizer.decode(new_tokens, skip_special_tokens=False))

        # Parse predictions
        predictions = parse_model_output(generated_text)

        # Check structure
        structure_info = check_output_structure(generated_text, prompt)
        if structure_info["structure_correct"]:
            n_correct_structure += 1

        # Check pairs
        pairs_info = check_correct_pairs(predictions, observed_pairs, held_out_pairs)
        if pairs_info["correct_new_pairs"]:
            n_correct_pairs += 1
        total_held_out_recall += pairs_info["held_out_recall"]
        total_held_out_precision += pairs_info["held_out_precision"]

        # Get true distances for held-out pairs
        held_out_distances = example["held_out_distances"]

        for (i, j), true_dist in zip(held_out_pairs, held_out_distances):
            key = (min(i, j), max(i, j))
            if key in predictions:
                all_pred_distances.append(predictions[key])
                all_true_distances.append(true_dist)

        # Log full examples
        if idx < log_examples:
            logged_examples.append(
                {
                    "prompt": prompt,
                    "generated": generated_text,
                    "new_text": new_text,
                    "prompt_tokens": prompt_length,
                    "held_out_pairs": [(i, j) for i, j in held_out_pairs],
                    "held_out_distances": held_out_distances,
                    "predictions": {f"({k[0]},{k[1]})": v for k, v in predictions.items()},
                    "structure_correct": structure_info["structure_correct"],
                    "correct_new_pairs": pairs_info["correct_new_pairs"],
                    "held_out_recall": pairs_info["held_out_recall"],
                }
            )

    # Compute metrics
    if len(all_pred_distances) > 0:
        pred_arr = np.array(all_pred_distances)
        true_arr = np.array(all_true_distances)

        mae = np.mean(np.abs(pred_arr - true_arr))
        pearson_r, pearson_p = stats.pearsonr(pred_arr, true_arr)
    else:
        mae = float("nan")
        pearson_r = float("nan")
        pearson_p = float("nan")

    return {
        "mae": mae,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "n_predictions": len(all_pred_distances),
        "n_examples": n_examples,
        "logged_examples": logged_examples,
        # New metrics
        "pct_correct_pairs": 100 * n_correct_pairs / n_examples if n_examples > 0 else 0,
        "pct_correct_structure": 100 * n_correct_structure / n_examples if n_examples > 0 else 0,
        "avg_held_out_recall": total_held_out_recall / n_examples if n_examples > 0 else 0,
        "avg_held_out_precision": total_held_out_precision / n_examples if n_examples > 0 else 0,
        # Raw data for plotting
        "all_pred_distances": all_pred_distances,
        "all_true_distances": all_true_distances,
    }


class ExampleLoggingCallback(TrainerCallback):
    """Callback to log example predictions during evaluation."""

    def __init__(
        self,
        eval_dataset: EvalDataset,
        tokenizer: PreTrainedTokenizer,
        log_examples: int = 5,
        use_wandb: bool = True,
    ):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.log_examples = log_examples
        self.use_wandb = use_wandb

    def on_evaluate(self, args, state, control, model, **kwargs):
        """Run example evaluation after each eval step."""
        device = next(model.parameters()).device

        # Run evaluation with examples
        eval_results = evaluate_model(
            model=model,
            tokenizer=self.tokenizer,
            eval_dataset=self.eval_dataset,
            device=device,
            n_examples=min(20, len(self.eval_dataset)),  # Evaluate on subset for speed
            log_examples=self.log_examples,
        )

        # Print to terminal
        print(f"\n[Step {state.global_step}] Example Evaluation:")
        print(f"  MAE: {eval_results['mae']:.4f}")
        print(f"  Pearson r: {eval_results['pearson_r']:.4f}")
        print(f"  N predictions: {eval_results['n_predictions']}")
        print(f"  % correct pairs: {eval_results['pct_correct_pairs']:.1f}%")
        print(f"  % correct structure: {eval_results['pct_correct_structure']:.1f}%")
        print(f"  Avg held-out recall: {eval_results['avg_held_out_recall']:.2f}")

        # Log one example to terminal
        if eval_results["logged_examples"]:
            ex = eval_results["logged_examples"][0]
            print("\n  Sample prediction:")
            print(f"  Prompt tokens: {ex.get('prompt_tokens', 'N/A')}")
            print(f"  NEW TOKENS ONLY: {ex.get('new_text', 'N/A')[:200]}")
            print(f"  Held-out pairs: {ex['held_out_pairs'][:5]}...")
            print(f"  True distances: {ex['held_out_distances'][:5]}...")
            # Show which held-out pairs were predicted
            held_out_canonical = {(min(i, j), max(i, j)) for i, j in ex["held_out_pairs"]}
            matched_preds = {
                k: v
                for k, v in ex["predictions"].items()
                if tuple(map(int, k.strip("()").split(","))) in held_out_canonical
            }
            print(f"  Matched predictions: {matched_preds}")

        # Log to wandb
        if self.use_wandb and wandb.run is not None:
            wandb.log(
                {
                    "eval_examples/mae": eval_results["mae"],
                    "eval_examples/pearson_r": eval_results["pearson_r"],
                    "eval_examples/n_predictions": eval_results["n_predictions"],
                    "eval_examples/pct_correct_pairs": eval_results["pct_correct_pairs"],
                    "eval_examples/pct_correct_structure": eval_results["pct_correct_structure"],
                    "eval_examples/avg_held_out_recall": eval_results["avg_held_out_recall"],
                    "eval_examples/avg_held_out_precision": eval_results["avg_held_out_precision"],
                    "global_step": state.global_step,
                }
            )

            # Log scatterplot of predicted vs true distances
            if eval_results["all_pred_distances"] and eval_results["all_true_distances"]:
                scatter_data = [
                    [true_d, pred_d]
                    for true_d, pred_d in zip(
                        eval_results["all_true_distances"],
                        eval_results["all_pred_distances"],
                    )
                ]
                scatter_table = wandb.Table(
                    data=scatter_data, columns=["true_distance", "predicted_distance"]
                )
                wandb.log(
                    {
                        f"eval_examples/scatter_step_{state.global_step}": wandb.plot.scatter(
                            scatter_table,
                            "true_distance",
                            "predicted_distance",
                            title=f"Predicted vs True Distances (Step {state.global_step})",
                        )
                    }
                )

            # Log examples as a table
            example_table = wandb.Table(
                columns=[
                    "step",
                    "prompt",
                    "prompt_tokens",
                    "new_text",
                    "held_out_pairs",
                    "true_distances",
                    "matched_predictions",
                    "structure_correct",
                    "correct_pairs",
                ]
            )
            for ex in eval_results["logged_examples"]:
                # Get matched predictions for held-out pairs
                held_out_canonical = {(min(i, j), max(i, j)) for i, j in ex["held_out_pairs"]}
                matched = {
                    k: v
                    for k, v in ex["predictions"].items()
                    if tuple(map(int, k.strip("()").split(","))) in held_out_canonical
                }
                example_table.add_data(
                    state.global_step,
                    ex["prompt"],
                    ex.get("prompt_tokens", 0),
                    ex.get("new_text", ""),
                    str(ex["held_out_pairs"]),
                    str(ex["held_out_distances"]),
                    str(matched),
                    ex.get("structure_correct", False),
                    ex.get("correct_new_pairs", False),
                )
            wandb.log({f"eval_examples/examples_step_{state.global_step}": example_table})


def train(
    train_samples: int = 10000,
    val_samples: int = 500,
    eval_samples: int = 100,
    n_points: int = 20,
    coord_range: float = 100.0,
    n_observed: int = 180,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    lr: float = 2e-5,
    n_epochs: int = 3,
    warmup_ratio: float = 0.1,
    seed: int = 42,
    output_dir: str = "outputs/exp1c",
    use_wandb: bool = True,
    wandb_project: str = "distance-llm",
    wandb_run_name: str | None = None,
    log_examples: int = 5,
) -> dict[str, Any]:
    """Train the distance prediction LLM from scratch.

    Args:
        train_samples: Number of training documents.
        val_samples: Number of validation documents.
        eval_samples: Number of evaluation examples.
        n_points: Number of points per document.
        coord_range: Coordinates sampled uniformly from [-coord_range, coord_range].
        n_observed: Number of observed pairs in eval (rest are held out).
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

    Returns:
        Dictionary with training results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Config for logging
    config = {
        "experiment": "exp1c",
        "tokenization": "decimal",
        "train_samples": train_samples,
        "val_samples": val_samples,
        "eval_samples": eval_samples,
        "n_points": n_points,
        "coord_range": coord_range,
        "n_observed": n_observed,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "lr": lr,
        "n_epochs": n_epochs,
        "warmup_ratio": warmup_ratio,
        "seed": seed,
    }

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

    # Add model info to config for wandb
    config["n_params"] = n_params
    config["hidden_size"] = model_config.hidden_size
    config["num_layers"] = model_config.num_hidden_layers
    config["num_heads"] = model_config.num_attention_heads
    config["vocab_size"] = len(tokenizer)

    print("Creating datasets...")
    train_dataset = DistanceDataset(
        size=train_samples,
        n_points=n_points,
        coord_range=coord_range,
        seed=seed,
    )
    val_dataset = DistanceDataset(
        size=val_samples,
        n_points=n_points,
        coord_range=coord_range,
        seed=seed + 100000,
    )
    eval_dataset = EvalDataset(
        size=eval_samples,
        n_points=n_points,
        coord_range=coord_range,
        n_observed=n_observed,
        seed=seed + 200000,
    )

    # Wrap with tokenization
    train_text_dataset = TextDataset(train_dataset, tokenizer)
    val_text_dataset = TextDataset(val_dataset, tokenizer)

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

        # Log model summary
        model_summary = {
            "total_parameters": n_params,
            "trainable_parameters": n_trainable,
            "vocabulary_size": len(tokenizer),
            "hidden_size": model_config.hidden_size,
            "intermediate_size": model_config.intermediate_size,
            "num_layers": model_config.num_hidden_layers,
            "num_attention_heads": model_config.num_attention_heads,
            "num_kv_heads": model_config.num_key_value_heads,
            "max_sequence_length": model_config.max_position_embeddings,
            "device": str(device),
        }
        wandb.config.update({"model_summary": model_summary})

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
    train_result = trainer.train()

    # Save model
    trainer.save_model(str(output_path / "final_model"))
    tokenizer.save_pretrained(str(output_path / "final_model"))

    # Evaluate
    print("Evaluating model...")
    device = next(model.parameters()).device
    eval_results = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        eval_dataset=eval_dataset,
        device=device,
        n_examples=eval_samples,
        log_examples=log_examples,
    )

    print("\nEvaluation Results:")
    print(f"  MAE: {eval_results['mae']:.4f}")
    print(f"  Pearson r: {eval_results['pearson_r']:.4f}")
    print(f"  N predictions: {eval_results['n_predictions']}")
    print(f"  % correct pairs: {eval_results['pct_correct_pairs']:.1f}%")
    print(f"  % correct structure: {eval_results['pct_correct_structure']:.1f}%")
    print(f"  Avg held-out recall: {eval_results['avg_held_out_recall']:.2f}")
    print(f"  Avg held-out precision: {eval_results['avg_held_out_precision']:.2f}")

    # Log examples to terminal
    print(f"\n{'=' * 80}")
    print("LOGGED EXAMPLES")
    print("=" * 80)
    for i, example in enumerate(eval_results["logged_examples"]):
        print(f"\n--- Example {i + 1} ---")
        print(f"PROMPT TOKENS: {example.get('prompt_tokens', 'N/A')}")
        print(f"PROMPT (first 500 chars):\n{example['prompt'][:500]}...")
        print(f"\nNEW TOKENS ONLY:\n{example.get('new_text', 'N/A')}")
        print(f"\nHELD OUT PAIRS: {example['held_out_pairs']}")
        print(f"TRUE DISTANCES: {example['held_out_distances']}")
        print(f"PREDICTIONS: {example['predictions']}")
        print(f"STRUCTURE CORRECT: {example.get('structure_correct', 'N/A')}")
        print(f"CORRECT PAIRS: {example.get('correct_new_pairs', 'N/A')}")
        print("-" * 40)

    # Log to wandb
    if use_wandb:
        wandb.log(
            {
                "eval/mae": eval_results["mae"],
                "eval/pearson_r": eval_results["pearson_r"],
                "eval/n_predictions": eval_results["n_predictions"],
                "eval/pct_correct_pairs": eval_results["pct_correct_pairs"],
                "eval/pct_correct_structure": eval_results["pct_correct_structure"],
                "eval/avg_held_out_recall": eval_results["avg_held_out_recall"],
                "eval/avg_held_out_precision": eval_results["avg_held_out_precision"],
            }
        )

        # Log scatterplot
        if eval_results["all_pred_distances"] and eval_results["all_true_distances"]:
            scatter_data = [
                [true_d, pred_d]
                for true_d, pred_d in zip(
                    eval_results["all_true_distances"],
                    eval_results["all_pred_distances"],
                )
            ]
            scatter_table = wandb.Table(
                data=scatter_data, columns=["true_distance", "predicted_distance"]
            )
            wandb.log(
                {
                    "eval/scatter_final": wandb.plot.scatter(
                        scatter_table,
                        "true_distance",
                        "predicted_distance",
                        title="Final: Predicted vs True Distances",
                    )
                }
            )

        # Log examples as a table
        example_table = wandb.Table(
            columns=[
                "prompt",
                "prompt_tokens",
                "new_text",
                "held_out_pairs",
                "true_distances",
                "predictions",
                "structure_correct",
                "correct_pairs",
            ]
        )
        for example in eval_results["logged_examples"]:
            example_table.add_data(
                example["prompt"],
                example.get("prompt_tokens", 0),
                example.get("new_text", ""),
                str(example["held_out_pairs"]),
                str(example["held_out_distances"]),
                str(example["predictions"]),
                example.get("structure_correct", False),
                example.get("correct_new_pairs", False),
            )
        wandb.log({"eval/examples": example_table})

        if wandb.run is not None:
            wandb.run.summary["eval_mae"] = eval_results["mae"]
            wandb.run.summary["eval_pearson_r"] = eval_results["pearson_r"]
            wandb.run.summary["eval_pct_correct_pairs"] = eval_results["pct_correct_pairs"]
            wandb.run.summary["eval_pct_correct_structure"] = eval_results["pct_correct_structure"]
        wandb.finish()

    # Save results
    results = {
        "config": config,
        "train_loss": train_result.training_loss,
        "eval_mae": eval_results["mae"],
        "eval_pearson_r": eval_results["pearson_r"],
        "eval_n_predictions": eval_results["n_predictions"],
    }

    with open(output_path / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Train distance prediction LLM from scratch")
    parser.add_argument("--train-samples", type=int, default=10000)
    parser.add_argument("--val-samples", type=int, default=500)
    parser.add_argument("--eval-samples", type=int, default=100)
    parser.add_argument("--n-points", type=int, default=20)
    parser.add_argument("--coord-range", type=float, default=100.0)
    parser.add_argument("--n-observed", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--n-epochs", type=int, default=3)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/exp1c")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="distance-llm")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--log-examples", type=int, default=5)

    args = parser.parse_args()

    train(
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        eval_samples=args.eval_samples,
        n_points=args.n_points,
        coord_range=args.coord_range,
        n_observed=args.n_observed,
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
    )


if __name__ == "__main__":
    main()
