# Exp4 Evaluation Metrics

All metrics are logged to wandb at each evaluation step (every `--eval-steps` training steps).

## Standard Training Metrics

Logged automatically by the HuggingFace Trainer.

| Wandb key | Description |
|-----------|-------------|
| `train/loss` | Training cross-entropy loss (next-token prediction on full documents). |
| `eval/loss` | Validation cross-entropy loss on a random subset of `--eval-samples` documents (different sample each eval step). |

## Per-Contact-Position Perplexity

Computed on `--perplexity-samples` validation documents (default 200). For each document, runs a forward pass and extracts the per-token cross-entropy loss in the contacts section.

Each contact consists of 4 tokens: `<pX> <pY> <atom1> <atom2>`. The loss for contact position *i* (0-indexed, where 0 = first contact in the document) is the mean cross-entropy over its 4 tokens. Perplexity = exp(mean loss). Results are averaged across documents at each position.

| Wandb key | Description |
|-----------|-------------|
| `eval_examples/first_contact_ppl` | Perplexity for the first contact (position 0), averaged across documents. Measures how predictable the first contact is given the sequence. |
| `eval_examples/median_contact_ppl` | Median perplexity across all contact positions. |
| `eval_examples/last_contact_ppl` | Perplexity at the last contact position observed. |
| `eval_examples/n_contact_positions` | Number of distinct contact positions tracked. |
| `eval_examples/contact_perplexity_plot` | Image with two plots: (1) perplexity vs. contact position, (2) number of documents contributing at each position. Later positions have fewer documents since shorter proteins have fewer contacts. |

## Example Generated Document

| Wandb key | Description |
|-----------|-------------|
| `eval_examples/generated_document` | HTML-rendered text of a document generated from scratch. The model is prompted with `<deterministic-positives-only>` and generates up to 2000 tokens using greedy decoding. Useful for qualitative inspection of whether the model produces syntactically valid protein documents. |

## Generation Evaluation Metrics

Computed on `--gen-eval-samples` validation documents (default 50). For each document, the model is prompted with the sequence (everything up to and including `<begin_contacts>`) and generates up to 8000 tokens with greedy decoding.

All metrics below are computed for four prompt variants:

- **`prefix_0`**: Prompt contains only the sequence. The model must generate all contacts from scratch.
- **`prefix_5`**: Prompt includes the sequence plus the first 5 ground-truth contacts.
- **`prefix_10`**: Prompt includes the first 10 ground-truth contacts.
- **`prefix_20`**: Prompt includes the first 20 ground-truth contacts.

Each metric is logged under `gen_eval/{prefix_label}/{metric_name}`.

### Grammar Validity

| Wandb key (suffix) | Description |
|---------------------|-------------|
| `pct_valid_grammar` | Percentage of generated documents where the output after `<begin_contacts>` consists entirely of well-formed contacts: repeating groups of exactly 4 tokens matching `<pN> <pN> <atom> <atom>`, terminated by `<end_contacts>` or `<end>`. Any token that breaks this pattern (wrong token type, incomplete group) makes the document invalid. |
| `pct_valid_grammar_and_order` | Percentage of generated documents that have valid grammar **and** contacts in valid order. Valid order means the sequence separation `|pos1 - pos2|` is non-increasing across consecutive contacts (i.e., longest-range contacts come first). For prefix prompts, the ordering check spans both the prefix contacts and the generated contacts. |

### Atom Name Validity

| Wandb key (suffix) | Description |
|---------------------|-------------|
| `pct_valid_atoms` | Across all generated documents, the percentage of atom token references that name a valid heavy atom for the residue's amino acid. Each contact has 2 atom references (one per residue), checked against the standard PDB heavy atom set for that amino acid (backbone N/CA/C/O/OXT + side-chain atoms). Positions are 1-indexed: `<p1>` refers to the first residue in the sequence. **Documents with invalid grammar contribute 0 valid atoms** and an estimated denominator of `2 × floor(n_generated_tokens / 4)`. |

### Contact Recall (exact)

| Wandb key (suffix) | Description |
|---------------------|-------------|
| `contact_recall_top_1` | What fraction of each document's **first** ground-truth contact appears in the model's output. Averaged across all documents: `total_found / total_possible`. |
| `contact_recall_top_10` | Same, but considering the first 10 ground-truth contacts per document (or all contacts if the document has fewer than 10). |
| `contact_recall_top_100` | Same, but considering the first 100 ground-truth contacts per document. |

A contact is considered "found" if the exact 4-tuple `(pos1, pos2, atom1, atom2)` appears anywhere in the model's output. For prefix prompts, the prefix contacts are included in the output set (since they are part of the full generated document), so recall for small cutoffs will naturally be higher with larger prefixes.

### Position Recall (ignoring atoms)

| Wandb key (suffix) | Description |
|---------------------|-------------|
| `position_recall_top_1` | Like `contact_recall_top_1`, but a ground-truth contact is considered "found" if **any** generated contact has the same residue pair `(pos1, pos2)`, regardless of which atoms are named. |
| `position_recall_top_10` | Same, for the first 10 ground-truth contacts. |
| `position_recall_top_100` | Same, for the first 100 ground-truth contacts. |

This measures whether the model identifies the correct contacting residue pairs, even if it picks the wrong specific atoms. Position recall will always be ≥ contact recall.

### Example Wandb Keys

Full metric keys follow the pattern `gen_eval/{prefix}/{metric}`:

```
gen_eval/prefix_0/pct_valid_grammar
gen_eval/prefix_0/pct_valid_grammar_and_order
gen_eval/prefix_0/pct_valid_atoms
gen_eval/prefix_0/contact_recall_top_1
gen_eval/prefix_0/contact_recall_top_10
gen_eval/prefix_0/contact_recall_top_100
gen_eval/prefix_0/position_recall_top_1
gen_eval/prefix_0/position_recall_top_10
gen_eval/prefix_0/position_recall_top_100
gen_eval/prefix_5/pct_valid_grammar
gen_eval/prefix_5/pct_valid_grammar_and_order
...
gen_eval/prefix_20/position_recall_top_100
```
