"""Contact-level search: sample K candidate contacts at each step, pick best.

Optimized version that properly manages KV cache to avoid full re-encoding.
"""

import math
import sys
import tempfile
import time

import numpy as np
import torch
import torch.nn.functional as F
from biotite.database import rcsb
from biotite.structure import filter_amino_acids
from biotite.structure.io import pdbx
from scipy.spatial import KDTree
from transformers import LlamaForCausalLM

from experiments.exp5_contact_prediction.src.data import (
    AMINO_ACIDS,
    PLDDT_TOKENS,
    VALID_ATOMS,
)
from experiments.exp5_contact_prediction.src.train import (
    create_tokenizer,
    parse_generated_contacts,
)

DEVICE = "cuda"
CKPT = "outputs/exp5.ethereal-galaxy-3/checkpoint-125500"
PDB_ID = "7BNY"
NONSTANDARD = {"MSE": "MET", "CSE": "CYS", "SEC": "CYS", "HYP": "PRO",
               "TPO": "THR", "SEP": "SER", "PTR": "TYR"}

# Load model
tokenizer = create_tokenizer()
model = LlamaForCausalLM.from_pretrained(CKPT, torch_dtype=torch.bfloat16).to(DEVICE).eval()

end_token_id = tokenizer.convert_tokens_to_ids("<end>")
end_contacts_id = tokenizer.convert_tokens_to_ids("<end_contacts>")
begin_contacts_id = tokenizer.convert_tokens_to_ids("<begin_contacts>")
corr_id = tokenizer.convert_tokens_to_ids("<correction>")
non_corr_id = tokenizer.convert_tokens_to_ids("<non-correction>")
plddt_id_set = {tokenizer.convert_tokens_to_ids(t) for t in PLDDT_TOKENS}
plddt_midpoints = {
    tokenizer.convert_tokens_to_ids("<plddt_lt70>"): 65,
    tokenizer.convert_tokens_to_ids("<plddt_70_75>"): 72.5,
    tokenizer.convert_tokens_to_ids("<plddt_75_80>"): 77.5,
    tokenizer.convert_tokens_to_ids("<plddt_80_85>"): 82.5,
    tokenizer.convert_tokens_to_ids("<plddt_85_90>"): 87.5,
    tokenizer.convert_tokens_to_ids("<plddt_90_95>"): 92.5,
    tokenizer.convert_tokens_to_ids("<plddt_95_100>"): 97.5,
}

# Parse 7BNY
path = rcsb.fetch(PDB_ID, "cif", tempfile.gettempdir())
atoms = pdbx.get_structure(pdbx.CIFFile.read(path).block, model=1)
chain = atoms[(atoms.chain_id == atoms.chain_id[0]) & filter_amino_acids(atoms) & (atoms.element != "H")]
unique_res = sorted(set(chain.res_id))
r2p = {rid: i + 1 for i, rid in enumerate(unique_res)}
aa_set = set(AMINO_ACIDS)
sequence = [NONSTANDARD.get(str(chain[chain.res_id == rid].res_name[0]),
            str(chain[chain.res_id == rid].res_name[0])) for rid in unique_res]
seq_len = len(sequence)
all_known = set()
for aa in VALID_ATOMS:
    all_known.update(VALID_ATOMS[aa])
tree = KDTree(chain.coord)
pairs = tree.query_pairs(r=4.0)
best = {}
for i, j in pairs:
    pi, pj = r2p.get(chain.res_id[i]), r2p.get(chain.res_id[j])
    if pi is None or pj is None or abs(pi - pj) < 2:
        continue
    ai, aj = str(chain.atom_name[i]), str(chain.atom_name[j])
    if ai not in all_known or aj not in all_known:
        continue
    if sequence[pi - 1] not in VALID_ATOMS or ai not in VALID_ATOMS[sequence[pi - 1]]:
        continue
    if sequence[pj - 1] not in VALID_ATOMS or aj not in VALID_ATOMS[sequence[pj - 1]]:
        continue
    d = float(np.linalg.norm(chain.coord[i] - chain.coord[j]))
    k = (min(pi, pj), max(pi, pj))
    if k not in best or d < best[k]:
        best[k] = d
gt_pair_set = set(best.keys())
gt_short = {p for p in gt_pair_set if abs(p[0] - p[1]) < 6}
gt_long = {p for p in gt_pair_set if abs(p[0] - p[1]) >= 6}
print(f"{seq_len} residues, {len(gt_pair_set)} GT ({len(gt_short)} short, {len(gt_long)} long)")

seq_tokens = " ".join(f"<{aa}>" for aa in sequence)
base_prompt = f"<random-3-bins> <begin_sequence> {seq_tokens} <begin_contacts>"


def eval_pairs(pred, gt):
    n = len(pred)
    c = len(pred & gt)
    p = c / n if n else 0
    r = c / len(gt) if gt else 0
    f = 2 * p * r / (p + r) if (p + r) else 0
    return p, r, f


def expected_plddt(logits_at_pos):
    probs = torch.softmax(logits_at_pos, dim=0)
    return sum(probs[tid].item() * mid for tid, mid in plddt_midpoints.items())


def contact_level_search(
    prompt,
    max_contacts=350,
    K=16,
    long_range_bonus=0.0,
    plddt_weight=0.0,
    logprob_weight=1.0,
    temperature=1.0,
    rebuild_every=50,
):
    """Generate contacts one at a time with K-candidate scoring.

    Key optimization: save last_logits from the winning candidate's KV cache
    so we don't need to re-encode the full context at every step. Only rebuild
    periodically for numerical stability.
    """
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
    input_ids = enc["input_ids"].to(DEVICE)

    contacts = []

    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
        past = out.past_key_values
        last_logits = out.logits[0, -1]

    prev_plddt = expected_plddt(last_logits)

    for contact_idx in range(max_contacts):
        probs_first = torch.softmax(last_logits, dim=0)
        p_end = probs_first[end_contacts_id].item() + probs_first[end_token_id].item()
        p_plddt = sum(probs_first[tid].item() for tid in plddt_id_set)

        if p_end > 0.5 or (p_end + p_plddt) > 0.7:
            break

        # Handle pLDDT token
        if p_plddt > 0.3:
            plddt_logits = torch.tensor([last_logits[tid].item() for tid in sorted(plddt_id_set)])
            chosen_idx = torch.multinomial(torch.softmax(plddt_logits, dim=0), 1).item()
            chosen_id = sorted(plddt_id_set)[chosen_idx]
            with torch.no_grad():
                out = model(
                    input_ids=torch.tensor([[chosen_id]], device=DEVICE),
                    past_key_values=past,
                    use_cache=True,
                )
                past = out.past_key_values
                last_logits = out.logits[0, -1]
            continue

        # Sample K candidates (each is 6 autoregressive steps from shared cache)
        candidates = []
        for _ in range(K):
            token_ids = []
            total_lp = 0.0
            kv = past
            cur_logits = last_logits

            for step in range(6):
                probs = torch.softmax(cur_logits / temperature, dim=0)
                sampled_id = torch.multinomial(probs, 1).item()
                log_prob = torch.log_softmax(cur_logits, dim=0)[sampled_id].item()
                total_lp += log_prob
                token_ids.append(sampled_id)

                with torch.no_grad():
                    out = model(
                        input_ids=torch.tensor([[sampled_id]], device=DEVICE),
                        past_key_values=kv,
                        use_cache=True,
                    )
                    kv = out.past_key_values
                    cur_logits = out.logits[0, -1]

            tok_strs = [tokenizer.decode([tid]) for tid in token_ids]
            parsed, is_valid, _ = parse_generated_contacts(tok_strs)

            if is_valid and len(parsed) == 1:
                if not any(tid in (end_token_id, end_contacts_id) for tid in token_ids):
                    c = parsed[0]
                    post_plddt = expected_plddt(cur_logits)
                    # Save cur_logits so we can use it as last_logits if this wins
                    candidates.append((token_ids, total_lp, c, post_plddt, kv, cur_logits))

        if not candidates:
            break

        # Score and pick best
        best_score = -float("inf")
        best_idx = 0
        for ci, (token_ids, total_lp, c, post_plddt, kv, cur_logits) in enumerate(candidates):
            _, p1, p2, _, _, bin_tok = c
            sep = abs(p1 - p2)
            score = logprob_weight * total_lp
            if sep >= 6 and long_range_bonus > 0:
                score += long_range_bonus * math.log2(sep / 6)
            if plddt_weight > 0:
                score += plddt_weight * (post_plddt - prev_plddt)
            if score > best_score:
                best_score = score
                best_idx = ci

        _, _, c, post_plddt, best_kv, best_logits = candidates[best_idx]
        contacts.append(c)
        past = best_kv
        last_logits = best_logits  # Key: reuse logits from winning candidate
        prev_plddt = post_plddt

        # Periodic full rebuild for numerical stability
        if rebuild_every > 0 and (contact_idx + 1) % rebuild_every == 0:
            all_toks = [base_prompt]
            for cc in contacts:
                is_corr, p1, p2, a1, a2, bt = cc
                corr = "<correction>" if is_corr else "<non-correction>"
                all_toks.append(f"{corr} <p{p1}> <p{p2}> <{a1}> <{a2}> <{bt}>")
            full_text = " ".join(all_toks)
            enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=8192)
            full_ids = enc["input_ids"].to(DEVICE)
            if full_ids.shape[1] >= 8180:
                break
            with torch.no_grad():
                out = model(input_ids=full_ids, use_cache=True)
                past = out.past_key_values
                last_logits = out.logits[0, -1]
            prev_plddt = expected_plddt(last_logits)
            print(f"    [{contact_idx + 1}] rebuilt cache, {full_ids.shape[1]} tokens")

        if (contact_idx + 1) % 50 == 0:
            lt4 = {(min(c[1], c[2]), max(c[1], c[2])) for c in contacts if c[5] == "bin_lt4"}
            p, r, f = eval_pairs(lt4, gt_pair_set)
            ps, rs, fs = eval_pairs({pp for pp in lt4 if abs(pp[0] - pp[1]) < 6}, gt_short)
            pl, rl, fl = eval_pairs({pp for pp in lt4 if abs(pp[0] - pp[1]) >= 6}, gt_long)
            print(
                f"    [{contact_idx + 1}] {len(contacts)} contacts, {len(lt4)} lt4 | "
                f"all F1={f:.3f} | short F1={fs:.3f} | long F1={fl:.3f}"
            )

    return contacts


def evaluate_contacts(contacts, label):
    lt4 = {(min(c[1], c[2]), max(c[1], c[2])) for c in contacts if c[5] == "bin_lt4"}
    short_pred = {p for p in lt4 if abs(p[0] - p[1]) < 6}
    long_pred = {p for p in lt4 if abs(p[0] - p[1]) >= 6}
    pa, ra, fa = eval_pairs(lt4, gt_pair_set)
    ps, rs, fs = eval_pairs(short_pred, gt_short)
    pl, rl, fl = eval_pairs(long_pred, gt_long)
    bins = {}
    for c in contacts:
        bins[c[5]] = bins.get(c[5], 0) + 1
    n_corr = sum(1 for c in contacts if c[0])
    print(f"{label}: {len(contacts)} contacts ({n_corr} corrections), bins={bins}")
    print(f"  all:   P={pa:.1%} R={ra:.1%} F1={fa:.3f} ({len(lt4)} lt4 pairs)")
    print(f"  short: P={ps:.1%} R={rs:.1%} F1={fs:.3f} ({len(short_pred)} pairs)")
    print(f"  long:  P={pl:.1%} R={rl:.1%} F1={fl:.3f} ({len(long_pred)} pairs)")
    return fa, fs, fl


print("\n" + "=" * 70)
print("Contact-level search experiments")
print("=" * 70)

configs = [
    ("Logprob only K=16", dict(K=16, logprob_weight=1.0, long_range_bonus=0.0, plddt_weight=0.0)),
    ("LR bonus=2 K=16", dict(K=16, logprob_weight=1.0, long_range_bonus=2.0, plddt_weight=0.0)),
    ("LR bonus=5 K=16", dict(K=16, logprob_weight=1.0, long_range_bonus=5.0, plddt_weight=0.0)),
    ("LR=2 + pLDDT K=16", dict(K=16, logprob_weight=1.0, long_range_bonus=2.0, plddt_weight=0.5)),
    ("LR bonus=3 K=32", dict(K=32, logprob_weight=1.0, long_range_bonus=3.0, plddt_weight=0.0)),
]

for label, kwargs in configs:
    print(f"\n--- {label} ---")
    sys.stdout.flush()
    t0 = time.time()
    contacts = contact_level_search(base_prompt, max_contacts=300, rebuild_every=100, **kwargs)
    elapsed = time.time() - t0
    print(f"Time: {elapsed:.0f}s")
    evaluate_contacts(contacts, label)
    sys.stdout.flush()

# Vanilla baseline
print("\n--- Vanilla rollout ---")
enc = tokenizer(base_prompt, return_tensors="pt", truncation=True, max_length=8192)
enc = {k: v.to(DEVICE) for k, v in enc.items()}
with torch.no_grad():
    out = model.generate(
        **enc, max_new_tokens=3440, do_sample=True, temperature=1.0, top_k=0,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=end_token_id,
    )
gen_text = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=False)
vanilla_contacts, _, _ = parse_generated_contacts(gen_text.split())
evaluate_contacts(vanilla_contacts, "Vanilla rollout")
