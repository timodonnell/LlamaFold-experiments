# Learning to Search Through Macromolecular Structure Space

**Proposal for an Open Athena collaboration with Sergey Ovchinnikov’s group at MIT**  
*January 2026*

**Key relevant work from the lab:** ProteinEBM (paper, code)

---

## Table of Contents

- [Introduction](#introduction)
- [Approach](#approach)
- [Proposed Initial Derisking Experiments](#proposed-initial-derisking-experiments)
  - [Experiment 1: Distance Matrix Completion](#experiment-1-distance-matrix-completion)
  - [Experiment 2: Infilling Individual Residues into Complete Structures](#experiment-2-infilling-individual-residues-into-complete-structures)
  - [Experiment 3: Predict Structures from Constraints](#experiment-3-predict-structures-from-constraints)
- [Alternative Approaches](#alternative-approaches)
- [Application Areas and Success Criteria](#application-areas-and-success-criteria)
- [Timeline](#timeline)
- [Resource Requests](#resource-requests)
- [References](#references)

---

## Introduction

The use of machine learning has triggered a revolution in protein science and engineering, with AlphaFold often being described as a “solution” to the problem of understanding how a protein’s folded structure arises from its primary sequence [1]. In reality, many aspects of the problem remain open. One key open question is whether it is possible to accurately predict protein structures without relying on coevolutionary information.

Prior to AlphaFold, a large body of work demonstrated that patterns of covariation in groups of evolutionarily related sequences—multiple sequence alignments (MSAs)—can reveal structural contacts in folded proteins [2,3]. Intuitively, if two residues are in contact, mutations at one position induce selective pressure at the other, leading to detectable covariance.

AlphaFold is heavily reliant on MSAs and generally fails to predict confident structures when MSAs are removed [4]. Although later models use protein language model (PLM) embeddings trained on billions of sequences, there is strong evidence that these models implicitly store MSA-like statistics in their weights and still perform poorly for proteins with few evolutionary relatives [4,5].

Structure prediction without coevolutionary information is practically important. De novo protein design aims to create proteins with no evolutionary history [6–8]. While this has yielded binders, biosensors, and enzymes, designed proteins remain much simpler than evolved ones. A major reason is that designs are filtered by whether AlphaFold confidently predicts the intended fold. Since AlphaFold often fails on complex single-sequence predictions, this constrains the design space [9]. Accurate single-sequence structure prediction would remove this bottleneck.

From a biophysical perspective, Anfinsen’s dogma and energy landscape theory state that proteins fold into structures minimizing free energy, determined by sequence alone [10]. With a sufficiently accurate approximation of this energy function, structure prediction becomes an optimization problem over configuration space.

In earlier work, we showed that AlphaFold can accurately *rank* candidate structures without an MSA when provided templates, even though it cannot predict structures from scratch [11]. This suggests that MSAs help locate minima in an internal scoring function, turning a global search into a local optimization. This insight led to AF2Rank and later ProteinEBM, a fast energy-based model distilled from AlphaFold that approaches its ranking ability while being orders of magnitude faster and useful for mutation-effect prediction and Langevin dynamics refinement [12].

Based on this evidence, we hypothesize that general single-sequence structure prediction can be achieved by scaling computation at test time using a learned agent that searches structure space, guided by scoring functions such as AF2Rank and ProteinEBM—analogous to test-time scaling in large language models.

---

## Approach

Our goal is to develop an agent that intelligently optimizes protein structural hypotheses, guided by an arbitrary scoring or energy model (e.g. ProteinEBM). At each step, given previously observed states and energies, the agent proposes a new structure for evaluation. In parallel, we will improve the energy models themselves, including by training them to discriminate native structures from plausible but incorrect states generated during search.

We propose using a large language model (LLM) architecture for the search task. Advantages include:

- **Mature RL tooling:** Discrete outputs enable established reinforcement learning methods. RL for diffusion models is less mature.
- **Optimized infrastructure:** Transformers benefit from highly optimized training and inference stacks and favorable hardware scaling, especially compared to triangular attention.
- **Flexibility:** Inputs, outputs, and supervision signals can be changed via data engineering rather than architectural redesign.
- **Reduced data concerns:** With high-quality structure predictors available for distillation, data efficiency is less of a concern than in earlier years.

A potential drawback is weaker inductive bias compared to bespoke architectures (e.g. AF3-style pair representations), but this can be mitigated with sufficient data and distillation.

### Representation Choices

One promising search space is **contact maps**, representing residue–residue contacts. The number of contacts scales roughly linearly with protein length, making them token-efficient. Contact maps can be converted to 3D structures using contact-conditioned diffusion models (already trained in ProteinEBM) or used as templates for AF2Rank.

Alternatively, the LLM could directly generate atomic coordinates, as demonstrated for molecules and materials without graph priors [16]. AlphaFold3 shows that transformers can reason over raw Cartesian coordinates [17]. We will empirically compare these approaches.

### Pretraining Strategy

We propose training a ~1B parameter LLM on synthetic data derived from AFDB. Training documents may include:



```
<BEGIN_SEQ>MLFIFFL...<END_SEQ>
<BEGIN_DISTANCES>
<POS_20><POS_30><DISTANCE_BIN_10>
...
<END_DISTANCES>
<BEGIN_COORDINATES>
<POS_55><X_BIN_15><Y_BIN_20><Z_BIN_22>
...
<END_COORDINATES>
<TOTAL_ENERGY_BIN_50>
<BEGIN_PER_POSITION_ENERGIES>
<POS_1><ENERGY_BIN_1>
...
<END_PER_POSITION_ENERGIES>
```


Per-position items are randomly ordered (tagged by index) to avoid enforcing N→C biases.

This setup encourages learning:
1. Energies given structures  
2. Coordinates from sparse constraints  
3. Distance/contact structure from sequence  

Multiple predictions per sequence can be included, ordered from least to most favorable energy, to encourage improvement.

If promising, we will explore:
- **Self-distillation:** Use the model to generate additional states scored by ProteinEBM.
- **Reinforcement learning:** Optimize search policies balancing energy minimization and diversity.
- **Auxiliary tasks:** Secondary structure, clashes, hydrogen bonds, etc.
- **Expanded scope:** Protein complexes, nucleic acids, and small molecules.

---

## Proposed Initial Derisking Experiments

### Experiment 1: Distance Matrix Completion

**Goal:** Test geometric reasoning ability.

- **Task:** Given a partially observed distance matrix for random 3D point clouds, predict missing distances.
- **Evaluation:** Near-zero loss on held-out distances.
- **Extension:** Reconstruct coordinates up to rigid-body transforms.

### Experiment 2: Infilling Individual Residues into Complete Structures

**Goal:** Test binned-coordinate tokenization.

- **Task:** Given a structure with one missing residue, predict its CA coordinate.
- **Evaluation:** Error in Å; success if within a few Å most of the time.
- **Additional:** Test equivariance via rigid-body transforms.

### Experiment 3: Predict Structures from Constraints

**Goal:** Assess end-to-end structure generation.

- **Task:** Given a sequence and sparse distance constraints, predict full CA coordinates.
- **Evaluation:** RMSD after alignment; success if median RMSD < 3 Å on well-folded proteins.
- **Extensions:** Vary constraint sparsity; introduce corrupted constraints.

---

## Alternative Approaches

- **Bespoke architectures:** Adapt AF3-style modules if LLMs underperform.
- **Tool-augmented frontier models:** Use large pretrained models with custom tools, though costly and opaque.

---

## Application Areas and Success Criteria

Early test sets should include:
- Low-MSA-depth proteins known to fold
- De novo proteins that existing predictors fail on
- Fold-switching proteins
- Proteins with multiple conformations

---

## Timeline

**Phase 1 (Feb–Apr 2026)**  
Derisking experiments, data curation, go/no-go on LLM approach.

**Phase 2 (May–Jun 2026)**  
Train full models; integrate with Marin infrastructure.

**Phase 3 (Jul–Sep 2026)**  
Evaluation and post-training.

**Phase 4 (Oct–Dec 2026)**  
Benchmarking and dissemination.

---

## Resource Requests

We request **$30k in Lambda Labs credits**, shared between Open Athena and MIT, for feasibility experiments. Full-scale training is expected to require Google TPU Research Cloud resources.

---

## References

[1] Jumper et al., *Nature* (2021)  
[2] Morcos et al., *PNAS* (2011)  
[3] Ovchinnikov et al., *Science* (2017)  
[4] Lin et al., *Science* (2023)  
[5] Zhang et al., *PNAS* (2024)  
[6] Watson et al., *Nature* (2023)  
[7] Dauparas et al., *Science* (2022)  
[8] Pacesa et al., *Nature* (2025)  
[9] Korbeld et al., *bioRxiv* (2025)  
[10] Anfinsen et al., *PNAS* (1961)  
[11] Roney & Ovchinnikov, *PRL* (2022)  
[12] Roney et al., *bioRxiv* (2025)  
[13] Levine et al., *bioRxiv* (2026)  
[14] Furman, peptide–protein AlphaFold metrics  
[15] Wang et al., *PNAS* (2024)  
[16] Kreiman et al., *arXiv* (2025)  
[17] Abramson et al., *Nature* (2024)

**Protein tokenizers benchmark:** https://arxiv.org/abs/2503.00089
