# Experimental Design: High-Density Pre-training for Enhanced LLM Capabilities

**Version:** 1.0
**Date:** Friday, April 11, 2025

**1. Introduction**

**1.1. Project Goal & Hypothesis**
This document outlines the experimental design for investigating the impact of pre-training data quality on Large Language Model (LLM) capabilities. Inspired by the results observed with models like Microsoft's Phi series, the central hypothesis is that **a meticulously curated, high-density pre-training dataset can significantly enhance foundational model capabilities (e.g., reasoning, factuality, instruction following), potentially reducing the need for extensive downstream fine-tuning or yielding superior performance compared to models of similar size trained on standard, large-scale web corpora.**

**1.2. Context and Constraints**
The experiment aims to explore the trade-offs between data quality, model scale, and computational cost. It operates under significant resource constraints, being self-funded without organizational or grant support. This necessitates a data-centric approach, prioritizing information density and learning efficiency over sheer model or dataset scale.

**1.3. Document Purpose**
This document consolidates the refined experimental plan, incorporating decisions made regarding dataset structure, model architecture, training methodology, evaluation strategy, and interpretability analysis. It serves as a reference for the project's design and execution.

**2. Dataset Design: High-Density, Multi-Dimensional Corpus**

**2.1. Core Philosophy**
The foundation of the experiment is a pre-training dataset designed for maximal "educational value" and information density, diverging from standard large-scale web scrapes. The dataset will be structured along multiple orthogonal dimensions to ensure comprehensive coverage of targeted skills and knowledge domains.

**2.2. Refined Dataset Dimensions**
Based on an initial proposal and refinement focusing on learnability, orthogonality, and operationalization for unsupervised pre-training, the following 9 dimensions will guide dataset curation and generation:

1.  **Content Focus (Implicit Goal):** Defines the primary purpose or type of information conveyed.
2.  **Format & Structure:** Specifies the structural layout of the text.
3.  **Domain Emphasis:** Ensures coverage across key knowledge areas.
4.  **Reasoning Explicitness:** Controls the degree to which reasoning steps are overtly presented.
5.  **Conceptual Complexity:** Ranges the difficulty and interconnectedness of ideas.
6.  **Ambiguity Level:** Varies the degree of certainty or potential for multiple interpretations.
7.  **Factual Density:** Measures the concentration of verifiable information.
8.  **Temporal Focus:** Distinguishes between timeless and time-sensitive content.
9.  **Emotional Valence & Style:** Captures the tone and stylistic expression.

**2.3. Target Distribution Across Dimensions**
The dataset composition will aim for specific distributions across the subcategories of each dimension to prioritize reasoning, structured knowledge, and reflection. The target percentages are outlined below:

-  **Table 1: Content Focus Distribution**

| Subcategory                       | Proposed % |
|:----------------------------------|:-----------|
| Factual Knowledge & Explanation   | 30%        |
| Procedural & Task-Oriented        | 20%        |
| Reasoning & Logic Exposition      | 20%        |
| Argumentation & Multi-Perspective | 15%        |
| Creative & Narrative Expression   | 10%        |
| Meta-Cognitive & Self-Correction  | 5%         |
| **Total**                         | **100%**   |

-  **Table 2: Format & Structure Distribution**

| Subcategory                          | Proposed % |
|:-------------------------------------|:-----------|
| Structured Prose                     | 30%        |
| Dialogue/Transcript                  | 15%        |
| Instructional/Step-by-Step           | 15%        |
| Question-Answer (Direct)             | 10%        |
| Question-Answer (with Rationale/CoT) | 25%        |
| Fragmented/Note-like                 | 5%         |
| **Total**                            | **100%**   |

-  **Table 3: Domain Emphasis Distribution**

| Subcategory                     | Proposed % |
|:--------------------------------|:-----------|
| STEM                            | 35%        |
| Humanities & Social Sciences    | 25%        |
| Practical & Applied             | 15%        |
| Creative Arts & Literature      | 10%        |
| Abstract Reasoning & Philosophy | 15%        |
| **Total**                       | **100%**   |

-  **Table 4: Reasoning Explicitness Distribution**

| Subcategory                     | Proposed % |
|:--------------------------------|:-----------|
| Implicit Reasoning              | 20%        |
| Concise Reasoning               | 25%        |
| Detailed Chain-of-Thought (CoT) | 30%        |
| Self-Reflective/Corrective      | 10%        |
| Probabilistic Reasoning         | 15%        |
| **Total**                       | **100%**   |

-  **Table 5: Conceptual Complexity Distribution**

| Subcategory                | Proposed % |
|:---------------------------|:-----------|
| Basic Concepts             | 30%        |
| Intermediate Concepts      | 40%        |
| Advanced/Interdisciplinary | 30%        |
| **Total**                  | **100%**   |

-  **Table 6: Ambiguity Level Distribution**

| Subcategory        | Proposed % |
|:-------------------|:-----------|
| Low Ambiguity      | 40%        |
| Moderate Ambiguity | 35%        |
| High Ambiguity     | 25%        |
| **Total**          | **100%**   |

-  **Table 7: Factual Density Distribution**

| Subcategory      | Proposed % |
|:-----------------|:-----------|
| Low Density      | 20%        |
| Moderate Density | 40%        |
| High Density     | 40%        |
| **Total**        | **100%**   |

-  **Table 8: Temporal Focus Distribution**

| Subcategory           | Proposed % |
|:----------------------|:-----------|
| Timeless/Foundational | 70%        |
| Contemporary/Dynamic  | 30%        |
| **Total**             | **100%**   |

-  **Table 9: Emotional Valence & Style Distribution**

| Subcategory           | Proposed % |
|:----------------------|:-----------|
| Neutral/Objective     | 50%        |
| Positive/Constructive | 15%        |
| Negative/Critical     | 10%        |
| Empathetic/Personal   | 15%        |
| Creative/Figurative   | 10%        |
| **Total**             | **100%**   |

**2.4. Data Generation and Curation**
The dataset will be constructed using a blend of:
-  **Synthetic Data Generation:** Leveraging state-of-the-art LLMs (e.g., GPT-4, Claude 3) prompted to generate examples matching specific dimensional combinations according to the target distributions.
-  **High-Quality Human Data:** Incorporating carefully selected and filtered human-written text (e.g., textbooks, scientific papers, curated web content) that exemplify the target dimensions (aiming for ~20-40% human data).
-  **Quality Control:** Implementing rigorous automated filtering (e.g., scoring, perplexity, deduplication) and targeted human review to ensure data quality and alignment with the dimensional framework.

**3. Model Architecture**

**3.1. Target Model Size: ~1-1.5 Billion Parameters**
The experiment will utilize a model with approximately 1.3 billion parameters. This size is chosen as a strategic balance between:
-  **Feasibility:** Significantly lower estimated training costs compared to larger models (3B, 7B+), making it viable within self-funded constraints, especially given the potentially high token counts needed for the dense data strategy (see Section 4.2).
-  **Experimental Validity:** Sufficiently large to potentially demonstrate complex emergent behaviors and the effects of the high-density data, providing a meaningful testbed for the hypothesis.
-  **Risk Management:** Mitigates financial risk associated with the uncertainty surrounding the optimal token count for the novel dataset.
-  **Iterative Potential:** Allows for faster training and evaluation cycles compared to larger models.

**3.2. Core Architecture: Multi-head Latent Attention (MLA)**
The primary model architecture will employ Multi-head Latent Attention (MLA), as pioneered by DeepSeek.
-  **Rationale:**
    -  *Memory Efficiency:* MLA significantly reduces the KV cache size compared to MHA/GQA by compressing K/V pairs into a latent vector before caching. This is crucial for maximizing resource utilization on constrained hardware, potentially enabling larger batch sizes or longer sequence lengths.
    -  *Potential Expressiveness:* Theoretical arguments and some empirical results suggest MLA may offer greater expressiveness than GQA for equivalent cache overhead, potentially breaking the typical efficiency-performance trade-off.
    -  *Data Synergy:* The compression mechanism might interact favorably with the potentially lower intrinsic dimensionality of structured, high-density data.
-  **Challenges:** Implementation complexity (especially RoPE integration), potential lack of highly optimized kernels (e.g., Flash Attention incompatibility), and relative novelty compared to GQA.
-  **Architectural Parameters (for N ≈ 1.3B):**
    -  Hidden Dimension (`d_model`): 2048 (calculated as `128 * round(sqrt(1.3e9)/128)`)
    -  Layers (`n_layers`): 26 (calculated as `2 * round(sqrt(1.3e9)/128)`)
    -  Attention Heads (`n_heads`): 16 (calculated as `max(8, 2048/128)`)
    -  FF Dimension (`d_ff`): 8192 (`4 * d_model`)
    -  MLA Latent Dimension (`d_c`): 512 (`d_model / 4`)
    -  Head Dim (`d_head`): 128 (`d_model / n_heads`)
    -  RoPE Dim (`d_rope`): 32 (`d_head // 4`)
    -  Compressed Head Dim (`d_head_comp`): 96 (`d_head - d_rope`)

**3.3. Baseline Architecture: Grouped-Query Attention (GQA)**
To isolate the effects of MLA, a baseline model will be trained under identical conditions (same parameters, data, hyperparameters) but using Grouped-Query Attention (GQA).
-  **GQA Groups:** A value like 4 or 8 KV groups will be chosen for the 16 attention heads (e.g., 4 groups = 4 KV heads; 8 groups = 2 KV heads).

**4. Training Methodology**

**4.1. Training Duration Metric: Total Tokens Processed**
Progress and duration will be measured by the **total number of tokens processed**, not epochs. This aligns with standard LLM pre-training practices and scaling laws, reflecting the volume of data exposure rather than passes over a potentially repeating dataset.

**4.2. Target Token Count**
The optimal token count for the high-density dataset is unknown. Based on precedents like the Phi models and the goal of observing significant effects, the target range is substantially higher than the Chinchilla-optimal ratio (~20 tokens/param). Training will aim for **at least 100-300 billion tokens**, with the possibility of extending further towards ~1 trillion tokens if budget and observed progress permit.

**4.3. Batch Size Determination**
The goal is to achieve a large **target global token batch size** (e.g., 2M-4M tokens) per optimizer step for stability. This will be approximated using the standard methodology:
1.  Determine max sequences per GPU (`per_device_batch_size_examples`) empirically based on memory limits (e.g., 4 sequences on 80GB GPU for a ~1.3B model).
2.  Calculate `avg_sequence_length` from the dataset.
3.  Estimate `avg_tokens_per_pass_per_device = per_device_batch_size_examples * avg_sequence_length`.
4.  Calculate `avg_tokens_per_pass_global = avg_tokens_per_pass_per_device * num_gpus`.
5.  Determine `gradient_accumulation_steps = target_global_token_batch_size // avg_tokens_per_pass_global` (adjusting as needed).
    -  *Note:* Token count per step will fluctuate around the target due to variable sequence lengths; this is expected and managed by averaging over time.

**4.4. Learning Rate Scheduler**
A cosine decay schedule with linear warmup is recommended. The total number of steps for the scheduler (`total_scheduler_steps`) will be calculated *once* before training based on the `target_tokens` and the estimated `tokens_per_optimizer_step`:
`total_scheduler_steps = target_tokens // (train_dataset.total_tokens / (len(train_dataloader) // gradient_accumulation_steps))`

**4.5. Starting Hyperparameters (for ~1.3B Model)**

| Hyperparameter        | Proposed Starting Value     | Notes                                           |
|:----------------------|:----------------------------|:------------------------------------------------|
| Optimizer             | AdamW                       | Standard                                        |
| Peak Learning Rate    | 3e-4                        | Mid-range, adjust based on stability/batch size |
| LR Schedule           | Cosine decay w/ Warmup      | Warmup: ~2k-3k steps; Min LR: 10% of peak       |
| Global Batch Size     | 2 Million tokens            | Target token count per optimizer step           |
| Weight Decay          | 0.1                         | Common value                                    |
| AdamW Betas           | β₁=0.9, β₂=0.95             | Common practice                                 |
| Grad Clipping         | 1.0 (global norm)           | Essential for stability                         |
| Sequence Length       | 2048 tokens                 | Start here, potentially increase later          |
| Initialization        | Normal, std ≈ 0.02 (scaled) | Standard practice                               |
| Dropout               | 0.0                         | Often omitted in pre-training                   |
| Precision             | BF16                        | Preferred for stability if supported            |
| MLA `d_c`             | 512                         | Based on user spec (`d_model/4`)                |
| MLA RoPE Handling     | Decoupled                   | Requires careful implementation                 |
| GQA Groups (Baseline) | 4 or 8                      | For the GQA baseline model                      |

**4.6. Curriculum Learning**
Consider implementing a curriculum, potentially involving:
-  **Sequence Length Scheduling:** Start with shorter sequences (e.g., 1024 or 2048) and increase to the final target (e.g., 4096) later in training.
-  **Data Complexity Scheduling:** Introduce simpler concepts or formats before more complex ones.

**4.7. Infrastructure and Checkpointing**
-  **Infrastructure:** Define GPU type, count, interconnect, software stack (PyTorch, DeepSpeed/FSDP), and MLA implementation details.
-  **Checkpointing:** Implement frequent, robust checkpointing (saving model weights, optimizer/scheduler states, RNG states, step count) to allow exact resumption and evaluation.

**5. Evaluation Strategy**

**5.1. Iterative Evaluation at Token Milestones**
Evaluate model checkpoints at predefined token milestones (e.g., 50B, 100B, 200B, 400B tokens) rather than epoch boundaries.

**5.2. Core Metrics**
-  **During Training (Frequent Logging):** Training Loss, Validation Loss, Global Gradient Norm (before/after clipping), Learning Rate, Throughput (Tokens/sec, MFU), Memory Usage. Use tools like TensorBoard or W&B.
-  **At Checkpoints (Comprehensive Evaluation):**
    -  Validation Loss (on held-out pre-training data).
    -  Downstream Task Performance (Zero-shot / Few-shot).

**5.3. Downstream Task Selection**
Choose a diverse suite of benchmarks reflecting the goals of the high-density data and general capabilities:
-  **Reasoning:** GSM8K, MATH, Big-Bench Hard subsets.[16]
-  **Coding:** HumanEval, MBPP (if relevant to data).
-  **General Understanding:** MMLU, HellaSwag, ARC-Challenge.[17]
-  **Factuality:** Benchmarks like TruthfulQA or custom probes relevant to dataset domains.[10, 17]
-  **Domain-Specific:** Tasks directly related to the curated data domains.

**5.4. Comparative Analysis**
-  **MLA vs. GQA:** Rigorously compare the MLA model against the GQA baseline on all metrics at each checkpoint.
-  **External Baselines:** Compare against publicly available models of similar size (~1.3B) trained on standard datasets (e.g., Pythia-1.4B, OLMo-1B [14, 15]) to quantify the impact of the high-density data strategy.

**5.5. Training Management Criteria**
Use evaluation results to decide whether to:
-  **Continue:** If validation loss and downstream performance show significant improvement.
-  **Adjust/Monitor:** If validation loss plateaus but downstream tasks improve, or if instability occurs.
-  **Stop:** Based on diminishing returns (minimal improvement checkpoint-to-checkpoint), budget exhaustion, or achieving target performance.

**6. Interpretability Plan**

**6.1. Hypothesis**
Investigate whether the high-density, structured pre-training data leads to more interpretable internal model representations compared to standard training.

**6.2. Techniques**
-  **Attention Analysis:** Visualize and compare attention patterns (MLA vs. GQA).
-  **Probe Tasks:** Train simple probes on internal activations to detect specific linguistic, semantic, or structural features expected from the curated data.
-  **Representation Analysis:** Use CKA/RSA to compare representations between MLA/GQA models and track representational changes during training.
-  **MLA Latent Space (`c_kv`) Analysis:** Focus probes and analysis specifically on the compressed latent representation in the MLA model to understand its information content and abstraction level.
-  **Ablation Studies:** Systematically vary MLA parameters (e.g., `d_c`) or compare different RoPE handling methods.

**6.3. Goals**
-  Understand the functional differences between MLA and GQA when trained on dense data.
-  Assess whether the structured data fosters more human-understandable internal features.
-  Investigate the interaction between the MLA architecture (specifically the bottleneck) and the structured data.

**7. Conclusion**

This experimental design provides a framework for rigorously testing the hypothesis that high-density pre-training data can enhance LLM capabilities, using a ~1.3B parameter model with Multi-head Latent Attention (MLA) as the primary architecture and Grouped-Query Attention (GQA) as a baseline. The plan emphasizes a token-centric training methodology, iterative evaluation at defined milestones, and careful management of computational resources. By focusing on data quality and architectural efficiency, the experiment aims to contribute valuable insights into alternative LLM scaling paradigms within practical constraints.