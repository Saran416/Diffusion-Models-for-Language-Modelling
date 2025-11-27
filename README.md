# Research Mini-Project: Autoregressive vs. Diffusion Language Models

This document records the theoretical study, implementation, and evaluation of non-autoregressive text generation techniques, conducted under the supervision of **Prof. Saketha Nath**. The primary objective was to compare a standard Autoregressive (AR) model against Diffusion Language Models to understand the trade-offs in inference speed, generation quality, control and diversity.

## 1. Literature Survey and Theoretical Foundations

The study began with a review of foundational generative models, including DDPM, Score-Based Generative Modeling, and D3PM. The research progression moved through the following key papers:

### Diffusion-LM - Continuous Space

- **Concept:** This approach maps discrete text to continuous space to apply standard Gaussian diffusion.
- **Key Learnings:**
  - Fixed embeddings were found to be suboptimal compared to end-to-end training.
  - The model initially failed to generate tokens that "committed" to a single word embedding, necessitating a complex objective function involving a constraint term.
  - **Limitations:** Higher perplexity and substantially slower decoding compared to AR models were observed.

### DiffuSeq - Continuous Space

- **Concept:** An extension of diffusion models to Sequence-to-Sequence tasks.
- **Key Learnings:**
  - The model adds noise only to the target sections of the sequence.

### MDLM: Simple and Effective Masked Diffusion - Discrete Space

- **Concept:** An extension of D3PM that operates in discrete space using an "absorbing state" (masking) rather than Gaussian noise.
- **Key Learnings:**
  - **Forward Process:** Tokens are masked with a probability varying over time such that the sequence eventually becomes fully masked.
  - **Optimization:** The approach uses the Rao-Blackwellized trick to simplify the loss function by ignoring gradients for unmasked tokens, significantly stabilizing training.

### FlexMDLM: Any-Order Flexible Length Masked Diffusion - Discrete Space

- **Concept:** Explored as a future direction, this method introduces an insertion step prior to unmasking.
- **Key Learnings:**
  - The objective function includes a divergence metric involving the predicted token position to handle flexible lengths.

## 2. Implementation

### Model Configuration

Both models were configured with approximately 25 million parameters:

- **Layers:** 8
- **Embedding Dimension:** 512
- **Attention Heads:** 8
- **Backbone:** Transformer (Decoder-only for AR; Encoder-only for MDLM).

#### Autoregressive (AR) Model

- **Mechanism:** Standard causal masking was used to predict the next token based on the history of previous tokens.
- **Training:** Optimized using standard Cross-Entropy loss.

#### Masked Diffusion Language Model (MDLM)

- **Parameterization:** Implemented using SUBS (Subsample) parameterization.
- **Diffusion Schedule:** A continuous time formulation was used to manage noise levels.
- **Sampling:** Ancestral sampling was implemented to denoise the sequence iteratively from a fully masked state.

## 3. Experimental Evaluation

> Please note that the observations and metrics presented below are derived from models trained for a limited number of iterations due to resource constraints. Consequently, these results should be interpreted as preliminary indicators of relative behavior rather than absolute performance benchmarks of fully converged models.

The models were evaluated based on inference speed and generation diversity (Self-BLEU).
The models were evaluated based on inference speed and generation diversity (Self-BLEU).

### Inference Speed Analysis

Tests were conducted on NVIDIA RTX 4060 GPU. The MDLM utilized 100 sampling steps for a block size of 256, while the AR model required 256 forward passes (one per token)

- **AR Performance:**
  - Average Time (100 tokens): 0.1412 seconds.
  - Speed: 708.25 tokens/second.
- **MDLM Performance:**
  - Average Time (256 tokens): 0.3159 seconds.
  - Speed: 810.37 tokens/second.

**Observation:** The MDLM achieved higher token throughput. Even though diffusion is iterative, the number of required steps (100) was less than the sequence length (256), offering a speed advantage for longer sequences.

### Diversity Analysis (Self-BLEU)

Self-BLEU was calculated over 50 generated samples to measure diversity (lower scores indicate higher diversity).

- **AR Self-BLEU:** 0.0282.
- **MDLM Self-BLEU:** 0.0171.

**Observation:** The MDLM demonstrated superior diversity. The non-autoregressive nature allows the model to plan globally rather than relying solely on local, greedy next-token predictions.

## 4. Code Repository Structure

- **ar.py**: Implementation and training script for the Autoregressive model.
- **mdlm.py**: Implementation and training script for the Masked Diffusion model.
- **eval_ar.py**: Evaluation script for AR speed and Self-BLEU metrics.
- **eval_mdlm.py**: Evaluation script for MDLM speed and Self-BLEU metrics.
- **ar.txt / mdlm.txt**: Logs containing training progress and evaluation outputs.

## 5. References

1. **Diffusion-LM:** Li et al., "Diffusion-LM Improves Controllable Text Generation," arXiv:2205.14217, 2022.
2. **DiffuSeq:** Gong et al., "DiffuSeq: Sequence to Sequence Text Generation with Diffusion Models," arXiv:2210.08933, 2022.
3. **MDLM:** Sahoo et al., "Simple and Effective Masked Diffusion Language Models," arXiv:2406.07524, 2024.
4. **FlexMDLM:** Arefeen et al., "Any-Order Flexible Length Masked Diffusion," arXiv:2509.01025, 2024.
