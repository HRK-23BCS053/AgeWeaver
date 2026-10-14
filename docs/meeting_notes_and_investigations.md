# AgeWeaver Research Investigations & Meeting Notes Analysis

This document formally analyzes the meeting notes, unresolved theoretical questions, and strategic research directions for AgeWeaver in relation to FaceTT, contemporary vision-language models, and publication targets.

---

## 1. Analysis of Specific Meeting Notes

### 1.1 Qualitative Case Study: Rajesh Khanna
- **Discussion Context**: Actor Rajesh Khanna (1942–2012) was proposed as a longitudinal case study for facial aging due to his iconic 40-year cinematic career spanning distinct age phases:
  - Youthful breakthrough: 1969–1972 (*Aradhana*, *Kati Patang*, *Anand*), ages 27–30.
  - Mature character phase: 1980s (*Avtaar*, *Amrit*), ages 40–46.
  - Senior public phase: 2000s–2010s, ages 60–70.
- **Data Governance & Legal Compliance**:
  - Unlicensed film stills or arbitrary web photographs cannot be distributed in Git.
  - The dataset manifest incorporates entry `ID001_RAJESH_KHANNA` referencing an official public domain archival portrait from the Films Division / Government of India archives (1970).
  - This provides a legally compliant, historically verifiable anchor for evaluating qualitative age progression against real-life longitudinal photographs.

### 1.2 OpenCLIP vs. Model-Native Text Encoder Investigation
- **Meeting Point**: Exploration of OpenCLIP as an alternative or supplementary text encoder.
- **Architectural Analysis**:
  - The local base model (Stable Diffusion v1-5) uses OpenAI's CLIP ViT-L/14 with a 768-dimensional latent projection:
    $$\tau_\theta: \mathcal{V}^L \to \mathbb{R}^{77 \times 768}$$
  - OpenCLIP checkpoints (e.g. `ViT-H/14`, `ViT-bigG-14`, or `ConvNeXt-XXL`) produce different embedding dimensions (e.g. 1024 or 1280) and live in a distinct vector manifold.
  - **Critical Technical Reality**: Replacing the text encoder in an existing diffusion model cannot be done naively at inference time. The cross-attention projection matrices in the UNet:
    $$\mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{d_{\text{unet}} \times d_{\text{text}}}$$
    are rigidly parameter-matched to $d_{\text{text}} = 768$.
  - **Finding**: OpenCLIP can serve as an external scoring or alignment metric (e.g. measuring prompt-image cosine alignment), but cannot directly replace $\tau_\theta$ in SD 1.5 without retraining all UNet cross-attention blocks or inserting a projection adapter.

### 1.3 Text Encoder Selection
- For SD 1.5, preserving the native CLIP ViT-L/14 guarantees semantic stability.
- For modern diffusion architectures (SDXL, SD3, FLUX), dual text encoders (CLIP ViT-L + OpenCLIP ViT-G/14, or T5-XXL) are natively supported.
- On our CPU hardware constraint (16 GB RAM), loading T5-XXL (4.7 GB alone) would exceed memory headroom. Preserving CLIP ViT-L/14 is the optimal and stable path for SD 1.5.

### 1.4 The "Give-Back Prompt" (Unresolved Research Question)
- **Transcription**: The handwritten phrase *"give-back prompt"* appears in early supervisor notes.
- **Hypothesis A (Prompt Decomposition)**: A dual-branch prompting technique where the prompt first extracts/anchors identity features, and then "gives back" age-specific anatomical attributes (e.g. residual injection).
- **Hypothesis B (Reconstruction / Cyclic Inversion Prompt)**: A prompt designed for the backward phase of the cyclic consistency loop, where target age attributes are subtracted or inverted back to restore source age features.
- **Unresolved Status**: Preserved explicitly as an open research question for guide clarification rather than guessing.

### 1.5 Controller Network Exploration
- **Concept**: A learned policy network that predicts optimal generation parameters $(\sigma^*, s^*)$ given source face embeddings $\mathbf{e}_{\text{src}}$ and target age $y_{\text{tgt}}$.
- **Implemented Exploration**: A deterministic heuristic controller based on normalized age delta $\Delta_{\text{age}} = |y_{\text{tgt}} - y_{\text{src}}|$:
  $$\sigma^*(\Delta) = \sigma_{\min} + (\sigma_{\max} - \sigma_{\min}) \cdot \min\left(1.0, \frac{\Delta_{\text{age}}}{50}\right)$$
- **Distinction**: We explicitly label this as a *heuristic rule-based controller* rather than a *trained neural controller*, preserving scientific truthfulness.

---

## 2. Research Directions: R-1, R-2, R-3

### Direction R-1: Multi-Objective Loss Formulation
For future training or fine-tuning (e.g. LoRA or DreamBooth), the optimization objective is defined as:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{diff}} + \lambda_{\text{id}} \mathcal{L}_{\text{id}} + \lambda_{\text{age}} \mathcal{L}_{\text{age}} + \lambda_{\text{cyc}} \mathcal{L}_{\text{cyc}}$$
Where:
1. $\mathcal{L}_{\text{diff}} = \mathbb{E}_{t, \epsilon} [\|\epsilon - \epsilon_\theta(z_t, t, c)\|^2]$ (Standard latent diffusion loss).
2. $\mathcal{L}_{\text{id}} = 1 - \cos(\mathcal{F}_{\text{arc}}(I_{\text{gen}}), \mathcal{F}_{\text{arc}}(I_{\text{src}}))$ (ArcFace cosine loss).
3. $\mathcal{L}_{\text{age}} = \|\mathcal{G}_{\text{age}}(I_{\text{gen}}) - y_{\text{target}}\|_1$ (Age regression penalty).
4. $\mathcal{L}_{\text{cyc}} = 1 - \cos(\mathcal{F}_{\text{arc}}(I_{\text{recon}}), \mathcal{F}_{\text{arc}}(I_{\text{src}}))$ (Cyclic trajectory consistency).

*Honest Note*: In our current inference-only prototype, these terms are evaluated post-generation as quantitative metrics rather than backpropagated as gradients.

### Direction R-2: Prompt-Block Structural Modularization
- Decomposition of text conditioning into structured orthogonal blocks:
  $$\mathcal{P} = [\text{Subject Anchor}] \oplus [\text{Identity Invariant}] \oplus [\text{Age Bracket}] \oplus [\text{Skin Matrix}] \oplus [\text{Hair Matrix}] \oplus [\text{Style Prior}]$$
- Evaluated in Experiment B and shown to systematically outperform generic prompts.

### Direction R-3: Migration to Next-Generation Diffusion Models
- Baseline evaluation performed on RunwayML Stable Diffusion v1-5 due to local caching and CPU feasibility.
- Future evaluation planned for SDXL 1.0 (larger 1024x1024 native resolution), PixArt-$\Sigma$, and Flux.1-dev upon accessing GPU compute nodes.

---

## 3. Publication Target: WACV & WACV 2027

- **Target Venue**: IEEE/CVF Winter Conference on Applications of Computer Vision (WACV).
- **Target Year**: WACV 2027 (Paper submission expected summer/autumn 2026).
- **Track**: Applications in Biometrics, Face and Gesture Analysis, or Generative Modeling.
- **Status**: Future research aspiration. No claims of paper acceptance or review status are made.
- **Publication Requirements**:
  - Full evaluation on 100+ celebrity benchmark.
  - Ablation study across all three conditions (A, B, C).
  - Implementation of attention intervention (AAC).
  - Direct quantitative benchmark against FaceTT, SAM, and StyleCLIP baselines.

