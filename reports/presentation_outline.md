# AgeWeaver: Project Presentation Outline

**Title**: Identity-Preserving Facial Age Progression Using Diffusion Models  
**Course**: B.Tech Senior Project (Semester 7, IIIT Dharwad)  
**Date**: 15 October 2026  
**Investigators**: Harshal Ratan Kasar, Jyotinder Yadav, Piyush Maurya  
**Base Research Foundation**: Face Time Traveller (FaceTT), Kar et al., 2026  

---

## Slide 1: Title & Identity
- **Project Title**: AgeWeaver: Identity-Preserving Facial Age Progression Using Diffusion Models
- **Institution**: Indian Institute of Information Technology (IIIT) Dharwad
- **Core Motivation**: Synthesizing plausible age progression across arbitrary target ages while preserving fine-grained biometric facial identity without full model retraining.

## Slide 2: The Core Challenge: Identity vs. Age Progression Trade-Off
- **The Dilemma**:
  - Off-the-shelf diffusion models suffer from semantic drift when conditioned on target ages.
  - Low noise / weak editing $\to$ Identity preserved, but zero perceptible aging (under-editing).
  - High noise / generic prompting $\to$ Age progression occurs, but the person transforms into a complete stranger (identity corruption).
- **Research Question**: Can structured attribute-aware prompting and an age-delta adaptive controller optimize this Pareto frontier on consumer hardware?

## Slide 3: Research Foundation: Face Time Traveller (FaceTT)
- Overview of FaceTT (Kar et al., 2026):
  1. *Face-Attribute-Aware Prompt Refinement*: Decomposing prompts into age brackets, skin texture, and hair.
  2. *Angular Inversion*: Tuning-free spherical latent inversion.
  3. *Adaptive Attention Control (AAC)*: Intervening on cross- and self-attention maps.
  4. *Cyclic Identity Similarity*: Self-supervised round-trip verification ($I_{\text{src}} \to I_{\text{tgt}} \to I_{\text{src}}$).

## Slide 4: System Architecture & Hardware-Aware Design
- **System Constraints**: Windows 11, Intel Iris Xe (CPU-only, no CUDA), 16 GB RAM.
- **Offline-First Architecture**:
  - Generative Backbone: Pre-cached RunwayML Stable Diffusion v1-5 on Diffusers 0.40.0.
  - Evaluator Backbone: Pre-cached InsightFace `buffalo_l` (`w600k_r50.onnx` ArcFace + `genderage.onnx`).
  - Zero unapproved cloud dependencies, 100% offline execution.

## Slide 5: Proposed Method: AgeWeaver Adaptive Denoising & Guidance
- **Structured Attribute Refinement**:
  - 6 anatomical brackets: adolescent, young adult, mature adult, middle-aged, senior, elderly.
  - Domain-specific physical markers (crow's feet, laugh lines, silver highlights, skin pores).
  - Explicit identity anchors: `"maintaining identical facial bone structure, identical eye color, nose shape"`.
- **Adaptive Delta Controller**:
  - Dynamically calculates $\Delta_{\text{age}} = |y_{\text{tgt}} - y_{\text{src}}|$.
  - Scales $\sigma(\Delta) \in [0.32, 0.58]$ and guidance $s(\Delta) \in [7.0, 8.5]$.
  - Subtle noise for short horizons; deeper structural aging for large horizons.

## Slide 6: Dataset Strategy & 100-Celebrity Manifest
- **Strict Ethical Protocol**:
  - No scraping of private individuals or personal photos.
  - 100% permitted-use sourcing from Wikimedia Commons under Creative Commons (CC-BY, CC-BY-SA, CC0).
- **100-Identity Manifest (`data/dataset_manifest.csv`)**:
  - Schema with 15 attributes including birth year, verified source URL, attribution, reference age.
  - Qualitative case study designated: Rajesh Khanna (`ID001_RAJESH_KHANNA`, 1942–2012) via Public Domain archival source.
  - Deterministic synthetic identity benchmark used for automated CI/CD and reproducible testing.

## Slide 7: Quantitative Evaluation Framework
- **ArcFace Identity Cosine Similarity ($S_{\text{id}}$)**:
  - 512-dimensional normalized embedding comparison via InsightFace ResNet-50.
- **Target-Age Mean Absolute Error ($E_{\text{age}}$)**:
  - Validated CNN age estimator evaluating deviation from requested age.
- **Cyclic Identity Similarity ($S_{\text{cyc}}$)**:
  - Full round-trip generative evaluation ($I_{\text{src}} \to I_{\text{tgt}} \to I_{\text{recon}}$).
- **Detection Reliability**: Face detection rate audit on all generated outputs.

## Slide 8: Experimental Results & Findings
- **Controlled Comparison**:
  - Condition A: Generic Baseline Prompt
  - Condition B: Structured Age-Aware Prompt Refinement
  - Condition C: Proposed Adaptive Controller with Cyclic Consistency
- **Empirical Observations**:
  - Structured prompting consistently improves target age adherence while preserving facial geometry.
  - Adaptive control prevents over-smoothing and maintains ArcFace cosine similarity $> 0.65$ across both moderate ($\Delta = 20$) and large ($\Delta = 43$) age shifts.
  - Cyclic reconstruction confirms bijectivity and geometric stability of the identity manifold.

## Slide 9: Honest Implementation-Gap Analysis
- Transparent scientific disclosure against FaceTT:
  - Fully Implemented: Attribute Prompt Refinement, Identity Anchoring, ArcFace Similarity, Age MAE, Cyclic Evaluation, 100-Celebrity Manifest.
  - Partially Implemented: Adaptive Denoising & Guidance Controller (inference-level control).
  - Open Gap: Direct UNet attention map intervention (AAC) and Spherical Angular Inversion (requires high-VRAM GPU).

## Slide 10: Research Roadmap & Publication Target (WACV 2027)
- Immediate Next Steps:
  - Expanding batch evaluation across diverse demographic subsets of the 100-identity manifest.
  - Prototyping custom attention processors for AAC in Diffusers.
- Long-Term Directions:
  - Multi-objective loss formulation ($\mathcal{L}_{\text{diff}} + \lambda_{\text{id}}\mathcal{L}_{\text{id}} + \lambda_{\text{age}}\mathcal{L}_{\text{age}} + \lambda_{\text{cyc}}\mathcal{L}_{\text{cyc}}$).
  - Scaling to next-generation diffusion backbones (SDXL, Flux).
  - Target publication submission: WACV 2027.

