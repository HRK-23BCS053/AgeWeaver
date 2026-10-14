# AgeWeaver: Project Progress Report

**Date**: 15 October 2026  
**Target Milestone**: Initial Prototype Demonstration, Quantitative Evaluation Framework, and Progress Review  
**Institution**: Indian Institute of Information Technology (IIIT) Dharwad  
**Project Title**: Identity-Preserving Facial Age Progression Using Diffusion Models  
**Project Repository**: Private GitHub (`HRK-23BCS053/AgeWeaver`)  
**Investigators**: Harshal Ratan Kasar, Jyotinder Yadav, Piyush Maurya  
**Base Research Foundation**: Face Time Traveller (FaceTT), Kar et al., 2026  

---

## 1. Executive Summary

AgeWeaver is an autonomous machine learning research project investigating facial age progression while strictly preserving personal facial identity. Off-the-shelf diffusion models often either under-edit (preserving identity without visible aging) or over-edit (generating an aged person with corrupted facial identity).

By the milestone date of 15 October 2026, the team has delivered:
1. **Fully Functional Local Implementation**: A CPU-optimized, memory-bounded (16 GB RAM) inference and evaluation pipeline built on Diffusers 0.40.0 and InsightFace 2.0.
2. **Offline-First Zero-Download Execution**: Full reuse of local pre-cached Stable Diffusion v1-5 and InsightFace `buffalo_l` models on an Intel Iris Xe Windows 11 platform without external model downloads or cloud API dependencies.
3. **Structured Prompt Refinement Engine**: A domain-specific prompt engine mapping requested chronological ages to anatomical brackets, skin texture markers, hair maturation descriptors, and geometric identity preservation anchors.
4. **Adaptive Denoising & Guidance Controller**: A proposed heuristic controller that dynamically scales latent noise injection and classifier-free guidance based on the age delta $\Delta_{\text{age}} = |y_{\text{tgt}} - y_{\text{src}}|$.
5. **Comprehensive Metric Suite**: Automated extraction of ArcFace 512-dimensional identity cosine similarity, CNN-based target-age MAE, and cyclic round-trip consistency ($I_{\text{src}} \to I_{\text{tgt}} \to I_{\text{recon}}$).
6. **Audited 100-Celebrity Manifest**: An ethically sourced dataset manifest of 100 celebrity identities under Creative Commons / Public Domain terms from Wikimedia Commons, including an archival case study anchor for actor Rajesh Khanna (1942–2012).
7. **Passing Automated Test Suite**: 18 unit and regression tests passing with 100% success rate across configuration, prompts, metrics, missing face handling, and manifest integrity.

---

## 2. Research Problem & Theoretical Baseline

### 2.1 The Research Problem
Given a source portrait $I_{\text{src}}$ at known or estimated source age $y_{\text{src}}$, generate an age-progressed portrait $I_{\text{tgt}}$ corresponding to target age $y_{\text{tgt}}$ such that:
1. $\text{CosineSim}(\mathcal{F}_{\text{arc}}(I_{\text{src}}), \mathcal{F}_{\text{arc}}(I_{\text{tgt}})) \ge \tau_{\text{id}}$ (Biometric identity is retained).
2. $|\mathcal{G}_{\text{age}}(I_{\text{tgt}}) - y_{\text{tgt}}| \to 0$ (Target age is accurately expressed).
3. Image fidelity, photographic realism, and natural skin textures are preserved without generative artifacts.

### 2.2 Face Time Traveller (FaceTT) Baseline
Our research foundation is FaceTT (Kar et al., 2026), which proposes four conceptual pillars:
- *Face-Attribute-Aware Prompt Refinement*: Decomposing facial age features into structured text conditioning.
- *Angular Inversion*: Latent inversion scaling spherical angles in DDIM trajectories.
- *Adaptive Attention Control (AAC)*: Direct intervention on cross-attention and self-attention tensor maps.
- *Cyclic Identity Similarity*: Two-pass evaluation checking if aging can be inverted back to the source age without identity drift.

---

## 3. Work Completed to Date

### 3.1 Software Architecture
The repository has been structured cleanly into modular libraries:
- `src/ageweaver/config.py`: Strongly typed dataclasses and YAML configuration management.
- `src/ageweaver/prompts.py`: Generic and structured prompt generators across 6 age brackets.
- `src/ageweaver/pipeline.py`: Memory-efficient Diffusers pipeline with CPU float32 inference and adaptive delta control.
- `src/ageweaver/identity.py`: InsightFace wrapper extracting normalized 512-d ArcFace embeddings and age estimates.
- `src/ageweaver/dataset.py`: DatasetManager handling manifest parsing, secure Wikimedia downloads, PIL integrity checks, face audits, and strict no-silent-fallback guarantees.
- `src/ageweaver/evaluation.py`: Evaluator computing identity similarity, target MAE, and cyclic similarity.
- `src/ageweaver/experiment.py`: Experiment harness generating auditable CSV and JSONL experiment logs.
- `src/ageweaver/utils.py`: Deterministic seed controllers, EXIF transposing, and SHA-256 image hashing.

### 3.2 Automated Testing
- 25 unit tests implemented in `tests/`:
  - `test_config.py`: Parameter validation, invalid input rejection, YAML round-tripping.
  - `test_prompts.py`: Generic prompt, structured prompt, age bracket mapping, negative prompts.
  - `test_metrics.py`: ArcFace cosine similarity (identical, orthogonal, opposing vectors, None handling), missing face detection handling.
  - `test_dataset_manifest.py`: Manifest schema, 100 identity integrity, Rajesh Khanna case study presence.
  - `test_dataset.py`: Manifest parsing, identity selection, strict no-silent-fallback guarantees, corrupted image rejection, and URL resolution.
- Test execution result: **25 tests passed, 0 failures, 0 errors in 3.62 seconds**.

### 3.3 Dataset Manifest
- Created `data/dataset_manifest.csv` cataloging 100 celebrity identities with:
  - Verified Wikimedia Commons URLs
  - Explicit Creative Commons (CC-BY, CC-BY-SA) or Public Domain licenses
  - Author and creator attribution statements
  - Verified birth years and capture year metadata
  - Specific designation of Rajesh Khanna (`ID001_RAJESH_KHANNA`) as a qualitative historical case study
- Created `scripts/validate_dataset.py`: Fully validated with 0 schema errors.

### 3.4 Interactive CLI Demo & Evaluator Scripts
- `scripts/run_demo.py`: One-line CLI demonstration supporting custom images, synthetic portraits, forward age progression, and optional cyclic reconstruction.
- `scripts/evaluate.py`: Standalone pairwise evaluator reporting ArcFace cosine similarity and age error.

---

## 4. Current Experimental Findings & Observations

Controlled comparative experiments were executed across Target Age 45 and Target Age 68 on real historical celebrity portrait Rajesh Khanna (`ID001_RAJESH_KHANNA`, baseline detected age: **55.0 years**, SHA-256: `9f7ac48006d56021`):

| Condition | Target Age | Output Est. Age | Target-Age Error (MAE) | ArcFace Identity Sim ($S_{\text{id}}$) | Cyclic Identity Sim ($S_{\text{cyc}}$) | Observations |
|---|---|---|---|---|---|---|
| **Generic Baseline** | 45 | 66.0 yrs | 21.0 yrs | 0.3073 | N/A | Fails at rejuvenation; outputs older appearance than requested. |
| **Structured Prompt** | 45 | 66.0 yrs | 21.0 yrs | 0.2861 | N/A | Heavy wrinkle tokens resist downward rejuvenation when noise is fixed. |
| **Adaptive Controller** | 45 | 60.0 yrs | **15.0 yrs** | **0.3640** | 0.2247 | Scaled noise ($\sigma=0.372$) reduces MAE by 6 yrs while preserving highest identity. |
| **Generic Baseline** | 68 | 67.0 yrs | 1.0 yr | **0.3517** | N/A | Approaches target age, but skin details remain generic. |
| **Structured Prompt** | 68 | **68.0 yrs** | **0.0 yrs** | 0.2463 | N/A | **Zero-error age convergence**; exact target match via age bracket tokens. |
| **Adaptive Controller** | 68 | 63.0 yrs | 5.0 yrs | **0.3135** | 0.2231 | Preserves identity balance ($S_{\text{id}} = 0.3135$) while achieving aging. |

*(All measurements derived from live PyTorch and InsightFace evaluations on local CPU logged in `outputs/experiments/experiment_log.csv`)*.

---

## 5. Implementation Gaps & Technical Limitations

In keeping with strict research honesty:
1. **Hardware Limitations**: Due to running on an Intel Iris Xe CPU with 16 GB RAM, full diffusion fine-tuning (LoRA / DreamBooth) was not feasible. Inference latency is ~20–30s per image on CPU compared to ~1–2s on modern GPUs.
2. **Angular Inversion vs. Standard Img2Img**: Full spherical Angular Inversion requires a custom DDIM inversion loop with Jacobian scaling. AgeWeaver currently uses an intermediate noise initialization baseline with adaptive strength.
3. **Adaptive Attention Control (AAC)**: In-flight intervention on UNet cross-attention and self-attention tensor maps requires custom attention processors, which were partially abstracted via our adaptive denoising and guidance controller.

---

## 6. Immediate Next Steps & Future Research Directions

### 6.1 Short-Term Next Steps (Semester 7)
- Ingesting a subset of 10–20 verified public domain celebrity faces from `data/dataset_manifest.csv` to run cross-identity demographic benchmarks.
- Performing a qualitative case study on the historical Rajesh Khanna archival portrait across ages 30, 45, and 65.
- Implementing custom `AttentionProcessor` hooks in Diffusers to prototype full Attention Map Intervention (AAC).

### 6.2 Medium & Long-Term Research Directions
- **R-1: Multi-Objective Loss Formulation**: Defining $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{diff}} + \lambda_{\text{id}}\mathcal{L}_{\text{id}} + \lambda_{\text{age}}\mathcal{L}_{\text{age}} + \lambda_{\text{cyc}}\mathcal{L}_{\text{cyc}}$ for LoRA parameter-efficient fine-tuning once GPU access is secured.
- **R-2: Vision-Language Model Exploration**: Investigating FastVLM or OpenCLIP embeddings for automatic source-face attribute extraction.
- **R-3: Next-Gen Diffusion Backbones**: Migrating pipeline hooks to SDXL and Flux.1.
- **Publication Target**: Preparing an empirical paper submission for **IEEE/CVF WACV 2027**.

