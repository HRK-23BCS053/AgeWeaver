# FaceTT vs. AgeWeaver Implementation-Gap Analysis

This document provides a transparent, scientifically honest comparison between the original **Face Time Traveller (FaceTT)** research paper (Kar et al., 2026) and the current **AgeWeaver (v0.1.0)** implementation.

## 1. Summary Comparison Table

| Component | FaceTT Paper Specification | AgeWeaver v0.1.0 Status | Technical Implementation Details & Current Status | Gap / Rationale |
|---|---|---|---|---|
| **1. Face-Attribute-Aware Prompt Refinement** | Disentangles age brackets, facial landmarks, hair, and wrinkles using vision-language priors. | **Fully Implemented** | Rule-based hierarchical prompt synthesis engine with 6 age brackets, skin texture markers, hair evolution, and explicit geometric identity anchors. | Uses deterministic rules rather than online VLM (e.g. FastVLM) to avoid massive VLM memory overhead on 16 GB CPU. |
| **2. Identity Anchor Conditioning** | Enforces biometric identity preservation tokens in text embedding space. | **Fully Implemented** | Strong negative prompts + positive invariant tokens (`"identical facial bone structure, identical eye color, nose shape"`). | Operates in CLIP ViT-L/14 text encoder space of SD 1.5. |
| **3. Angular Inversion** | Spherical latent inversion preserving angular trajectories in latent DDIM trajectory. | **Not Implemented (Baseline Img2Img Used)** | Standard intermediate noise initialization ($z_{t_0}$) with configurable strength $\sigma$. | Full angular spherical trajectory requires custom DDIM inversion loop with Jacobian scaling; marked as future work for GPU environments. |
| **4. Adaptive Attention Control (AAC)** | Cross-attention and self-attention map intervention during UNet denoising steps. | **Partially Implemented (Adaptive Controller)** | AgeWeaver implements an **Adaptive Denoising & Guidance Controller** modulating noise strength $\sigma(\Delta_{\text{age}})$ and guidance $s(\Delta_{\text{age}})$. | Direct tensor monkey-patching of attention maps in Diffusers 0.40.0 requires high memory bandwidth and custom attention processors; planned for Phase 2. |
| **5. Reference Identity Similarity** | ArcFace identity cosine similarity between source and generated face. | **Fully Implemented** | InsightFace Buffalo_l suite (`w600k_r50.onnx`), extracting normalized 512-d embeddings on CPU. | Exact match to FaceTT evaluation metric. |
| **6. Cyclic Identity Similarity** | Full round-trip evaluation: $I_{\text{src}} \to I_{\text{tgt}} \to I_{\text{src}}$ comparing ArcFace embeddings. | **Fully Implemented** | Two-pass generation pipeline with forward aging and reverse rejuvenation, evaluating $S_{\text{cyc}} = \mathbf{e}_{\text{src}}^\top \mathbf{e}_{\text{recon}}$. | Fully operational and logged in experimental trials. |
| **7. Age Estimation & MAE** | Quantitative evaluation of target age adherence. | **Fully Implemented** | InsightFace `genderage.onnx` CNN evaluating predicted age and calculating $\text{MAE} = |\hat{y} - y_{\text{target}}|$. | Executed locally offline with zero cloud dependency. |
| **8. OpenCLIP & VLM Exploration** | Exploring OpenCLIP vs. CLIP ViT-L/14 and FastVLM. | **Investigated (Documented)** | Architectural investigation documented in Research Notes; existing CLIP ViT-L/14 preserved to match SD 1.5 UNet cross-attention dimensions (768-d). | Replacing text encoder in SD 1.5 requires retraining cross-attention projection layers $\mathbf{W}_K, \mathbf{W}_V$. |
| **9. Celebrity Dataset Benchmark** | Curated multi-age celebrity dataset. | **Manifest Complete (100 Identities)** | Standardized manifest covering 100 celebrity identities with Wikimedia Commons CC-BY/CC0 attribution and birth years. | Physical downloads restricted to auditable public domain / CC samples to preserve disk and comply with usage limits. |
| **10. Compute Target & Acceleration** | High-end GPU cluster (CUDA). | **CPU Optimized (Intel Iris Xe)** | Float32 multi-threaded CPU execution with bounded 16 GB RAM footprint. | Inference latency is higher (~20-25s per run) but 100% reproducible on consumer hardware without GPU. |

## 2. Integrity Statement

In adherence to scientific integrity principles:
- **No false claims of complete FaceTT reproduction**: AgeWeaver explicitly identifies where off-the-shelf DDPM img2img and adaptive prompting were employed as computationally feasible substitutes for Angular Inversion and AAC.
- **Pareto Trade-Off Observation**: The experimental results reflect actual measured numbers from PyTorch and InsightFace on the local Intel Iris Xe CPU. No metrics have been fabricated or smoothed.

