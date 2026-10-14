
# AgeWeaver — Research Blueprint

**Project:** Identity-Preserving Facial Age Progression Using Diffusion Models  
**Base paper:** Face Time Traveller (FaceTT), Kar et al., 2026  
**Status:** Research planning; implementation and results must be verified experimentally.

## 1. Research Problem

Facial age progression must introduce plausible age-related changes while preserving identity, facial structure, and relevant visual context.

## 2. Research Questions

- RQ1: How much does structured age-aware prompting improve target-age control compared with a generic prompt?
- RQ2: How does generation strength affect identity similarity and visible age progression?
- RQ3: Can a lightweight controller or prompt-refinement strategy improve the trade-off between age change and identity preservation?
- RQ4: Which evaluation metrics are feasible and reliable on a CPU-only development machine?

## 3. Base-Paper Components to Investigate

### A. Face-Attribute-Aware Prompt Refinement
Represent source age, target age, visible skin texture, hair characteristics, and other relevant attributes in a structured prompt. Investigate FastVLM and OpenCLIP as separate candidates; do not assume they are interchangeable.

### B. Angular Inversion
Study the paper's tuning-free latent inversion method and its angular scaling mechanism. Do not label a standard img2img noise-strength setting as Angular Inversion.

### C. Adaptive Attention Control (AAC)
Study the paper's cross-attention and self-attention control using attention statistics. Standard attention slicing alone is not AAC.

### D. Cyclic Identity Similarity
Evaluate a round trip: source-age image → older target → source-age reconstruction. Compare source and reconstructed face embeddings using a documented identity model.

## 4. Proposed Minimum Viable Experiment

- Use a small, documented set of appropriately licensed or permitted facial images.
- Include multiple source-to-target age requests.
- Compare a generic prompt against a structured age-aware prompt.
- Keep the generation model, input image, seed, and settings fixed wherever possible.
- Record failures as well as successful outputs.
- Report results only after executing the corresponding evaluation code.

## 5. Dataset Plan

Target: a documented manifest of at least 100 celebrity identities, subject to data availability and permitted use.

Record identity identifier, image source, source URL, estimated or verified age metadata, license/usage basis, and quality-control status.

A manifest of 100 identities is not equivalent to 100 downloaded, verified, usable images. Do not claim dataset completion until audited.

Use Rajesh Khanna as a potential qualitative case study only after verifying image provenance and permitted use. Do not upload unlicensed images to the public repository.

## 6. Evaluation Protocol

Primary:
- ArcFace cosine similarity for source-to-generated identity similarity.
- Cyclic identity similarity for round-trip consistency, where the full round trip is actually executed.
- Target-age error (MAE), only if a suitable age estimator is available and validated.

Secondary:
- Qualitative age plausibility and visible artifacts.
- Background consistency.
- Runtime and peak resource use.
- Face-detection success rate.

Compare methods under identical inputs and settings. Report sample count and variability when feasible.

## 7. Loss and Controller Investigation

First establish a reproducible baseline. Then investigate candidate objectives for age change, identity preservation, and image fidelity.

Do not claim a trained controller, optimized loss, or improvement unless the relevant implementation and controlled experiment exist. A prompt-selection heuristic is not a trained controller.

## 8. Compute and Scope

Development environment: Windows, Python 3.11, CPU-only.

Prioritize lightweight evaluation, reproducible baselines, prompt experiments, and clearly documented limitations. Full diffusion fine-tuning or faithful reproduction of every FaceTT component may not be feasible by the immediate deadline.

## 9. Integrity Rules

- No fabricated metrics, plots, dataset counts, or completed experiments.
- No claiming FaceTT reproduction when core components are absent.
- Preserve experiment configurations, random seeds, logs, and source metadata.
- Distinguish reproduced results, exploratory results, and planned work.
- Verify licenses and attribution before redistributing data or code.

## 10. Immediate Deliverable

Produce a working baseline, an auditable evaluation script, a small completed experiment, a concise report, and an explicit implementation-gap table. Expand only after these are reproducible.
