# AgeWeaver: Limitations, Constraints & Architectural Scope
**Document ID:** `research_artifacts/06_Paper_and_Supplementary/LIMITATIONS.md`  
**Date:** 18 October 2026  
**Auditor:** Technical Documentation Lead & Computer Vision Researcher  
**Integrity Rule:** Complete transparency regarding academic constraints and unrun items.  

---

## 1. Hardware & Compute Constraints

| Dimension | FaceTT Published Specification | AgeWeaver Execution Environment | Technical Discrepancy & Impact |
| :--- | :--- | :--- | :--- |
| **GPU Hardware** | NVIDIA A100 (80GB VRAM, PCIe Gen4) | Google Colab NVIDIA Tesla T4 (16GB VRAM) / Intel CPU | Precludes training large batch sizes on full 1024x1024 datasets; inference adapted to 512x512 resolution. |
| **Compute Budget** | Multi-day dedicated GPU cluster | Free tier Colab sessions (~3-5 hours session limit) | Batch sweeps partitioned into discrete 30-trial modules to avoid mid-run preemptions. |
| **Precision** | PyTorch FP32 / TF32 | Colab FP16 / CPU FP32 | FP16 on T4 yields $80\times$ speedup ($1.25$s latency) with negligible numerical drift ($<0.001$ cosine similarity delta). |

---

## 2. Dataset Accessibility & Storage Limitations

1. **FFHQ-Aging 70,000-Image Fine-Tuning:**
   - The paper fine-tunes Stable Diffusion for 150 steps across 70,000 images at $1024 \times 1024$ resolution (~50GB archive).
   - *Limitation:* Downloading, parsing, and fine-tuning on 50GB of image data was impossible within the 20-hour submission deadline. AgeWeaver consequently operated via zero-shot adaptive prompt conditioning on the pre-trained SD 1.5 weights.
2. **CelebA-HQ 1,000 Young Image Cohort:**
   - The paper's young-to-60 evaluation requires 1,000 high-resolution young faces.
   - *Limitation:* Local execution evaluated verified young celebrity references (`ID001_RAJESH_KHANNA` and `ID009_DEEPIKA_PADUKONE`). Paper-reported aggregate numbers for the full 1,000-image cohort are documented transparently from literature.
3. **Longitudinal Ground-Truth Reference Sourcing:**
   - Sourcing verified, high-resolution photographs of identical celebrity subjects at exact target ages (e.g. Rajesh Khanna at age 50 and 70) with permissive open licenses remains an open archival task.
   - In accordance with our **Strict Non-Substitution Policy**, reference identity similarity ($ID_{\text{sim}}^{\text{ref}}$) is recorded as `"N/A"` rather than synthesized or approximated.

---

## 3. Biometric & Evaluation Metric Boundaries

1. **Age Estimator Reliability:**
   - Target-Age MAE is evaluated using InsightFace's ResNet-34 `genderage.onnx` model.
   - *Boundary:* The estimator's output is an algorithmic prediction, not absolute ground truth. Age estimators exhibit demographic variance, particularly on historical sepia portraits, postage stamp textures, and extreme childhood ages ($0\text{--}4$ years).
2. **Biometric FNMR@FMR Calibration:**
   - Calculating False Non-Match Rate (FNMR) at fixed False Match Rate (FMR = 0.01%) requires calibrating decision thresholds over hundreds of thousands of non-matching impostor face pairs.
   - *Boundary:* Without a massive biometric cohort, calculating true FNMR locally would be mathematically invalid. AgeWeaver reports direct ArcFace cosine similarity and cites the paper's calibrated FNMR table.
3. **Kernel Inception Distance (KID):**
   - KID requires polynomial kernel evaluation against thousands of real reference images per age group. It cannot be reliably calculated on small sample sizes.

---

## 4. Architectural Gaps vs. Base Paper

1. **Absence of Algorithm 1 (Angular Inversion):**
   - AgeWeaver uses bounded latent noise initialization ($\sigma \le 0.58$) rather than dual-branch spherical angular trajectory downweighting ($\exp(-\xi \cdot \theta)$).
2. **Absence of Algorithm 2 (Adaptive Attention Control):**
   - AgeWeaver modulates global prompt tokens and CFG scale ($s \in [6.5, 8.5]$) rather than injecting PyTorch forward hooks into U-Net decoder cross-attention layers L4–L14.
3. **2D Latent Craniofacial Warping:**
   - Latent diffusion operates in image feature space. For extreme childhood regression (Age 2), adult craniofacial proportions (e.g. chin length, skull-to-face ratio) are difficult to reshape purely through 2D img2img without 3D morphable mesh priors.

