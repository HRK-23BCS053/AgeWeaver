# AgeWeaver: Core Configuration, Parameter Equations & Evaluation Protocols
**Document ID:** `research_artifacts/02_Configs_and_Protocols/PROTOCOL_SUMMARY.md`  
**Date of Audit:** 17 October 2026  
**Authoritative Reference:** Implemented in `src/ageweaver/pipeline.py`, `prompts.py`, `evaluation.py`, and `configs/default.yaml`

---

## 1. System Architecture Overview

AgeWeaver is designed for controllable bidirectional facial age progression and regression while strictly preserving identity representations. It interfaces latent diffusion generation with real-time biometric validation:

```
[Input Portrait I_src]
       │
       ▼
[InsightFace SCRFD + ArcFace] ──► Baseline Biometrics: Age y_src, Embedding e_src
       │
       ▼
[Adaptive Parameter Controller] ──► Prompt P(y_tgt), Denoising σ(|Δy|), Guidance s(|Δy|)
       │
       ▼
[Stable Diffusion v1-5 (DPM++ 2M Karras)] ──► Latent Diffusion img2img (15 Steps)
       │
       ▼
[Age-Progressed Portrait I_tgt]
       │
       ▼
[Evaluator: ArcFace + ResNet-34] ──► ID_sim_src, Age MAE, Cyclic Consistency
```

---

## 2. Mathematical Formulations & Parameter Control

### 2.1 Age Discrepancy Formulation
Let $y_{\text{src}} \in [1, 100]$ be the verified or estimated baseline age of the source identity, and $y_{\text{tgt}} \in [1, 100]$ be the requested target age. The absolute age jump is defined as:
$$\Delta y = y_{\text{tgt}} - y_{\text{src}}, \quad |\Delta y| = |y_{\text{tgt}} - y_{\text{src}}|$$

### 2.2 Adaptive Denoising Strength ($\sigma$)
To prevent identity collapse during short jumps while enabling necessary morphological aging across long jumps, denoising strength $\sigma$ is dynamically modulated:
$$\sigma(\Delta y) = \text{clamp}\left(0.35 + 0.20 \times \left(\frac{|\Delta y|}{50}\right), \, 0.32, \, 0.58\right)$$

- **Short-range aging ($|\Delta y| \le 10$):** $\sigma \approx 0.35\text{--}0.39$. Preserves fine bone structure, pores, and facial geometry.
- **Medium-range aging ($10 < |\Delta y| \le 30$):** $\sigma \approx 0.40\text{--}0.47$. Introduces localized wrinkles, nasolabial folds, and skin texture evolution.
- **Long-range aging ($|\Delta y| > 40$):** $\sigma \approx 0.50\text{--}0.58$. Modulates jawline skin laxity, hair silvering/thinning, and sub-orbital changes without exceeding the identity-collapse threshold ($\sigma > 0.60$).

### 2.3 Adaptive Classifier-Free Guidance Scale ($s$)
The CFG scale $s$ controls prompt adherence versus natural photo fidelity:
$$s(\Delta y) = \text{clamp}\left(7.0 + 1.5 \times \left(\frac{|\Delta y|}{50}\right), \, 6.5, \, 8.5\right)$$

- Prevents over-saturation and high-frequency edge burning by capping maximum guidance at $8.5$.

---

## 3. Diffusion Scheduling & Artifact Eradication Protocol

### 3.1 The Diagnostic Finding
Early iterations utilizing the default Diffusers `PNDMScheduler` at low step counts ($10\text{--}20$ steps) on CPU runtimes exhibited severe high-frequency checkerboard artifacts, blotchy skin discoloration, and fragmented backgrounds.

### 3.2 The DPMSolver++ Resolution
Through rigorous scheduler ablation across PNDM, Euler Discrete, DDIM, and DPMSolverMultistep, the optimal configuration was established:
- **Scheduler:** `DPMSolverMultistepScheduler`
- **Algorithm Type:** `dpmsolver++`
- **Solver Order:** 2 (`solver_order=2`)
- **Noise Formulation:** Karras sigmas enabled (`use_karras_sigmas=True`)
- **Inference Steps:** 15 steps (providing convergence identical to 50-step PNDM at $3.3\times$ lower latency)

---

## 4. Prompt Conditioning Protocols (3-Arm Controlled Comparison)

To systematically evaluate the contribution of conditioning logic, three standardized experimental arms are maintained:

| Experimental Arm | Prompt Structure | Hyperparameter Regime | Target Metric Expected |
| :--- | :--- | :--- | :--- |
| **Arm 1: Generic Baseline** | `"a photo of a person aged {age}"` | Fixed $\sigma = 0.55$, fixed $s = 7.5$ | Poor age precision, high identity variance |
| **Arm 2: Structured Prompt** | Anatomically calibrated prompt with gender, demographic, and aging tokens (e.g. skin laxity, hair pigmentation) | Fixed $\sigma = 0.55$, fixed $s = 7.5$ | Better age precision, moderate identity preservation |
| **Arm 3: Adaptive Control** | Structured Prompt + age modifier tokens | Dynamic $\sigma(\Delta y)$, dynamic $s(\Delta y)$ | Optimal Pareto frontier between Age MAE and $ID_{\text{sim}}^{\text{src}}$ |

### 4.1 Anatomical Token Calibration
- **Infant / Child ($0\text{--}12$ yrs):** `"smooth baby skin, large round eyes, soft chubby cheeks, childlike facial proportions"`
- **Young Adult ($18\text{--}35$ yrs):** `"smooth firm skin, youthful jawline, natural complexion, sharp facial definition"`
- **Middle Age ($36\text{--}55$ yrs):** `"subtle laugh lines, faint crow's feet, mature skin texture, distinguished expression"`
- **Senior ($56\text{--}75$ yrs):** `"visible wrinkles, skin laxity, graying silver hair, crow's feet, mature seasoned features"`
- **Elderly ($76+$ yrs):** `"deep facial wrinkles, age spots, thin silver hair, aged delicate skin, elderly facial structure"`

---

## 5. Biometric Evaluation Protocol & Equations

Biometric evaluation is executed fully locally and offline:

### 5.1 Source Identity Similarity ($ID_{\text{sim}}^{\text{src}}$)
Extracted via InsightFace ArcFace (`w600k_r50.onnx`, 512-dimensional normalized feature embedding):
$$ID_{\text{sim}}^{\text{src}} = \frac{\mathbf{e}_{\text{src}} \cdot \mathbf{e}_{\text{tgt}}}{\|\mathbf{e}_{\text{src}}\| \|\mathbf{e}_{\text{tgt}}\|}$$
- Acceptance threshold: $ID_{\text{sim}}^{\text{src}} \ge 0.40$ (indicating verified same-identity preservation).

### 5.2 Target-Age Mean Absolute Error (Age MAE)
Evaluated via InsightFace ResNet-34 age estimator (`genderage.onnx`):
$$\text{MAE} = |y_{\text{tgt}} - \hat{y}|$$

### 5.3 Cyclic Inversion / Round-Trip Consistency ($S_{\text{cyc}}$)
Measures round-trip reconstruction fidelity:
$$I_{\text{src}} \xrightarrow{\Delta y = y_{\text{tgt}} - y_{\text{src}}} I_{\text{tgt}} \xrightarrow{\Delta y' = y_{\text{src}} - y_{\text{tgt}}} \hat{I}_{\text{src}}$$
$$S_{\text{cyc}} = \frac{\mathbf{e}_{\text{src}} \cdot \hat{\mathbf{e}}_{\text{src}}}{\|\mathbf{e}_{\text{src}}\| \|\hat{\mathbf{e}}_{\text{src}}\|}$$

### 5.4 Reference Identity Similarity ($ID_{\text{sim}}^{\text{ref}}$)
$$ID_{\text{sim}}^{\text{ref}} = \frac{\mathbf{e}_{\text{ref}} \cdot \mathbf{e}_{\text{tgt}}}{\|\mathbf{e}_{\text{ref}}\| \|\mathbf{e}_{\text{tgt}}\|}$$
- Strictly recorded as `"N/A"` if no authentic, ground-truth same-subject target-age photograph is present.

---

## 6. Seed & Environment Control

- **Default Seed:** `42` (Deterministic across all comparative benchmark runs).
- **Ablation Seeds:** `41`, `49`, `53` (Used to evaluate cross-seed variance and diversity).
- **Execution Hardware:**
  - Local workstation: Intel CPU with OpenVINO / PyTorch float32.
  - Colab GPU: NVIDIA Tesla T4 (CUDA 12.2, FP16 half-precision pipeline).

