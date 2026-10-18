# Face Time Traveller (FaceTT): Implementation Fidelity Audit
**Document ID:** `research_artifacts/02_Configs_and_Protocols/FACETT_IMPLEMENTATION_FIDELITY_AUDIT.md`  
**Date of Audit:** 18 October 2026  
**Auditor:** Lead Research Engineer & Verification Framework  
**Base Paper:** *Face Time Traveller: Travel Through Ages Without Losing Identity*, Purbayan Kar et al., arXiv:2602.22819v1, 26 Feb 2026  
**Subject Codebase:** `src/ageweaver/` (AgeWeaver Mini Project-2)  

---

## 1. Auditing Standard & Integrity Mandate

In strict accordance with **Rule 6** (distinguishing verified code from reported claims) and **Rule 8** (never relabeling an approximation as an exact implementation of a published method), this audit conducts a rigorous line-by-line inspection of AgeWeaver's implementation against the mathematical algorithms published in the FaceTT base paper.

### Classification Taxonomy
- **`FAITHFUL`**: The codebase implements the published mathematical equations, algorithms, and tensor operations as specified in the paper.
- **`PARTIAL`**: Foundational logic or analogous components are implemented, but specific architectural steps differ or are evaluated on a subset.
- **`APPROXIMATED`**: A distinct engineering mechanism is implemented to achieve the same empirical objective, but using a completely different mathematical formulation.
- **`ABSENT`**: The paper mechanism is not present in the codebase.
- **`UNVERIFIED`**: Code exists but lacks execution verification or unit test coverage.

---

## 2. Component-by-Component Fidelity Breakdown

### 2.1 Component A: Face-Attribute-Aware Prompt Refinement (Paper Sec 3.1)
- **Paper Formulation:**
  - Extracts source facial attributes (gender, ethnic background, skin tone, facial hair, accessories) and temporal aging markers.
  - Generates conditioned text prompts $C_{\text{src}}$ and $C_{\text{tgt}}$ targeting specific biological ages while preserving non-aging attributes.
- **AgeWeaver Implementation:**
  - Implemented in [`src/ageweaver/prompts.py`](file:///d:/Academics/IIIT%20Dharwad%20Exams/SEMESTER-7/AgeWeaver/src/ageweaver/prompts.py#L45-L98): `build_structured_prompt()`.
  - Maps target age integers into 8 distinct age brackets (toddler to elderly) with physiological skin texture tokens (`delicate smooth skin`, `subtle laugh lines`, `pronounced wrinkles`, `weathered skin texture`) and hair pigmentation markers (`fine soft hair`, `salt and pepper`, `silver-white hair`).
  - Embeds explicit identity anchors: `"portrait of the exact same {subject}, identical facial bone structure, identical eyes and nose"`.
  - Integrates automated InsightFace gender hints (`man`, `woman`, `individual`).
- **Fidelity Assessment:** **`PARTIAL / HIGH FIDELITY`**
  - *Differences:* AgeWeaver utilizes deterministic anatomical token interpolation and InsightFace gender hints rather than an external vision-language model (e.g. BLIP/LLaVA) for dynamic accessory extraction.

---

### 2.2 Component B: Angular Inversion (Paper Sec 3.2 & Algorithm 1)
- **Paper Formulation (Algorithm 1):**
  - Decouples latent inversion into separate source ($z_{\text{src}}^t$) and target ($z_{\text{tgt}}^t$) branches starting from inverted latent $z_T^*$.
  - In the source branch, adds reconstruction offset $o_{\text{src}}^{t-1} = z_{t-1}^* - z_{\text{src}}^{t-1}$ directly to $z_{\text{src}}^{t-1}$.
  - Computes the spherical angle between inversion and generation trajectories:
    $$\theta_{\text{branch}}^{t-1} = \arccos\left(\frac{\langle z^*_{t-1} - z^*_T, \, z_{\text{branch}}^{t-1} - z^*_T \rangle}{\|z^*_{t-1} - z^*_T\| \cdot \|z_{\text{branch}}^{t-1} - z^*_T\|}\right)$$
  - Downweights trajectory divergence using exponential angular scaling:
    $$o_{\text{branch}}^{t-1} \leftarrow o_{\text{branch}}^{t-1} \cdot \exp(-\xi \cdot \theta_{\text{branch}}^{t-1})$$
  - Blends target and source offsets via cosine similarity $\beta_{t-1} = \cos(z^*_{t-1}, z_{\text{tgt}}^{t-1})$:
    $$z_{\text{tgt}}^{t-1} \leftarrow \text{DF}_{\text{tgt}}(z_{\text{tgt}}^t, t, C_{\text{tgt}}) + \beta_{t-1} \cdot o_{\text{tgt}}^{t-1} + (1 - \beta_{t-1}) \cdot o_{\text{src}}^{t-1}$$
- **AgeWeaver Implementation:**
  - Implemented in [`src/ageweaver/pipeline.py`](file:///d:/Academics/IIIT%20Dharwad%20Exams/SEMESTER-7/AgeWeaver/src/ageweaver/pipeline.py#L153-L179): `compute_adaptive_parameters()` and `progress_age()`.
  - Operates via standard HuggingFace Diffusers `StableDiffusionImg2ImgPipeline` with a 2nd-order `DPMSolverMultistepScheduler` (Karras noise formulation).
  - Approximates trajectory bounds by dynamically modulating the img2img denoising strength $\sigma$ and CFG guidance $s$:
    $$\sigma(\Delta y) = \text{clamp}\left(0.35 + 0.20 \times \left(\frac{|\Delta y|}{50}\right), \, 0.32, \, 0.58\right)$$
    $$s(\Delta y) = \text{clamp}\left(7.0 + 1.5 \times \left(\frac{|\Delta y|}{50}\right), \, 6.5, \, 8.5\right)$$
- **Fidelity Assessment:** **`APPROXIMATED / ARCHITECTURAL DIVERGENCE`**
  - > [!WARNING]
  - > In strict compliance with Rule 8, AgeWeaver's Adaptive Parameter Controller is **NOT** FaceTT Angular Inversion.
  - AgeWeaver achieves identity preservation by bounding the latent noise initialization ($\sigma \le 0.58$) rather than executing Algorithm 1's dual-branch angular trajectory scaling.
  - *Engineering Justification:* Enables deployment on standard HuggingFace Diffusers pipelines without requiring custom CUDA backward graph hooks.

---

### 2.3 Component C: Adaptive Attention Control (AAC, Paper Sec 3.3 & Algorithm 2)
- **Paper Formulation (Algorithm 2):**
  - Directly intercepts internal cross-attention ($M_{\text{cross}}$) and self-attention ($M_{\text{self}}$) tensors in U-Net decoder layers L4–L14.
  - Early phase ($t > \tau_1$, steps 36–50): Enforces strict cross-attention replacement ($M_{\text{cross}}^{\text{tgt}} \leftarrow M_{\text{cross}}^{\text{src}}$).
  - Intermediate phase ($\tau_2 \le t \le \tau_1$, steps 15–35):
    - Computes attention map KL-divergence: $\eta = D_{\text{KL}}(M_{\text{cross}}^{\text{src}} \,\|\, M_{\text{cross}}^{\text{tgt}})$.
    - If $\eta > \eta_{\text{th}} = 0.05$, blends cross-attention with entropy weight $w_{t-1} = 1 - H(M_{\text{cross}}^{\text{src}})$.
    - If $\eta \le \eta_{\text{th}}$, blends self-attention with entropy weight $w_{t-1} = 1 - H(M_{\text{self}}^{\text{src}})$.
  - Late phase ($t < \tau_2$, steps 1–14): Enforces self-attention replacement ($M_{\text{self}}^{\text{tgt}} \leftarrow M_{\text{self}}^{\text{src}}$) in layers 4–14 to preserve facial geometry and accessories.
- **AgeWeaver Implementation:**
  - AgeWeaver does not attach PyTorch forward hooks to U-Net cross-attention modules.
  - Instead, AgeWeaver relies on:
    1. Fixed identity tokens in the CLIP text conditioning vector.
    2. 15-step DPMSolver++ 2M ODE trajectory convergence.
- **Fidelity Assessment:** **`ABSENT`**
  - AgeWeaver does **NOT** implement internal U-Net attention tensor interception, KL-divergence thresholding ($\eta_{\text{th}}$), or entropy-based blending ($w_t = 1 - H(M)$).
  - *Audit Note:* Documented in `research_artifacts/08_Future_Work/FUTURE_WORK_QUEUE.md` as priority task **TSK-06** for Mini Project-3.

---

### 2.4 Component D: Cyclic Identity Similarity Protocol (Paper Sec 4.0 & 4.2)
- **Paper Formulation:**
  - Measures identity preservation without ground-truth target pairs via round-trip transformation:
    $$I_{\text{src}} \xrightarrow{y_{\text{tgt}}} I_{\text{tgt}} \xrightarrow{y_{\text{src}}} \hat{I}_{\text{src}}$$
  - Computes cyclic identity similarity $ID_{\text{sim}}^{\text{cyc}} = \cos(\mathcal{E}_{\text{ID}}(I_{\text{src}}), \mathcal{E}_{\text{ID}}(\hat{I}_{\text{src}}))$ using ArcFace normalized feature embeddings.
  - Separates reference-based identity similarity $ID_{\text{sim}}^{\text{ref}} = \cos(\mathcal{E}_{\text{ID}}(I_{\text{tgt\_real}}), \mathcal{E}_{\text{ID}}(I_{\text{tgt}}))$ which requires authentic same-person target photos.
- **AgeWeaver Implementation:**
  - Implemented in [`src/ageweaver/experiment.py`](file:///d:/Academics/IIIT%20Dharwad%20Exams/SEMESTER-7/AgeWeaver/src/ageweaver/experiment.py#L151-L177):
    - Generates $I_{\text{tgt}}$ via `progress_age(source_image, target_age)`.
    - Automatically feeds $I_{\text{tgt}}$ into second pass: `progress_age(cyclic_input, source_age)`.
    - Evaluates ArcFace cosine similarity via [`src/ageweaver/evaluation.py`](file:///d:/Academics/IIIT%20Dharwad%20Exams/SEMESTER-7/AgeWeaver/src/ageweaver/evaluation.py#L65-L80) using InsightFace `w600k_r50.onnx`.
  - Implements strict non-substitution: if no target reference photo exists, `reference_identity_similarity` is logged as `None` / `"N/A"`.
- **Fidelity Assessment:** **`FAITHFUL`**
  - Fully implements the exact cyclic round-trip protocol, ArcFace embedding extraction, and non-substitution data isolation policy specified in FaceTT.

---

## 3. Summary Fidelity Matrix

| FaceTT Method Component | Paper Section & Algorithm | AgeWeaver Equivalent | Fidelity Classification | Verification Evidence |
| :--- | :---: | :--- | :---: | :--- |
| **Face-Attribute-Aware Prompts** | Sec 3.1 | Anatomical 8-bracket prompt builder | **`PARTIAL`** | [`src/ageweaver/prompts.py`](file:///d:/Academics/IIIT%20Dharwad%20Exams/SEMESTER-7/AgeWeaver/src/ageweaver/prompts.py) |
| **Angular Inversion** | Sec 3.2, Alg 1 | Dynamic $\sigma(\Delta y)$ bounded img2img | **`APPROXIMATED`** | [`src/ageweaver/pipeline.py`](file:///d:/Academics/IIIT%20Dharwad%20Exams/SEMESTER-7/AgeWeaver/src/ageweaver/pipeline.py#L153) |
| **Adaptive Attention Control (AAC)** | Sec 3.3, Alg 2 | CFG scale $s(\Delta y)$ + CLIP tokens | **`ABSENT`** | Not present in codebase |
| **Cyclic Inversion Protocol** | Sec 4.0, 4.2 | Two-pass round-trip + ArcFace cosine | **`FAITHFUL`** | [`src/ageweaver/experiment.py`](file:///d:/Academics/IIIT%20Dharwad%20Exams/SEMESTER-7/AgeWeaver/src/ageweaver/experiment.py#L151) |
| **Reference Identity Non-Substitution** | Sec 4.2 | Strict `"N/A"` reporting for missing GT | **`FAITHFUL`** | [`src/ageweaver/evaluation.py`](file:///d:/Academics/IIIT%20Dharwad%20Exams/SEMESTER-7/AgeWeaver/src/ageweaver/evaluation.py) |
| **Diffusion Model Architecture** | Sec 4.0 | Stable Diffusion v1-5 (`runwayml/stable-diffusion-v1-5`) | **`FAITHFUL`** | Verified in config & pipeline |
| **Inference Hardware** | Sec 4.0 | Google Colab NVIDIA Tesla T4 (Cloud) / Intel CPU (Local) | **`DIVERGENCE`** | Paper used NVIDIA A100 |

---

## 4. Academic Audit Conclusion

AgeWeaver is an **adaptive latent diffusion framework** inspired by the core principles of Face Time Traveller (identity preservation, continuous aging, cyclic verification), but diverges architecturally:
1. It replaces the complex dual-branch U-Net attention hijacking (Algorithms 1 & 2) with a lightweight, model-agnostic **Adaptive Parameter Controller** ($\sigma(\Delta y)$, $s(\Delta y)$) and 2nd-order DPMSolver++ ODE sampling.
2. In all subsequent sections of this report, we maintain absolute transparency: experimental results are presented as **AgeWeaver's Adaptive Controller**, and are compared fairly against **FaceTT's published benchmark figures**.

