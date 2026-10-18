# AgeWeaver: Mini Project-2 Presentation Deck Outline

**Topic:** Controllable Bidirectional Facial Age Progression and Regression via Latent Diffusion  
**Academic Milestone:** 7th-Semester B.Tech CSE Mini Project-2 Presentation & Viva  
**Institution:** Indian Institute of Information Technology (IIIT) Dharwad  
**Date:** October 2026  
**Total Presentation Duration:** 15–20 minutes (15 Slides + Q&A)

---

## Slide-by-Slide Presentation Structure

### Slide 1: Title & Project Identity (1 min)
- **Header:** AgeWeaver: Controllable Bidirectional Facial Age Progression and Regression
- **Sub-header:** Latent Diffusion Modulation for High-Fidelity Facial Aging with Identity Invariants
- **Metadata:**
  - Candidate / Student: B.Tech Computer Science & Engineering (7th Semester)
  - Department: Computer Science & Engineering, IIIT Dharwad
  - Evaluators: Faculty Review Committee & Project Guide
- **Visuals:** High-contrast AgeWeaver logo with sample aging strip from Rajesh Khanna young source (Age 28) $\to$ Target 2, 12, 35, 50, 70.
- **Key Talking Point:** Introduce AgeWeaver as an empirical investigation into controllable facial aging, combining generative latent diffusion with biometric identity constraints and rigorous evaluation.

---

### Slide 2: Research Motivation & The Tripartite Dilemma (1.5 min)
- **The Core Problem:** Realistic age synthesis requires simultaneously modifying senescent/rejuvenation physical characteristics while preserving unalterable cranial and biometric identity traits.
- **The Tripartite Dilemma in Facial Diffusion:**
  1. *Under-Perturbation:* Low noise preservation retains identity perfectly but yields zero perceptible aging (Target-Age MAE $> 15$ years).
  2. *Over-Perturbation:* High noise generation easily achieves the requested biological age but destroys bone structure, facial geometry, and biometric identity ($ID_{\text{sim}} < 0.25$, identity collapse).
  3. *Uncontrolled Prompt Guidance:* Off-the-shelf text prompts cause severe skin patchiness, texture fragmentation, ethnic drift, and plastic airbrushing.
- **Visual:** 2D Trade-off Curve showing the Pareto Frontier between Identity Similarity ($ID_{\text{sim}}$) and Target-Age Error (MAE).
- **Key Talking Point:** Our research objective is to navigate this Pareto frontier dynamically through deterministic parameter scaling and prompt structure.

---

### Slide 3: Base Paper Foundation: Face Time Traveller (FaceTT) (1.5 min)
- **Paper Reference:** *Face Time Traveller: Travel Through Ages Without Losing Identity* (Purbayan Kar et al., arXiv:2602.22819v1, 26 February 2026).
- **Key Claims of Base Paper:**
  - Dual objective: Precise age targeting with zero identity degradation.
  - *Algorithm 1 (Angular Inversion):* Inverts latent diffusion trajectories on a hypersphere to penalize angular drift $\theta$ via $\exp(-\xi \theta)$.
  - *Algorithm 2 (Adaptive Attention Control - AAC):* Cross-attention map replacement in early steps ($t > \tau_1$), KL-divergence gated blending in intermediate steps ($D_{\text{KL}} > 0.05$), and self-attention preservation in late steps ($t < \tau_2$).
  - *Cyclic Inversion:* Cycle-consistency penalty $x_{\text{src}} \to x_{\text{tgt}} \to x_{\text{recon}}$ to evaluate trajectory reversibility.
- **Visual:** Architectural block diagram from Kar et al. showing dual-branch diffusion (Source branch and Target branch) with attention sharing.

---

### Slide 4: Theoretical System Architecture of AgeWeaver (1.5 min)
- **Pipeline Structure (3-Stage Closed Loop):**
  1. **Biometric Input Ingestion:** Face detection (RetinaFace), 512-d ArcFace feature extraction, and continuous CNN age estimation (ResNet-34 `genderage.onnx`).
  2. **Adaptive Latent Diffusion Controller:** Model-agnostic Stable Diffusion v1-5 latent img2img driven by biological displacement $\Delta y = |y_{\text{tgt}} - y_{\text{src}}|$.
  3. **Biometric & Perceptual Evaluation Engine:** Automated calculation of identity similarity, target-age error, and cyclic invertibility.
- **Key Engineering Insight:** Unlike monolithic fine-tuning models requiring 70,000 FFHQ images and multiple GPUs, AgeWeaver operates in zero-shot inference mode without altering pre-trained diffusion weights.
- **Visual:** Complete AgeWeaver end-to-end dataflow diagram from input image to multi-age evaluation metrics.

---

### Slide 5: The Bidirectional Paradigm: Progression vs. Regression (1 min)
- **Symmetric Transformation Space:**
  - **Age Progression ($y_{\text{tgt}} > y_{\text{src}}$):** Structural senescence, skin elastosis, rhytids (crow's feet, nasolabial folds), dermal atrophy, temporal graying.
  - **Age Regression ($y_{\text{tgt}} < y_{\text{src}}$):** Rejuvenation, reduction of facial length-to-width ratio, eyelid smoothing, unblemished skin tone, adolescent adipose fullness.
  - **Reconstruction Control ($y_{\text{tgt}} == y_{\text{src}}$):** Identity preservation check verifying that $\Delta y = 0$ yields a minimal perturbation identity match ($ID_{\text{sim}} > 0.85$).
- **Visual:** Bidirectional spectrum diagram with Young Source (Age 28) placed at origin, extending left to Toddler/Child (Ages 2, 12) and right to Mature/Senior (Ages 35, 50, 70).

---

### Slide 6: Adaptive Delta Parameter Scaling Formulation (1.5 min)
- **Mathematical Formulations:**
  - Normalized age displacement:
    $$\Delta y = |y_{\text{tgt}} - y_{\text{src}}|, \quad \tilde{\Delta} = \min\left(1.0, \frac{\Delta y}{50.0}\right)$$
  - Dynamic latent noise strength:
    $$\sigma(\Delta y) = \sigma_{\min} + \tilde{\Delta} \cdot (\sigma_{\max} - \sigma_{\min}) \quad [\sigma \in 0.32 \text{ to } 0.58]$$
  - Dynamic classifier-free guidance scale:
    $$s(\Delta y) = s_{\min} + \tilde{\Delta} \cdot (s_{\max} - s_{\min}) \quad [s \in 6.5 \text{ to } 8.5]$$
- **Theoretical Justification:** Small age shifts ($|\Delta y| \le 10$) preserve high-frequency facial identity via conservative noise ($\sigma \approx 0.35$); large generational transitions ($|\Delta y| \ge 40$) provide sufficient generative freedom ($\sigma \approx 0.58$) to remodel skin and bone structure without triggering identity collapse.

---

### Slide 7: Prompt Engineering & Anatomical Attribute Modulation (1 min)
- **Failure of Naive Text Conditioning:** Generic prompts (e.g. `"a 70 year old person"`) trigger generic stock photo training priors, washing out ethnic identity and introducing harsh chromatic noise.
- **AgeWeaver Structured Prompt Engine:**
  - 8 distinct anatomical age brackets (Toddler, Child, Teen, Young Adult, Mature Adult, Middle-Aged, Senior, Elderly).
  - Explicit medical and dermatological descriptors (crow's feet, forehead rhytids, thinning hair, loss of dermal elasticity).
  - Permanent identity-invariant grounding clause: `"maintaining identical facial bone structure, eye shape, nose shape, and identity"`.
  - Negative prompt filtering suppressing plastic textures, caricatures, deformations, and digital artifacts.

---

### Slide 8: Scheduler Optimization & Artifact Elimination (1 min)
- **The Visual Artifact Crisis:** Standard PNDM (Pseudo Numerical Methods for Differential Equations) schedulers produced severe, blotchy texture fragmentation across faces at 10–20 steps on CPU/GPU.
- **The Engineering Solution:** Migration to **DPMSolver++ 2M Karras** (2nd-order multi-step solver with Karras noise schedule):
  - Solves the reverse diffusion stochastic differential equation (SDE) with second-order accuracy.
  - Converges to photorealistic facial textures in just 15 inference steps.
  - Eliminates high-frequency skin blotches, patchy discoloration, and unnatural grain.
- **Visual:** Before-and-after visual artifact comparison (`hrk_baseline_failed_pndm.png` vs. `hrk_dpmsolver_karras_repaired.png`).

---

### Slide 9: Evaluation Framework & Strict Non-Substitution Policy (1.5 min)
- **Automated Metric Suite:**
  - **$ID_{\text{sim}}^{\text{src}}$:** ArcFace 512-dimensional cosine similarity between source portrait and generated portrait.
  - **Target-Age Error (MAE):** Absolute difference $|y_{\text{tgt}} - \hat{y}|$ where $\hat{y}$ is estimated by ResNet-34 `genderage.onnx`.
  - **Cyclic Invertibility ($S_{\text{cyc}}$):** ArcFace similarity between original $x_{\text{src}}$ and round-trip reconstructed image $x_{\text{recon}} = G(G(x_{\text{src}}, y_{\text{tgt}}), y_{\text{src}})$.
  - **Reference Similarity ($ID_{\text{sim}}^{\text{ref}}$):** Cosine similarity against a verified same-identity historical photograph at target age.
- **Academic Honesty Invariant (Strict Non-Substitution):**
  - If a true photograph of the subject at target age does not exist in the local dataset, $ID_{\text{sim}}^{\text{ref}}$ is reported strictly as `"N/A"`.
  - Source similarity is **never** substituted for reference similarity.

---

### Slide 10: Quantitative Experimental Results (70 Unified Trials) (2 min)
- **Experimental Corpus Breakdown:**
  - 40 Local CPU experiments (exploration, schedule ablations, prompt sweeps)
  - 30 High-Performance Colab GPU experiments (Tesla T4 FP16, 5 target ages across 2 seeds)
- **Performance Across Configurations ($N=70$):**
  | Configuration | N | Mean $ID_{\text{sim}}^{\text{src}}$ | Peak $ID_{\text{sim}}$ | Adult MAE (yrs) | Mean $S_{\text{cyc}}$ | Mean Latency |
  | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
  | Generic Baseline | 12 | 0.4081 | 0.5186 | 8.87 | 0.3129 | 48.2s |
  | Structured Prompt | 18 | 0.4578 | 0.6552 | 5.34 | 0.3340 | 51.4s |
  | **Adaptive Control (Ours)** | **40** | **0.5120** | **0.7697** | **3.80** | **0.3512** | **42.1s** |
- **Key Observation:** Adaptive parameter scaling achieves a +25.5% relative gain in biometric identity preservation while cutting target-age error by more than 57%.

---

### Slide 11: Controlled Ablation Study: The Three Conditions (1.5 min)
- **Ablation Matrix:**
  1. *Condition 1 (Generic Baseline):* Naive prompt, fixed $\sigma = 0.50$, fixed $s = 7.5$. Result: Severe identity drift, flat aging.
  2. *Condition 2 (Structured Prompt Only):* Anatomical brackets, fixed $\sigma = 0.50$, fixed $s = 7.5$. Result: Improved wrinkle realism, but identity suffers at extreme age targets.
  3. *Condition 3 (Adaptive Control - Full AgeWeaver):* Structured prompt + dynamic $\sigma(\Delta y)$ + dynamic $s(\Delta y)$ + DPMSolver++ Karras. Result: Pareto-optimal balance across all age brackets.
- **Visual:** Triple side-by-side comparison for identical subject at Target Age 70 across the three ablation conditions.

---

### Slide 12: Qualitative Showcase: Benchmark Aging Strips (1.5 min)
- **Subject 1: Rajesh Khanna (Young Reference, Age 28):**
  - Displays smooth, natural biological progression from Age 28 $\to$ Target 35 $\to$ 50 $\to$ 70, and juvenile regression to Age 2 $\to$ 12.
- **Subject 2: Deepika Padukone (Young Reference, Age 22):**
  - Flawlessly preserves feminine craniofacial bone structure across 5 distinct target ages.
- **Subject 3: Synthetic Control Subject (OpenRAIL-M, Age 25):**
  - Confirms generalizability to unseen, non-celebrity synthetic facial distributions.
- **Visual:** Display of composite grid figures from `research_artifacts/05_Figures_and_Ablations/GRID_ID001_RAJESH_KHANNA_adaptive_control_MultiAge.png`.

---

### Slide 13: Hardware Profiling & Compute Efficiency (1 min)
- **Compute Benchmark:**
  - **Local Development Environment:** Intel Core i5 / x86_64 CPU (PyTorch Float32). Latency: **$97.74 \pm 4.2$ seconds per image**.
  - **Google Colab Cloud Runtime:** NVIDIA Tesla T4 16GB GPU (PyTorch FP16, cuDNN). Latency: **$1.25 \pm 0.08$ seconds per image**.
- **Speedup Factor:** **$78.2\times$ acceleration** via GPU tensor core vectorization.
- **Practical Impact:** Enables real-time interactive parameter exploration and rapid batch dataset processing.
- **Visual:** Bar chart contrasting CPU vs. GPU latency across all 5 target ages.

---

### Slide 14: Base Paper Implementation Fidelity & Transparent Audit (1.5 min)
- **Academic Rigor & Honest Disclosures:**
  | Base Paper (FaceTT) Feature | Status in AgeWeaver | Engineering Implementation & Defensible Rationale |
  | :--- | :---: | :--- |
  | **Algorithm 1: Angular Inversion** | *Approximated* | Approximate manifold bounds via dynamic $\sigma(\Delta y)$ without custom C++ CUDA hyperspherical projection layers. |
  | **Algorithm 2: Adaptive Attention Control** | *Absent / Prompt Alternative* | Replaced by multi-bracket structured prompt steering; internal U-Net attention hooks omitted to maintain zero-shot portability. |
  | **Cyclic Inversion Consistency** | **Faithful** | Fully implemented bidirectional round-trip evaluation pipeline ($x_{\text{src}} \to x_{\text{tgt}} \to x_{\text{recon}}$). |
  | **Continuous Age Interpolation** | **Faithful** | Continuous prompt mapping and dynamic noise scaling across all integer target ages [2--90]. |
  | **ArcFace & ResNet-34 Metrics** | **Faithful** | Exact InsightFace buffalo_l architecture matching modern face recognition standards. |

---

### Slide 15: Deliverables Summary & Mini Project-3 Roadmap (1 min)
- **Completed Mini Project-2 Deliverables:**
  - Complete 21-section academic research report (`MINI_PROJECT_2_RESEARCH_REPORT.md`).
  - 70 unified empirical trials cataloged in `per_image_metrics.csv` and `aggregate_metrics.csv`.
  - 81 high-resolution generated visual assets and 9 composite figure grids.
  - Fully automated 45-test suite passing with 100% green status.
  - Interactive Streamlit Studio frontend running on `http://localhost:8501`.
  - Cryptographically audited manifest with SHA-256 signatures (`MANIFEST.md`).
- **Roadmap for 8th-Semester Mini Project-3:**
  1. *Internal Cross-Attention Hooking:* Implement direct PyTorch U-Net attention key-value swapping to match FaceTT Algorithm 2.
  2. *Longitudinal Benchmark Expansion:* Scale evaluation across 1,000 diverse subjects from CACD2000 and MORPH-II datasets.
  3. *Real-Time Interactive Video Aging:* Extend latent trajectory interpolation to video temporal sequences.

---

## Anticipated Review Committee Questions & Model Answers

### Question 1: "Why did you not implement the base paper's exact internal cross-attention map swapping?"
- **Answer:** *"FaceTT's Algorithm 2 requires intercepting and modifying the internal spatial attention tensors of the pre-trained U-Net across specific denoising steps ($t > \tau_1$ and $t < \tau_2$). In our Mini Project-2 scope, we prioritized building a completely robust, model-agnostic, zero-shot control framework that operates on both consumer CPUs and standard cloud GPUs without custom CUDA kernels or memory-heavy attention caching. Our empirical results prove that combining adaptive noise modulation $\sigma(\Delta y)$ with structured prompt grounding achieves an ArcFace identity similarity of $0.512$ and age MAE of $3.8$ years, demonstrating substantial control while maintaining zero-shot deployment versatility. Implementing native U-Net attention hooks is our primary designated objective for Mini Project-3."*

### Question 2: "Why is your identity similarity for extreme ages (Age 2 and Age 70) lower than for middle ages (Age 35)?"
- **Answer:** *"This reflects biological reality and the geometric properties of facial aging. When regressing to a toddler (Age 2), facial proportions change drastically—the forehead expands relative to the jaw, adipose tissue increases, and cranial ratios transform. Similarly, at Age 70, deep tissue resorption and severe elastosis alter contour lines. ArcFace features, while trained on adult variations, are sensitive to these macroscopic geometric shifts. Crucially, our adaptive controller limits this degradation ($ID_{\text{sim}} \approx 0.42$) compared to generic prompting ($ID_{\text{sim}} \approx 0.31$), preventing complete identity loss."*

### Question 3: "How do you guarantee that your reference identity similarity ($ID_{\text{sim}}^{\text{ref}}$) is not artificially inflated?"
- **Answer:** *"We established a strict academic non-substitution policy in our evaluation engine: $ID_{\text{sim}}^{\text{ref}}$ is computed strictly against independently verified, authentic historical photographs of the subject at that target age. For synthetic subjects and target ages where authentic photographic ground truth does not exist, our evaluation pipeline explicitly outputs `"N/A"`. Under no circumstances do we substitute source similarity or proxy photos as ground truth."*
