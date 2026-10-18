# AgeWeaver: Future Research & Engineering Work Queue
**Document ID:** `research_artifacts/08_Future_Work/FUTURE_WORK_QUEUE.md`  
**Date of Audit:** 17 October 2026  
**Directory:** `research_artifacts/08_Future_Work/`  
**Academic Horizon:** Transition from Mini Project-2 to Mini Project-3 & Conference Submission

---

## 1. Prioritization & Engineering Roadmap

The tasks below represent the systematic continuation of AgeWeaver's research trajectory, addressing the remaining partial, blocked, and architectural gaps identified in the [Master Experimental Tracker](file:///d:/Academics/IIIT%20Dharwad%20Exams/SEMESTER-7/AgeWeaver/research_artifacts/00_Project_Tracking/FACETT_MASTER_TRACKER.md).

```
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 2.5 (Immediate GPU Execution - Oct/Nov 2026)                    │
│ ├── TSK-01: Full 10-Bracket Multi-Age Sweep on Google Colab GPU        │
│ ├── TSK-02: Automated Multi-Seed Face Diversity Metric (LPIPS/ArcFace) │
│ └── TSK-03: Sourcing Young References for Candidate Manifest (98 IDs)  │
├────────────────────────────────────────────────────────────────────────┤
│ Phase 3 (Advanced Architecture & Comparative SOTA - Mini Project-3)    │
│ ├── TSK-04: Longitudinal Ground-Truth Target-Age Reference Pairs       │
│ ├── TSK-05: Implementation of Spherical Geodesic Angular Inversion     │
│ ├── TSK-06: Cross-Attention Map Injection in U-Net Decoder (L3–L16)    │
│ └── TSK-07: Comparative Evaluation Against Public SOTA Checkpoints     │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Detailed Task Specifications & Acceptance Criteria

### TSK-01: Full 10-Bracket Multi-Age GPU Sweep (FaceTT Sec 4.2 / Tab 2)
- **Priority:** `HIGH (P1)`
- **Current Status:** `PARTIAL` (5 key milestone ages evaluated: 2, 12, 35, 50, 70).
- **Objective:** Execute systematic aging across all 10 standard demographic brackets:
  $$[0\text{--}2, \, 3\text{--}6, \, 7\text{--}9, \, 10\text{--}14, \, 15\text{--}19, \, 20\text{--}29, \, 30\text{--}39, \, 40\text{--}49, \, 50\text{--}69, \, 70+]$$
- **Implementation Strategy:** Execute `scripts/run_colab_benchmark.py` on Google Colab T4 GPU across active identities (`ID001`, `ID009`, `SYNTH_ID_2024`, `UPLOAD_HRK`).
- **Acceptance Criteria:**
  1. Complete $10 \times 4 = 40$ GPU trials logged in `experiment_log.csv` and `experiment_log.jsonl`.
  2. Compute mean target-age MAE and ArcFace $ID_{\text{sim}}^{\text{src}}$ for every age bracket.
  3. Export publication-quality 10-panel visual grids into `outputs/experiments/grids/`.

### TSK-02: Automated Face Diversity Scoring via LPIPS (FaceTT Sec 4.1 / Tab 1)
- **Priority:** `HIGH (P1)`
- **Current Status:** `PARTIAL` (Multiple random seeds evaluated qualitatively).
- **Objective:** Quantify visual diversity among generated images across 5 stochastic seeds for the exact same target age prompt:
  $$\text{Diversity}(y_{\text{tgt}}) = \frac{1}{\binom{N}{2}} \sum_{i < j} \text{LPIPS}(I_{\text{tgt}}^{(s_i)}, I_{\text{tgt}}^{(s_j)})$$
- **Implementation Strategy:** Author `scripts/evaluate_diversity.py` integrating `torchvision` and `lpips` AlexNet/VGG perceptual distance.
- **Acceptance Criteria:**
  1. Automated CLI script computing average pairwise LPIPS score across seeds 41, 42, 49, 53, 100.
  2. Verified output table demonstrating diversity without identity degradation ($ID_{\text{sim}}^{\text{src}} \ge 0.40$).

### TSK-03: Sourcing Young References for Candidate Manifest (98 IDs)
- **Priority:** `MEDIUM (P2)`
- **Current Status:** `BLOCKED` (98 entries cataloged in manifest, awaiting young reference vetting).
- **Objective:** Expand active benchmark set from 2 celebrity identities to 20 vetted diverse identities across 10 demographic groups.
- **Implementation Strategy:** Author an automated downloader with SCRFD face detection gating to filter out low-resolution, occluded, or non-frontal images from Wikimedia Commons.
- **Acceptance Criteria:**
  1. Downloaded images verified with SHA-256 hashes in `data/download_log.json`.
  2. SCRFD face detection confidence $\ge 0.75$.
  3. Calibrated young reference age between 18 and 30 years.

### TSK-04: Longitudinal Ground-Truth Target-Age Pairs (FaceTT Sec 4.1 / Tab 1)
- **Priority:** `MEDIUM (P2)`
- **Current Status:** `BLOCKED` (Strict non-substitution policy currently logs $ID_{\text{sim}}^{\text{ref}}$ as `"N/A"`).
- **Objective:** Acquire authentic historical portraits of identical subjects captured at target ages (e.g., Rajesh Khanna at age 50 and age 70; Deepika Padukone at age 35) to measure true ground-truth identity alignment:
  $$ID_{\text{sim}}^{\text{ref}} = \cos(\mathcal{E}_{\text{ID}}(I_a^{\text{real}}), \mathcal{E}_{\text{ID}}(I_a^{\text{synth}}))$$
- **Acceptance Criteria:**
  1. Archival sourcing with verified date-of-capture metadata.
  2. Automated computation of non-null $ID_{\text{sim}}^{\text{ref}}$ logged in evaluation tables.

### TSK-05: Implementation of Spherical Geodesic Angular Inversion (FaceTT Sec 5.1 & S5)
- **Priority:** `MEDIUM-LOW (P3)`
- **Current Status:** `NOT RUN` (Architectural departure: AgeWeaver currently uses bounded latent initialization).
- **Objective:** Implement true spherical trajectory angular inversion in PyTorch/Diffusers to project latent trajectories onto the unit hypersphere:
  $$\tilde{z}_{t-1} = \frac{\|z_{t-1}\|}{\|z_t\|} \cdot \left(\cos(\Delta \theta) z_t + \sin(\Delta \theta) \frac{z_{t-1} - \cos(\Delta \theta) z_t}{\|z_{t-1} - \cos(\Delta \theta) z_t\|}\right)$$
- **Acceptance Criteria:**
  1. Custom Diffusers scheduler implementing angular geodesic step.
  2. Direct mathematical comparison against DDIM on cyclic reconstruction error.

### TSK-06: Cross-Attention Decoder Hooking (FaceTT Suppl S1 & S1.1)
- **Priority:** `LOW (P4)`
- **Current Status:** `NOT RUN` (Documented as advanced interpretability topic).
- **Objective:** Probe and modify intermediate cross-attention matrices in Stable Diffusion U-Net decoder blocks (L3 through L16) using PyTorch forward hooks.
- **Acceptance Criteria:**
  1. Module dumping spatial cross-attention heatmaps for prompt tokens `"young"`, `"wrinkles"`, `"silver hair"`.
  2. Evaluation of attention map swapping vs. prompt-based modulation.

---

## 3. Resource & Compute Allocation Budget

| Project Milestone | Compute Target | Estimated GPU Hours | Storage Budget | Primary Deliverable |
| :--- | :---: | :---: | :---: | :--- |
| **Phase 2.5 (Batch GPU Sweep)** | Google Colab T4 | $\sim 3\text{--}5$ GPU hrs | $\sim 500$ MB | Full 10-bracket benchmark table and composite strips |
| **Phase 3 (Attention & Inversion)** | Dedicated RTX 4090 / A100 | $\sim 20$ GPU hrs | $\sim 5$ GB | Custom scheduler & attention-swapping ablation |
| **Final Conference Paper** | Academic Cluster / Colab Pro | $\sim 50$ GPU hrs | $\sim 15$ GB | Large-scale 100-identity demographic evaluation |

