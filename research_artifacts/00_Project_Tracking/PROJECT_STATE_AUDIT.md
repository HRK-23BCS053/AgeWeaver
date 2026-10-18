# AgeWeaver Project State & Infrastructure Audit
**Document ID:** `research_artifacts/00_Project_Tracking/PROJECT_STATE_AUDIT.md`  
**Date of Audit:** 18 October 2026, 00:48 IST  
**Author:** Lead Research Engineer & Verification Framework  
**Project:** AgeWeaver — 7th-Semester B.Tech CSE Mini Project-2  
**Target Base Paper:** *Face Time Traveller: Travel Through Ages Without Losing Identity*, Purbayan Kar et al., arXiv:2602.22819v1, 26 Feb 2026  
**Primary Submission Deadline:** 18 October 2026, 9:00 PM IST (~20 hours remaining)  

---

## 1. Executive Summary & Verification Overview

This audit establishes the rigorous, empirical baseline of all code, datasets, experiment records, figures, metrics, and models in the AgeWeaver repository as of October 18, 2026. Every item reported here has been inspected directly on disk or in Google Drive.

```
Total Empirical Experimental Trials Verified: 70
├── Local CPU Pilot Experiments: 40 trials (100% matched in CSV, JSONL, and PNG images)
└── Colab GPU T4 Age Sweep:     30 trials (100% matched in CSV, JSONL, and PNG images)
```

---

## 2. Inventory of What Actually Exists & Has Been Independently Verified

### 2.1 Codebase & Test Suite
- **Core Library (`src/ageweaver/`):**
  - `config.py`: YAML configuration loader (`configs/default.yaml`).
  - `dataset.py`: Image loading, landmark cropping, bicubic resizing, square padding.
  - `evaluation.py`: Evaluator computing ArcFace cosine identity similarity and ResNet-34 Age MAE.
  - `experiment.py`: Experiment runner for 3-arm conditioning (Generic Baseline, Structured Prompt, Adaptive Control) and cyclic inversion round-trip.
  - `identity.py`: InsightFace wrapper extracting 512-d normalized L2 embeddings and detecting faces via SCRFD.
  - `pipeline.py`: Stable Diffusion v1-5 latent img2img wrapper with `DPMSolverMultistepScheduler` (Karras sigmas) and dynamic parameter equations $\sigma(\Delta y)$ and $s(\Delta y)$.
  - `prompts.py`: Continuous age prompt builder mapping target age integers to calibrated anatomical tokens.
  - `frontend/`: Interactive Streamlit application (`studio.py`, `state.py`, `styles.py`, `metrics_view.py`, `about_view.py`).
- **Automated Test Suite (`tests/`):**
  - 7 test modules: `test_config.py`, `test_dataset.py`, `test_dataset_manifest.py`, `test_metrics.py`, `test_pipeline.py`, `test_prompts.py`, `test_studio_integrity.py`.
  - **Status:** **45 / 45 tests passing** (verified via `.venv\Scripts\python.exe -m unittest discover tests` in 5.54s).

### 2.2 Datasets & Manifests (`data/`)
- **`data/dataset_manifest.csv`:** 100 cataloged candidate identities across 10 demographic groups, verified schema.
- **Downloaded Image Assets:**
  - `data/images/ID001_RAJESH_KHANNA.jpg`: Verified young reference (Capture c. 1970, age 28, GODL-India license, SHA-256 `702bd522419f1ccd`).
  - `data/images/ID009_DEEPIKA_PADUKONE.jpg`: Verified young reference (Capture c. 2008, age 22, CC BY-SA 3.0 license, SHA-256 `d6f644e66d13d335`).
- **Synthetic Identity Baseline:**
  - `SYNTH_ID_2024`: Generated via SD 1.5 text-to-image (age 23 baseline) for privacy-preserving ablation.
- **User Upload Benchmark Portrait:**
  - `UPLOAD_HRK_ea7e3a97`: Custom test portrait processed through Studio frontend (estimated age 19).

### 2.3 Experimental Records & Generated Images
1. **Local CPU Pilot Trials (`outputs/experiments/`):**
   - Exactly **40 completed trials** in `experiment_log.csv` and `experiment_log.jsonl` in 1-to-1 parity.
   - All 40 output PNG images verified on local storage.
   - 4 active subjects: `ID001_RAJESH_KHANNA` (23 trials), `SYNTH_ID_2024` (8 trials), `UPLOAD_HRK_ea7e3a97` (5 trials), `ID009_DEEPIKA_PADUKONE` (4 trials).
   - Multi-age composite grid: [`GRID_ID001_RAJESH_KHANNA_adaptive_control_MultiAge.png`](file:///d:/Academics/IIIT%20Dharwad%20Exams/SEMESTER-7/AgeWeaver/outputs/experiments/grids/GRID_ID001_RAJESH_KHANNA_adaptive_control_MultiAge.png).
2. **Colab GPU Sweep Trials (`G:\My Drive\AgeWeaver_Research\08_Code_Backups\AgeWeaver_Colab\outputs\colab_age_sweep_20261017/`):**
   - Exactly **30 completed trials** on NVIDIA Tesla T4 GPU in `experiment_log.csv` and `experiment_log.jsonl`.
   - All 30 output PNG images verified in Google Drive.
   - Subjects: `ID001_RAJESH_KHANNA` (15 trials) and `ID009_DEEPIKA_PADUKONE` (15 trials).
   - Milestone Target Ages: 2, 12, 35, 50, 70 across Generic Baseline, Structured Prompt, and Adaptive Control.
   - 10 cyclic reconstruction trials logged ($S_{\text{cyc}} \in [0.43, 0.65]$).
   - Multi-condition grids: `GRID_ID001_RAJESH_KHANNA_generic_baseline_MultiAge.png`, `GRID_ID009_DEEPIKA_PADUKONE_generic_baseline_MultiAge.png`, `ID001_RAJESH_KHANNA_three_condition_comparison.png`, `ID009_DEEPIKA_PADUKONE_three_condition_comparison.png`.

---

## 3. Discrepancy & Gap Analysis

### 3.1 What is Only Reported (vs. Independently Measured)
- **FaceTT Published Paper Values:** All numbers from Kar et al. (arXiv:2602.22819v1) for CelebA-HQ young-to-60, FFHQ-Aging KID, and user study scores are documented in our reports, but are clearly labeled **[FaceTT Reported]**. They are not local measurements.
- **Baseline Models (HRFAE, CUSP, FADING):** The published papers report performance for these baselines. Their official checkpoints have not been re-executed locally due to missing environment weights and time constraints; they are cataloged from official published literature.

### 3.2 What is Missing
1. **Full 70,000-image FFHQ-Aging Dataset:** Fine-tuning Stable Diffusion on 70,000 images at 1024x1024 requires ~50GB storage and multi-day compute on an A100. It is impossible to download and fine-tune within 20 hours on a free Colab T4.
2. **CelebA-HQ 1,000-Image Evaluation Set:** The full 1,000 young images dataset has not been batch-downloaded locally.
3. **Internal U-Net Attention Hooking (Algorithm 2 AAC):** AgeWeaver currently uses an Adaptive Parameter Controller (modulating $\sigma$ and $s$) and prompt interpolation, rather than intercepting internal PyTorch U-Net decoder cross-attention tensors.
4. **Spherical Geodesic Angular Inversion (Algorithm 1):** AgeWeaver uses bounded latent img2img initialization rather than spherical geodesic angles.
5. **Authentic Longitudinal Target-Age Ground-Truth Pairs:** We lack historical photos of Rajesh Khanna at verified age 50 and 70; thus, $ID_{\text{sim}}^{\text{ref}}$ is strictly logged as `"N/A"` in compliance with the non-substitution policy.

### 3.3 What is Broken or Resolved
- **Runtime Error in Studio View:** Previously resolved (`AgeWeaverEvaluator.__init__() got unexpected keyword argument 'face_analyzer'`). All imports and evaluators now operate cleanly.
- **PNDM Noise Patch Artifacts:** Diagnosed and resolved by switching default scheduler to 15-step `DPMSolverMultistepScheduler` with Karras sigmas.

### 3.4 What is Duplicated
- Google Drive `08_Code_Backups/AgeWeaver_Colab` mirrors local codebase files, which is desirable for synchronization. The Colab sweep output is currently in Google Drive and will now be integrated into `research_artifacts/`.

---

## 4. Feasibility Assessment for Primary Submission Deadline (18 Oct 2026, 9:00 PM IST)

| Objective | Feasibility | Execution Route |
| :--- | :---: | :--- |
| **Comprehensive Paper Requirement Matrix (Sec 4, 5, S1–S7)** | **100% Feasible** | Create master CSV and Markdown matrix from exact paper text. |
| **Fidelity Audit of FaceTT Algorithms 1 & 2** | **100% Feasible** | Produce complete audit comparing math to codebase. |
| **Consolidation of All 70 Empirical Trials** | **100% Feasible** | Merge local (40) and Colab GPU (30) trials into unified tables. |
| **Standardized Visual Comparison Grids (Ages 2, 12, 35, 50, 70)** | **100% Feasible** | Already generated for Rajesh Khanna & Deepika Padukone; catalog and index. |
| **Ablation Studies (Conditions, Schedulers, Jump Distances, Seeds)** | **100% Feasible** | Extract and plot from verified empirical data. |
| **Biometric & Quantitative Tables (MAE, $ID_{\text{sim}}$, Runtime, Failure Rate)** | **100% Feasible** | Script automated calculation and export CSV/Markdown tables. |
| **Complete Professor-Ready Research Report & Handover** | **100% Feasible** | Comprehensive report citing real files, hashes, and exact paper mappings. |
| **Full 70k FFHQ-Aging Fine-tuning on A100** | **Infeasible (20h)** | Document protocol, dataset specs, and hardware divergence. |
| **Executing Third-Party Baselines (HRFAE, CUSP, FADING)** | **Infeasible (20h)** | Document official literature metrics in dedicated comparison tables. |

---

## 5. Prioritized Execution Backlog

1. **Phase 1:** Extract every requirement from Section 4, 4.0–4.3, Section 5, and Supplementary S1–S7 into `FACETT_REQUIREMENTS_MASTER.csv` and `FACETT_REQUIREMENTS_MASTER.md`.
2. **Phase 2:** Author `FACETT_IMPLEMENTATION_FIDELITY_AUDIT.md` analyzing Algorithms 1 and 2, Prompt Refinement, and Cyclic Similarity.
3. **Phase 3:** Create dataset statistics, licensing, and provenance reports in `01_Dataset_Audit/`.
4. **Phase 4:** Unify all 70 local and Colab GPU trials into `03_Generated_Images/` and `04_Metrics_and_Tables/`.
5. **Phase 5 & 6:** Build authoritative comparative tables, per-image CSVs, aggregate metrics, and failed-run logs.
6. **Phase 7, 8, 9:** Document ablations, inversion/editing comparisons, and Supplementary S1–S7 checklists.
7. **Phase 10:** Compile final visual figures, grid composites, and architecture diagrams.
8. **Phase 13 & 14:** Author formal 20-section Mini Project-2 Research Report, Slides Outline, and `FINAL_HANDOVER.md`.

