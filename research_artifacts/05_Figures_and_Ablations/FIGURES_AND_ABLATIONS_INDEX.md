# AgeWeaver: Master Figures & Visual Ablations Index

**Document ID:** `research_artifacts/05_Figures_and_Ablations/FIGURES_AND_ABLATIONS_INDEX.md`  
**Date of Audit:** 18 October 2026  
**Total Cataloged Visual Figures & Plots:** 21 publication-grade figures, rendered tables, and standardized multi-panel grids  
**Storage Directory:** [`research_artifacts/05_Figures_and_Ablations/`](file:///d:/Academics/IIIT%20Dharwad%20Exams/SEMESTER-7/AgeWeaver/research_artifacts/05_Figures_and_Ablations)  
**Tables Storage Directory:** [`research_artifacts/04_Metrics_and_Tables/`](file:///d:/Academics/IIIT%20Dharwad%20Exams/SEMESTER-7/AgeWeaver/research_artifacts/04_Metrics_and_Tables)

---

## 1. Executive Summary & Figure Architecture

This index provides a complete, traceable catalog of all scientific figures, multi-age comparison strips, hyperparameter sensitivity curves, and rendered empirical tables produced during the AgeWeaver research campaign across local CPU and Google Colab GPU runtimes.

These visual assets directly address every figure requirement specified in *Face Time Traveller: Travel Through Ages Without Losing Identity* (Purbayan Kar et al., arXiv:2602.22819v1, 2026):
- **Section 4.1 & Figure 6:** SOTA Visual Comparisons & Standardized Age Progression Strips.
- **Section 4.2 & Figure 9:** Identity Retention Curves, Cyclic Round-Trip Consistency, and Short-Range Aging.
- **Section 4.3 & Figures 10, 12:** 2x2 Component Ablations, Pareto Frontier, and Condition Heatmaps.
- **Supplementary S1–S7 & Figures S8–S10:** Hyperparameter Sensitivity Curves (\(\xi\), \(\eta_{\text{th}}\), \(\tau_1, \tau_2\)), Inversion & Editing Comparisons, and Artifact Elimination Profiles.

---

## 2. Standardized Multi-Age Progression Strips (Original Input | Age 2 | Age 12 | Age 35 | Age 50 | Age 70)

### Figure 6A: Rajesh Khanna Standardized 6-Panel Strip
- **Asset File:** [`FIG_06_MAIN_RAJESH_KHANNA_STANDARDIZED_GRID.png`](file:///d:/Academics/IIIT%20Dharwad%20Exams/SEMESTER-7/AgeWeaver/research_artifacts/05_Figures_and_Ablations/FIG_06_MAIN_RAJESH_KHANNA_STANDARDIZED_GRID.png)
- **Subject:** `ID001_RAJESH_KHANNA` (GODL-India Government of India Archive)
- **Source Age:** Chronological Age 28 (Young Adult Baseline)
- **Panels (6-Panel Composite):** `[Original Input 28] | [Target 2] | [Target 12] | [Target 35] | [Target 50] | [Target 70]`
- **Key Empirical Results:** Target 35 \(ID_{\text{sim}} = 0.7443\), Target 50 \(ID_{\text{sim}} = 0.6238\), Target 70 \(ID_{\text{sim}} = 0.4208\); Toddler regression at Target 2 achieves soft juvenile contouring.

### Figure 6B: Deepika Padukone Standardized 6-Panel Strip
- **Asset File:** [`FIG_06_DEEPIKA_PADUKONE_STANDARDIZED_GRID.png`](file:///d:/Academics/IIIT%20Dharwad%20Exams/SEMESTER-7/AgeWeaver/research_artifacts/05_Figures_and_Ablations/FIG_06_DEEPIKA_PADUKONE_STANDARDIZED_GRID.png)
- **Subject:** `ID009_DEEPIKA_PADUKONE` (CC BY-SA 3.0 Public Archive)
- **Source Age:** Chronological Age 22 (Young Female Baseline)
- **Panels (6-Panel Composite):** `[Original Input 22] | [Target 2] | [Target 12] | [Target 35] | [Target 50] | [Target 70]`
- **Key Empirical Results:** Preserves feminine craniofacial bone structure, ocular spacing, and jawline across all 5 target ages; peak adult identity \(ID_{\text{sim}} = 0.7697\).

### Figure 11: Rajesh Khanna High-Resolution Aging Strip
- **Asset File:** [`FIG_11_RK_AGING_STRIP.png`](file:///d:/Academics/IIIT%20Dharwad%20Exams/SEMESTER-7/AgeWeaver/research_artifacts/05_Figures_and_Ablations/FIG_11_RK_AGING_STRIP.png)
- **Platform:** Google Colab Tesla T4 GPU (FP16, DPMSolver++ Karras 15 steps)
- **Panels:** Source 28, Target 2, Target 12, Target 35, Target 50, Target 70.

### Figure 12: Deepika Padukone High-Resolution Aging Strip
- **Asset File:** [`FIG_12_DP_AGING_STRIP.png`](file:///d:/Academics/IIIT%20Dharwad%20Exams/SEMESTER-7/AgeWeaver/research_artifacts/05_Figures_and_Ablations/FIG_12_DP_AGING_STRIP.png)
- **Platform:** Google Colab Tesla T4 GPU (FP16, DPMSolver++ Karras 15 steps)
- **Panels:** Source 22, Target 2, Target 12, Target 35, Target 50, Target 70.

---

## 3. Comparative & Ablation Grids

### Figure 13: Three-Condition Comparative Grid (Baseline vs. Structured vs. Adaptive)
- **Asset File:** [`FIG_13_THREE_CONDITION_COMPARISON.png`](file:///d:/Academics/IIIT%20Dharwad%20Exams/SEMESTER-7/AgeWeaver/research_artifacts/05_Figures_and_Ablations/FIG_13_THREE_CONDITION_COMPARISON.png)
- **Structure:** 3x3 matrix comparing Target Ages (Rows: Age 2, Age 35, Age 70) across Conditioning Arms (Columns: Generic Baseline vs. Structured Prompt vs. Adaptive Control).
- **Key Insight:** Shows the progressive elimination of feature drift and texture wash from left to right.

### Figure 14: Scheduler Modernization & Artifact Elimination
- **Asset File:** [`FIG_14_SCHEDULER_COMPARISON.png`](file:///d:/Academics/IIIT%20Dharwad%20Exams/SEMESTER-7/AgeWeaver/research_artifacts/05_Figures_and_Ablations/FIG_14_SCHEDULER_COMPARISON.png)
- **Comparison:** PNDM Scheduler (patchy high-frequency texture corruption) vs. DPMSolver++ 2M Karras (clean photorealistic skin pore synthesis in 15 steps).

---

## 4. Quantitative Scientific Plots & Distributions

### Figure 1: Aggregate Method Comparison Bar Chart
- **Asset File:** [`FIG_01_METHOD_COMPARISON_BAR.png`](file:///d:/Academics/IIIT%20Dharwad%20Exams/SEMESTER-7/AgeWeaver/research_artifacts/05_Figures_and_Ablations/FIG_01_METHOD_COMPARISON_BAR.png)
- **Metrics Plotted:** Mean ArcFace \(ID_{\text{sim}}\), Target-Age MAE (years), and Mean Inference Latency across Generic Baseline, Structured Prompt, and Adaptive Control (N=70).

### Figure 2: Identity Similarity vs. Target Age Scatter
- **Asset File:** [`FIG_02_IDSIM_VS_TARGET_AGE.png`](file:///d:/Academics/IIIT%20Dharwad%20Exams/SEMESTER-7/AgeWeaver/research_artifacts/05_Figures_and_Ablations/FIG_02_IDSIM_VS_TARGET_AGE.png)
- **Analysis:** Displays identity retention trajectories across target ages [2--70]. Demonstrates how Adaptive Control maintains identity above 0.50 for adult transitions.

### Figure 3: Target Age MAE vs. Target Age Distribution
- **Asset File:** [`FIG_03_AGE_MAE_VS_TARGET_AGE.png`](file:///d:/Academics/IIIT%20Dharwad%20Exams/SEMESTER-7/AgeWeaver/research_artifacts/05_Figures_and_Ablations/FIG_03_AGE_MAE_VS_TARGET_AGE.png)
- **Analysis:** Evaluates ResNet-34 estimated age error across childhood, adult, and senior age bands.

### Figure 4: CPU vs. GPU Latency Profiling
- **Asset File:** [`FIG_04_CPU_VS_GPU_LATENCY.png`](file:///d:/Academics/IIIT%20Dharwad%20Exams/SEMESTER-7/AgeWeaver/research_artifacts/05_Figures_and_Ablations/FIG_04_CPU_VS_GPU_LATENCY.png)
- **Analysis:** Local CPU (97.74s) vs. Google Colab Tesla T4 GPU (1.25s) demonstrating a **78.2x hardware speedup**.

### Figure 5: Colab GPU Sweep Age-wise Breakdown
- **Asset File:** [`FIG_05_COLAB_AGEWISE_BREAKDOWN.png`](file:///d:/Academics/IIIT%20Dharwad%20Exams/SEMESTER-7/AgeWeaver/research_artifacts/05_Figures_and_Ablations/FIG_05_COLAB_AGEWISE_BREAKDOWN.png)
- **Analysis:** Detailed breakdown of \(ID_{\text{sim}}\) and MAE across Target Ages 2, 12, 35, 50, 70 under Adaptive Control on GPU.

### Figure 6C: Deterministic Adaptive Parameter Scaling Curves
- **Asset File:** [`FIG_06_ADAPTIVE_SCALING_CURVES.png`](file:///d:/Academics/IIIT%20Dharwad%20Exams/SEMESTER-7/AgeWeaver/research_artifacts/05_Figures_and_Ablations/FIG_06_ADAPTIVE_SCALING_CURVES.png)
- **Formulation:** Graphs of dynamic noise strength \(\sigma(|\Delta y|)\) and guidance scale \(s(|\Delta y|)\) over biological displacement \(|\Delta y| \in [0, 70]\).

### Figure 7: Round-Trip Cyclic Identity Consistency Scatter
- **Asset File:** [`FIG_07_CYCLIC_SIMILARITY.png`](file:///d:/Academics/IIIT%20Dharwad%20Exams/SEMESTER-7/AgeWeaver/research_artifacts/05_Figures_and_Ablations/FIG_07_CYCLIC_SIMILARITY.png)
- **Analysis:** Round-trip reconstructibility \(S_{\text{cyc}}\) color-mapped against forward identity similarity across valid cyclic trials.

### Figure 8: Noise Strength vs. Identity Similarity Trade-off
- **Asset File:** [`FIG_08_STRENGTH_VS_IDSIM.png`](file:///d:/Academics/IIIT%20Dharwad%20Exams/SEMESTER-7/AgeWeaver/research_artifacts/05_Figures_and_Ablations/FIG_08_STRENGTH_VS_IDSIM.png)
- **Analysis:** Empirically proves that increasing \(\sigma > 0.58\) causes identity degradation, justifying our bounded ceiling \(\sigma_{\max} = 0.58\).

### Figure 9: Ablation Heatmap (Condition x Target Age)
- **Asset File:** [`FIG_09_ABLATION_HEATMAP.png`](file:///d:/Academics/IIIT%20Dharwad%20Exams/SEMESTER-7/AgeWeaver/research_artifacts/05_Figures_and_Ablations/FIG_09_ABLATION_HEATMAP.png)
- **Structure:** 2D heatmap showing \(ID_{\text{sim}}\) across the 3 experimental regimes and 5 target ages.

### Figure 10: Pareto Frontier (Age Accuracy vs. Identity Retention)
- **Asset File:** [`FIG_10_PARETO_FRONTIER.png`](file:///d:/Academics/IIIT%20Dharwad%20Exams/SEMESTER-7/AgeWeaver/research_artifacts/05_Figures_and_Ablations/FIG_10_PARETO_FRONTIER.png)
- **Analysis:** Scatter of \(ID_{\text{sim}}\) vs. Target Age MAE with ideal corner annotation showing Adaptive Control nearest the optimal trade-off boundary.

---

## 5. Supplementary Hyperparameter Sensitivity Analyses (FaceTT Figures S8–S10)

### Figure S8: Angular Inversion Hyperparameter Sensitivity (\(\xi\))
- **Asset File:** [`FIG_S08_XI_SENSITIVITY.png`](file:///d:/Academics/IIIT%20Dharwad%20Exams/SEMESTER-7/AgeWeaver/research_artifacts/05_Figures_and_Ablations/FIG_S08_XI_SENSITIVITY.png)
- **Parameter:** \(\xi \in [0.1, 0.5, 1.2, 2.0]\) (FaceTT Algorithm 1).
- **Findings:** Low \(\xi\) yields insufficient trajectory constraint; high \(\xi\) restricts age progression; optimal balance at \(\xi = 1.2\).

### Figure S9: Adaptive Attention Control Threshold Sensitivity (\(\eta_{\text{th}}\))
- **Asset File:** [`FIG_S09_ETA_TH_SENSITIVITY.png`](file:///d:/Academics/IIIT%20Dharwad%20Exams/SEMESTER-7/AgeWeaver/research_artifacts/05_Figures_and_Ablations/FIG_S09_ETA_TH_SENSITIVITY.png)
- **Parameter:** \(\eta_{\text{th}} \in [0.03, 0.04, 0.05, 0.06]\) (FaceTT Algorithm 2).
- **Findings:** Governs KL-divergence attention map blending; optimal balance at \(\eta_{\text{th}} = 0.05\).

### Figure S10: Attention Control Step Schedule Sensitivity (\(\tau_1, \tau_2\))
- **Asset File:** [`FIG_S10_SCHEDULE_SENSITIVITY.png`](file:///d:/Academics/IIIT%20Dharwad%20Exams/SEMESTER-7/AgeWeaver/research_artifacts/05_Figures_and_Ablations/FIG_S10_SCHEDULE_SENSITIVITY.png)
- **Parameter Pairs:** \((\tau_1, \tau_2) \in [(25,15), (30,15), (35,15), (40,15), (35,5), (35,10), (35,20)]\).
- **Findings:** Confirms that \((\tau_1=35, \tau_2=15)\) minimizes target-age MAE while preserving self-attention structure.

---

## 6. Publication-Rendered Benchmark Tables (PNG & CSV)

| Table ID | CSV Source | Rendered PNG Asset | Focus |
| :--- | :--- | :--- | :--- |
| **Table 1** | [`TABLE_4_1_CELEBA_HQ.csv`](file:///d:/Academics/IIIT%20Dharwad%20Exams/SEMESTER-7/AgeWeaver/research_artifacts/04_Metrics_and_Tables/TABLE_4_1_CELEBA_HQ.csv) | [`TABLE_4_1_CELEBA_HQ.png`](file:///d:/Academics/IIIT%20Dharwad%20Exams/SEMESTER-7/AgeWeaver/research_artifacts/04_Metrics_and_Tables/TABLE_4_1_CELEBA_HQ.png) | CelebA-HQ Young-to-60 SOTA Comparison (FaceTT vs. HRFAE vs. CUSP vs. FADING vs. AgeWeaver). |
| **Table 2** | [`TABLE_4_1_FFHQ_AGING.csv`](file:///d:/Academics/IIIT%20Dharwad%20Exams/SEMESTER-7/AgeWeaver/research_artifacts/04_Metrics_and_Tables/TABLE_4_1_FFHQ_AGING.csv) | [`TABLE_4_1_FFHQ_AGING.png`](file:///d:/Academics/IIIT%20Dharwad%20Exams/SEMESTER-7/AgeWeaver/research_artifacts/04_Metrics_and_Tables/TABLE_4_1_FFHQ_AGING.png) | FFHQ-Aging 10-Age-Group MAE Across Methods. |
| **Table 3** | [`TABLE_4_2_IDENTITY_SIMILARITY.csv`](file:///d:/Academics/IIIT%20Dharwad%20Exams/SEMESTER-7/AgeWeaver/research_artifacts/04_Metrics_and_Tables/TABLE_4_2_IDENTITY_SIMILARITY.csv) | [`TABLE_4_2_IDENTITY_SIMILARITY.png`](file:///d:/Academics/IIIT%20Dharwad%20Exams/SEMESTER-7/AgeWeaver/research_artifacts/04_Metrics_and_Tables/TABLE_4_2_IDENTITY_SIMILARITY.png) | Identity Similarity (Cyclic & Reference) with Strict Non-Substitution Disclosure. |
| **Table 4** | [`TABLE_4_BIOMETRIC_FNMR.csv`](file:///d:/Academics/IIIT%20Dharwad%20Exams/SEMESTER-7/AgeWeaver/research_artifacts/04_Metrics_and_Tables/TABLE_4_BIOMETRIC_FNMR.csv) | [`TABLE_4_BIOMETRIC_FNMR.png`](file:///d:/Academics/IIIT%20Dharwad%20Exams/SEMESTER-7/AgeWeaver/research_artifacts/04_Metrics_and_Tables/TABLE_4_BIOMETRIC_FNMR.png) | Biometric Verification FNMR at FMR = 0.01% and 0.1% Across 6 Aging Distances. |
| **Table 5** | [`TABLE_4_3_ABLATION_STUDY.csv`](file:///d:/Academics/IIIT%20Dharwad%20Exams/SEMESTER-7/AgeWeaver/research_artifacts/04_Metrics_and_Tables/TABLE_4_3_ABLATION_STUDY.csv) | [`TABLE_4_3_ABLATION_STUDY.png`](file:///d:/Academics/IIIT%20Dharwad%20Exams/SEMESTER-7/AgeWeaver/research_artifacts/04_Metrics_and_Tables/TABLE_4_3_ABLATION_STUDY.png) | 2x2 Component Ablation (Angular Inversion x AAC) on CelebA-HQ. |
| **Table S4** | [`TABLE_S4_INVERSION_EDITING.csv`](file:///d:/Academics/IIIT%20Dharwad%20Exams/SEMESTER-7/AgeWeaver/research_artifacts/04_Metrics_and_Tables/TABLE_S4_INVERSION_EDITING.csv) | [`TABLE_S4_INVERSION_EDITING.png`](file:///d:/Academics/IIIT%20Dharwad%20Exams/SEMESTER-7/AgeWeaver/research_artifacts/04_Metrics_and_Tables/TABLE_S4_INVERSION_EDITING.png) | Comparative Analysis of 5 Inversion & 5 Diffusion Editing Techniques. |
| **Summary Table** | [`aggregate_by_condition.csv`](file:///d:/Academics/IIIT%20Dharwad%20Exams/SEMESTER-7/AgeWeaver/research_artifacts/04_Metrics_and_Tables/aggregate_by_condition.csv) | [`TABLE_METHOD_COMPARISON_RENDERED.png`](file:///d:/Academics/IIIT%20Dharwad%20Exams/SEMESTER-7/AgeWeaver/research_artifacts/04_Metrics_and_Tables/TABLE_METHOD_COMPARISON_RENDERED.png) | AgeWeaver Full Empirical Corpus Method Comparison (N=70). |
| **Age-Wise Table** | [`agewise_adaptive_colab.csv`](file:///d:/Academics/IIIT%20Dharwad%20Exams/SEMESTER-7/AgeWeaver/research_artifacts/04_Metrics_and_Tables/agewise_adaptive_colab.csv) | [`TABLE_AGEWISE_RESULTS_RENDERED.png`](file:///d:/Academics/IIIT%20Dharwad%20Exams/SEMESTER-7/AgeWeaver/research_artifacts/04_Metrics_and_Tables/TABLE_AGEWISE_RESULTS_RENDERED.png) | Adaptive Control Performance by Target Age on GPU. |

---

## 7. Verification & Reproducibility Guarantee

All 21 figures and rendered tables are reproducible from raw numerical experiment data using the following master scripts:
1. `python scratch/generate_all_figures.py`
2. `python scratch/generate_facett_benchmarks_and_grids.py`

Every figure has an associated SHA-256 cryptographic signature in [`research_artifacts/MANIFEST.md`](file:///d:/Academics/IIIT%20Dharwad%20Exams/SEMESTER-7/AgeWeaver/research_artifacts/MANIFEST.md).
