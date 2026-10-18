# AgeWeaver: Blocked Experiments, Architectural Gaps & Unblock Roadmaps
**Document ID:** `research_artifacts/08_Future_Work/BLOCKED_EXPERIMENTS.md`  
**Date of Audit:** 18 October 2026  
**Auditor:** Technical Documentation Lead & ML Experimentation Engineer  
**Academic Guideline:** Strict non-fabrication; exhaustive disclosure of unrun and blocked items.  

---

## 1. Inventory of Blocked and Unreproduced Paper Items

In accordance with **Rule 6** (distinguishing verified results from unrun items) and **Rule 7** (never fabricating measurements), the following table catalogs all experiments from *Face Time Traveller* (Kar et al., 2026) that were not executed in our environment, detailing the exact blockers and mitigation strategies:

| Requirement ID | Paper Section | Experiment Description | Precise Blocker | Mitigation / Handling Strategy | Mini Project-3 Unblock Criteria |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **BLK-01** | **Sec 4.0** | Full FFHQ-Aging Fine-Tuning (70k images at 1024x1024, 150 steps, batch size 2, lr 5e-6) | Requires ~50GB storage, multi-day download, and dedicated NVIDIA A100 GPU cluster. Infeasible within 20-hour deadline. | Zero-shot adaptation on pre-trained SD 1.5 weights; documented training protocol in `training_log.csv`. | Secure access to institutional GPU cluster (RTX 4090 / A100) and download full FFHQ-Aging tarballs. |
| **BLK-02** | **Sec 4.1 / Tab 1** | Local Re-execution of Baseline Checkpoints (HRFAE, CUSP, FADING) | Official baseline codebases rely on conflicting, legacy PyTorch/CUDA environments (PyTorch 1.10, Python 3.7) incompatible with Diffusers 0.40. | Cataloged official published metrics directly from literature in [`paper_reported_results.csv`](file:///d:/Academics/IIIT%20Dharwad%20Exams/SEMESTER-7/AgeWeaver/research_artifacts/04_Metrics_and_Tables/paper_reported_results.csv). | Build containerized Docker environments for each baseline. |
| **BLK-03** | **Sec 4.1 / Fig 7–8** | Direct Execution of Aging Multiverse [12] & MyTimeMachine [30] | Both projects are recent unpublished works; source code and model weights are proprietary and unreleased. | Qualitative audit based on official project webpage demonstration samples. | Monitor public GitHub releases of MyTM and Aging Multiverse. |
| **BLK-04** | **Sec 4.2 / Tab 3** | Reference Identity Similarity ($ID_{\text{sim}}^{\text{ref}}$) on Celebrity Set | Lack of verified, date-authenticated historical photographs of identical celebrity subjects at older target ages (50, 70). | Enforced strict non-substitution policy: $ID_{\text{sim}}^{\text{ref}}$ recorded as `"N/A"`. Evaluated cyclic similarity ($S_{\text{cyc}}$) as primary unsupervised metric. | Curate licensed archival photographs with verified metadata. |
| **BLK-05** | **Sec 4.2 / Tab 4** | Biometric FNMR @ FMR = 0.01% / 0.1% Calibration Curve | Calculating true FNMR requires evaluating ROC curves over $>10^5$ negative impostor pairs on a massive biometric cohort. | Reported direct ArcFace cosine similarity; cataloged paper-reported FNMR values in `paper_reported_results.csv`. | Assemble multi-identity biometric test set on Colab GPU. |
| **BLK-06** | **Sec 4.2 / Fig 11** | Human User Study (35 participants, 14 images, 6 target ages) | Recruiting, vetting, and collecting blinded survey responses from 35 participants was ethically and temporally impossible within 20 hours. | Zero fabricated ratings per Rule 7; documented paper user study findings in `FACETT_BASE_PAPER_SUMMARY.md`. | Conduct IRB-compliant blinded user perception survey during semester 8. |
| **BLK-07** | **Sec 4.3 / Alg 1** | Spherical Geodesic Angular Inversion (Algorithm 1) | Architectural divergence: AgeWeaver operates via bounded latent img2img noise initialization rather than dual-branch angular trajectory scaling. | Analyzed and documented mathematical equations in `FACETT_IMPLEMENTATION_FIDELITY_AUDIT.md`. | Author custom Diffusers scheduler implementing Algorithm 1 tensor operations. |
| **BLK-08** | **Sec 4.3 / Alg 2** | Adaptive Attention Control (AAC, Algorithm 2) | Requires attaching PyTorch forward hooks into U-Net decoder cross-attention layers L4–L14 for KL-divergence blending. | Modulated global text conditioning and CFG scale ($s \in [6.5, 8.5]$). | Implement attention hook interceptor in `src/ageweaver/pipeline.py`. |
| **BLK-09** | **Suppl S1** | Attention Probing Classifier (Layers L3–L16, SVD matrices) | Depends on dumping intermediate U-Net attention tensors during forward sampling. | Documented paper's layer-wise probing results in `supplementary_checklist.md`. | Script automated feature hook tensor logger. |

---

## 2. Integrity Summary

By maintaining this exhaustive record of blocked items, AgeWeaver preserves scientific transparency. No blocked item has been obscured, simulated, or deceptively relabeled.

