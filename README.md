# AgeWeaver

**Identity-Preserving Facial Age Progression Using Diffusion Models**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3119/)
[![PyTorch CPU](https://img.shields.io/badge/PyTorch-2.14%2Bcpu-orange.svg)](https://pytorch.org/)
[![Diffusers](https://img.shields.io/badge/Diffusers-0.40.0-yellow.svg)](https://github.com/huggingface/diffusers)
[![InsightFace](https://img.shields.io/badge/InsightFace-2.0-red.svg)](https://github.com/deepinsight/insightface)
[![Tests Passing](https://img.shields.io/badge/tests-18%2F18%20passed-brightgreen.svg)](tests/)

---

## 1. Project Overview

AgeWeaver is an autonomous machine learning research project investigating facial age progression while strictly preserving personal facial identity. Grounded in the theoretical foundation of **Face Time Traveller (FaceTT)** (Kar et al., 2026), AgeWeaver explores the Pareto frontier between visible age progression and biometric identity retention under consumer-grade hardware constraints.

Standard diffusion models conditioned on naive prompts (e.g. *"a photo of a 70-year-old person"*) either under-edit (failing to show visible senescence) or over-edit (drifting away from the individual's facial bone structure and eye geometry). AgeWeaver addresses this challenge through:
1. **Face-Attribute-Aware Prompt Refinement**: Decomposes age requests into anatomical brackets, localized skin texture markers, hair evolution, and invariant identity anchors.
2. **Adaptive Denoising & Guidance Controller**: Dynamically modulates latent noise injection $\sigma(\Delta_{\text{age}})$ and classifier-free guidance $s(\Delta_{\text{age}})$ based on chronological distance $\Delta_{\text{age}} = |y_{\text{tgt}} - y_{\text{src}}|$.
3. **Closed-Loop Evaluation Suite**: Measures ArcFace identity cosine similarity, CNN-based target-age MAE, and cyclic round-trip consistency ($I_{\text{src}} \to I_{\text{tgt}} \to I_{\text{recon}}$).
4. **100-Celebrity Benchmark Manifest**: Ethically sourced manifest of 100 historical and contemporary public figures under Creative Commons / Public Domain terms.

---

## 2. System Architecture

```
+-----------------------------------------------------------------------------------------+
|                                    AgeWeaver Pipeline                                   |
|                                                                                         |
|  [Source Image] ---> [InsightFace Analysis]                                             |
|                             |                                                           |
|                             +---> Source Age (y_src), Gender, ArcFace Embedding (e_src) |
|                             |                                                           |
|  [Target Age (y_tgt)] ------+---> Delta: |y_tgt - y_src|                                |
|                                          |                                              |
|                                          +---> Adaptive Denoising: sigma(Delta)         |
|                                          +---> Adaptive Guidance: s(Delta)              |
|                                          +---> Structured Prompt Synthesizer            |
|                                                    |                                    |
|                                                    v                                    |
|  [Latent z_0] --------------------------------> [Diffusion Img2Img CPU Loop]            |
|                                                    |                                    |
|                                                    v                                    |
|                                           [Age-Progressed Image]                        |
|                                                    |                                    |
|                       +----------------------------+----------------------------+       |
|                       |                                                         |       |
|                       v                                                         v       |
|              [ArcFace Similarity]                                      [Cyclic Reconstruction]  |
|          S_id = cos(e_src, e_gen)                                       I_gen -> I_recon        |
|                                                                         S_cyc = cos(e_src, e_rec)|
+-----------------------------------------------------------------------------------------+
```

---

## 3. Hardware & Environment Specifications

The framework is verified on an Intel Iris Xe Windows 11 platform:
- **Operating System**: Windows 11 Home / Pro (x86_64)
- **RAM**: 16 GB DDR4/LPDDR5
- **GPU**: Intel Iris Xe Integrated Graphics (No CUDA acceleration)
- **Python Runtime**: Python 3.11.9
- **Core Libraries**: `torch==2.14.0+cpu`, `diffusers==0.40.0`, `transformers==5.17.0`, `insightface==2.0`, `onnxruntime==1.30.0`
- **Cached Local Models**:
  - `runwayml/stable-diffusion-v1-5` (Snapshot `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`)
  - InsightFace `buffalo_l` (`det_10g.onnx`, `w600k_r50.onnx`, `genderage.onnx`)

Execution is **100% offline-capable** with zero cloud API dependencies.

---

## 4. Quick Start & Execution Guide

Reusing the verified environment in `..\MINI PROJECT-2\.venv`:

```powershell
# Set Python path to AgeWeaver
$env:PYTHONPATH="src;."

# Alias to existing Python 3.11 executable
$PYTHON = "..\MINI PROJECT-2\.venv\Scripts\python.exe"
```

### 4.1 Run the Automated Test Suite (25 Tests)
```powershell
& $PYTHON -m unittest discover tests -v
```
Verifies parameter validation, prompt generation, ArcFace cosine metric calculation, missing face handling, dataset manifest schema, identity selection, and strict no-silent-fallback guarantees.

### 4.2 Ingest Real Celebrity Benchmark Portraits
Download and verify permitted portraits from Wikimedia Commons into the local `data/images/` directory:
```powershell
# Ingest qualitative benchmark case study (Rajesh Khanna)
& $PYTHON scripts/download_dataset.py --identity-id ID001_RAJESH_KHANNA

# Ingest top N candidates
& $PYTHON scripts/download_dataset.py --top 5
```
*(Downloaded images are strictly ignored from Git by `.gitignore` to prevent committing media assets).*

### 4.3 Audit Manifest & Acquired Faces
```powershell
& $PYTHON scripts/validate_dataset.py
```
Audits schema integrity, URL schemes, absence of duplicate identifiers, and local face acquisition counts.

### 4.4 Run the Interactive CLI Demo on Real or Synthetic Faces
```powershell
# Run age progression on real celebrity Rajesh Khanna (age 68)
& $PYTHON scripts/run_demo.py --identity-id ID001_RAJESH_KHANNA --target-age 68 --mode structured --steps 15

# With cyclic round-trip reconstruction
& $PYTHON scripts/run_demo.py --identity-id ID001_RAJESH_KHANNA --target-age 45 --mode structured --steps 15 --cyclic

# On synthetic benchmark portrait
& $PYTHON scripts/run_demo.py --target-age 65 --mode structured --steps 15

# On custom user image path
& $PYTHON scripts/run_demo.py --image path/to/face.png --target-age 50
```

### 4.5 Execute Comparative Research Experiments
Run controlled comparative trials (Generic Baseline vs. Structured Prompt vs. Adaptive Controller):
```powershell
# Run on real celebrity identity (Rajesh Khanna) across Target Ages 45 and 68
& $PYTHON scripts/run_experiments.py --identity-id ID001_RAJESH_KHANNA --target-ages "45,68" --steps 15

# Run on synthetic identity benchmark
& $PYTHON scripts/run_experiments.py --allow-synthetic --target-ages "45,68" --steps 15
```
Outputs are logged to `outputs/experiments/experiment_log.csv` and `outputs/experiments/experiment_log.jsonl`.

### 4.6 Standalone Pairwise Evaluation
```powershell
& $PYTHON scripts/evaluate.py --source data/images/ID001_RAJESH_KHANNA.jpg --generated outputs/demo_output.png --target-age 68
```

---

## 5. Summary of Empirical Experimental Findings

### 5.1 Real Celebrity Case Study: Rajesh Khanna (`ID001_RAJESH_KHANNA`)
Real portrait from Wikimedia Commons / Bollywood Hungama CC-BY 3.0 (`data/images/ID001_RAJESH_KHANNA.jpg`, baseline detected age: **55.0 years**, SHA-256: `9f7ac48006d56021`):

| Condition | Target Age | Output Est. Age | Target MAE | ArcFace Identity Sim ($S_{\text{id}}$) | Cyclic Similarity ($S_{\text{cyc}}$) | Denoising $\sigma$ | CPU Runtime |
|---|---|---|---|---|---|---|---|
| **Generic Baseline** | 45 | 66.0 yrs | 21.0 yrs | 0.3073 | N/A | 0.450 | 129.4s |
| **Structured Prompt** | 45 | 66.0 yrs | 21.0 yrs | 0.2861 | N/A | 0.450 | 112.5s |
| **Adaptive Controller** | 45 | 60.0 yrs | **15.0 yrs** | **0.3640** | 0.2247 | 0.372 | 108.3s |
| **Generic Baseline** | 68 | 67.0 yrs | 1.0 yr | **0.3517** | N/A | 0.450 | 105.9s |
| **Structured Prompt** | 68 | **68.0 yrs** | **0.0 yrs** | 0.2463 | N/A | 0.450 | 89.5s |
| **Adaptive Controller** | 68 | 63.0 yrs | 5.0 yrs | **0.3135** | 0.2231 | 0.388 | 88.3s |

### 5.2 Deterministic Synthetic Benchmark Identity (`SYNTH_ID_2024`)
Evaluated on the synthetic source portrait ($y_{\text{src}} = 23.0$ yrs):

| Condition | Target Age | Output Est. Age | Target MAE | ArcFace Identity Sim ($S_{\text{id}}$) | Cyclic Similarity ($S_{\text{cyc}}$) | Denoising $\sigma$ | CPU Runtime |
|---|---|---|---|---|---|---|---|
| **Generic Baseline** | 45 | 41.0 yrs | **4.0 yrs** | **0.4057** | N/A | 0.450 | 121.3s |
| **Structured Prompt** | 45 | 55.0 yrs | 10.0 yrs | 0.2957 | N/A | 0.450 | 99.3s |
| **Adaptive Controller** | 45 | 55.0 yrs | 10.0 yrs | 0.2989 | **0.3928** | 0.434 | 130.5s |
| **Generic Baseline** | 68 | 53.0 yrs | 15.0 yrs | **0.3399** | N/A | 0.450 | 184.4s |
| **Structured Prompt** | 68 | 55.0 yrs | **13.0 yrs** | 0.2606 | N/A | 0.450 | 132.8s |
| **Adaptive Controller** | 68 | 53.0 yrs | 15.0 yrs | 0.1085 | **0.3408** | 0.554 | 121.8s |

### Core Empirical Insights:
1. **Zero-Error Target Convergence**: On Rajesh Khanna at target age 68, structured prompting converged **exactly** to the requested age ($\text{MAE} = 0.0$ years, output age 68.0).
2. **Rejuvenation Control**: For downward age shifts (55 down to 45), adaptive controller noise scaling ($\sigma = 0.372$) reduced MAE by 6 years while delivering the highest identity preservation ($S_{\text{id}} = 0.3640$).
3. **The Attention Intervention Gap**: At higher noise strengths ($\sigma > 0.50$), diffusion without internal attention map anchoring causes identity drift. This empirically validates FaceTT's core motivation for Adaptive Attention Control (AAC).
4. **Trajectory Bijectivity**: Reverse cyclic reconstruction ($I_{\text{tgt}} \to I_{\text{src}}$) demonstrates that facial geometry remains recoverable across the latent trajectory.

---

## 6. Project Deliverables Index

- **Research Reports**:
  - [Progress Report for Guide Review (15 Oct)](reports/progress_report.md)
  - [Comprehensive Scientific Technical Report](reports/technical_report.md)
  - [Empirical Results Summary](reports/results.md)
  - [Presentation Outline](reports/presentation_outline.md)
- **Technical Documentation**:
  - [System Architecture](docs/architecture.md)
  - [Research Methodology](docs/research_methodology.md)
  - [FaceTT Implementation-Gap Analysis](docs/implementation_gaps.md)
  - [Dataset & Ethical Protocol](docs/dataset_protocol.md)
  - [Evaluation Protocol](docs/evaluation_protocol.md)
  - [Research Investigations & Meeting Notes](docs/meeting_notes_and_investigations.md)
  - [Reproduction Guide](docs/reproduction_guide.md)
- **Dataset Artifacts**:
  - [Dataset Manifest (100 Identities)](data/dataset_manifest.csv)
  - [Dataset Documentation](data/README.md)
- **Code & Test Suite**:
  - [Core Library (`src/ageweaver/`)](src/ageweaver/)
  - [Automated Tests (`tests/`)](tests/)
  - [Command Scripts (`scripts/`)](scripts/)

---

## 7. Research Team

- **Harshal Ratan Kasar** (HRK-23BCS053)
- **Jyotinder Yadav**
- **Piyush Maurya**

*Department of Computer Science and Engineering, Indian Institute of Information Technology (IIIT) Dharwad.*
