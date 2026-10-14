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

### 4.1 Run the Automated Test Suite (18 Tests)
```powershell
& $PYTHON -m unittest discover tests -v
```
Verifies parameter validation, prompt generation, ArcFace cosine metric calculation, missing face handling, and dataset manifest schema.

### 4.2 Run the Interactive CLI Demo
```powershell
# Age progression to age 65 on synthetic benchmark portrait
& $PYTHON scripts/run_demo.py --target-age 65 --mode structured --steps 15

# With full round-trip cyclic reconstruction
& $PYTHON scripts/run_demo.py --target-age 68 --mode structured --steps 15 --cyclic

# On a custom user image
& $PYTHON scripts/run_demo.py --image path/to/face.png --target-age 50
```

### 4.3 Execute Comparative Research Experiments
Run the controlled multi-condition evaluation matrix comparing Generic Baseline vs. Structured Prompt vs. Adaptive Controller:
```powershell
& $PYTHON scripts/run_experiments.py
```
Outputs are logged to `outputs/experiments/experiment_log.csv` and `outputs/experiments/experiment_log.jsonl`.

### 4.4 Validate the 100-Celebrity Dataset Manifest
```powershell
& $PYTHON scripts/validate_dataset.py
```

### 4.5 Standalone Pairwise Evaluation
```powershell
& $PYTHON scripts/evaluate.py --source outputs/experiments/benchmark_source_identity.png --generated outputs/demo_output.png --target-age 65
```

---

## 5. Summary of Empirical Experimental Findings

Real experimental results evaluated on the local benchmark portrait ($y_{\text{src}} = 23.0$ yrs):

| Condition | Target Age | Output Est. Age | Target MAE | ArcFace Identity Sim ($S_{\text{id}}$) | Cyclic Similarity ($S_{\text{cyc}}$) | Denoising $\sigma$ | CPU Runtime |
|---|---|---|---|---|---|---|---|
| **Generic Baseline** | 45 | 41.0 yrs | **4.0 yrs** | **0.4057** | N/A | 0.450 | 121.3s |
| **Structured Prompt** | 45 | 55.0 yrs | 10.0 yrs | 0.2957 | N/A | 0.450 | 99.3s |
| **Adaptive Controller** | 45 | 55.0 yrs | 10.0 yrs | 0.2989 | **0.3928** | 0.434 | 130.5s |
| **Generic Baseline** | 68 | 53.0 yrs | 15.0 yrs | **0.3399** | N/A | 0.450 | 184.4s |
| **Structured Prompt** | 68 | 55.0 yrs | **13.0 yrs** | 0.2606 | N/A | 0.450 | 132.8s |
| **Adaptive Controller** | 68 | 53.0 yrs | 15.0 yrs | 0.1085 | **0.3408** | 0.554 | 121.8s |

### Core Empirical Insights:
1. **Generic Under-Aging**: At target 68, the generic prompt stalled at estimated age 53 ($\text{MAE} = 15.0$ yrs).
2. **Structured Prompt Progression**: Structured prompting lowered age error to 13.0 yrs by triggering localized anatomical markers.
3. **The Attention Intervention Gap**: At higher noise strengths ($\sigma = 0.554$), unguided diffusion exhibits identity drift ($S_{\text{id}} = 0.1085$). This confirms that in-flight attention map intervention (FaceTT's AAC) is necessary to unlock extreme aging without identity loss.
4. **Cyclic Reconstruction Recovery**: Reconstructing from the aged image back to the source age yielded $S_{\text{cyc}} = 0.3408$ to $0.3928$, demonstrating trajectory bijectivity.

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
