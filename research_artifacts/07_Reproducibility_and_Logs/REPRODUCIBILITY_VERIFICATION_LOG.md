# AgeWeaver: Reproducibility & Environment Verification Log
**Document ID:** `research_artifacts/07_Reproducibility_and_Logs/REPRODUCIBILITY_VERIFICATION_LOG.md`  
**Date of Audit:** 17 October 2026  
**Directory:** `research_artifacts/07_Reproducibility_and_Logs/`  

---

## 1. System Environment & Hardware Specifications

AgeWeaver was developed with a dual-execution architecture supporting fully offline local development on standard consumer PCs alongside accelerated cloud execution on NVIDIA GPUs:

### 1.1 Local Development Workstation
- **Operating System:** Windows 11 Home / Pro (64-bit)
- **Runtime Environment:** Python 3.11 Virtual Environment (`.venv`)
- **Execution Target:** CPU inference (PyTorch `torch.float32`)
- **Core Dependencies:**
  - `torch`: 2.1.2+cpu / 2.2.0
  - `diffusers`: 0.26.0+
  - `transformers`: 4.38.0+
  - `insightface`: 0.7.3
  - `onnxruntime`: 1.17.0
  - `opencv-python`: 4.9.0+
  - `streamlit`: 1.32.0+
  - `pillow`: 10.2.0+
- **Inference Profile:** $\sim 85\text{--}98$ seconds per $512 \times 512$ image using 15-step DPMSolver++ Karras.

### 1.2 Google Colab GPU Environment (Cloud Benchmark Runner)
- **Operating System:** Ubuntu 22.04 LTS (x86_64)
- **Accelerator:** NVIDIA Tesla T4 (16GB GDDR6 VRAM, Compute Capability 7.5)
- **CUDA Toolkit:** 12.2 / cuDNN 8.9
- **Precision:** `torch.float16` half-precision pipeline
- **Inference Profile:** $\sim 3.2\text{--}4.8$ seconds per $512 \times 512$ image.

---

## 2. Test Suite Inventory & Validation Status

The AgeWeaver test suite executes via Python's standard `unittest` framework:

```
tests/
├── test_config.py           - Validates YAML configuration parsing and parameter defaults
├── test_dataset.py          - Validates image loading, cropping, and color space handling
├── test_dataset_manifest.py - Enforces schema integrity, 100 manifest rows, and URL constraints
├── test_metrics.py          - Validates ArcFace cosine similarity, Age MAE, and edge cases
├── test_pipeline.py         - Tests adaptive parameter controller and prompt interpolation
├── test_prompts.py          - Tests age token generation and gender prompt conditioning
└── test_studio_integrity.py  - Enforces strict UI isolation between custom uploads and benchmarks
```

### Key Verification Invariants:
- **Zero Synthetic Reference Fabrication:** Unit tests verify that when reference ground-truth images are missing, $ID_{\text{sim}}^{\text{ref}}$ strictly returns `"N/A"`.
- **Upload Isolation:** Unit tests confirm that uploading a custom portrait clears cached benchmark outputs and prevents precomputed benchmark metrics from leaking into custom sessions.
- **Deterministic Seed Handling:** Unit tests verify that identical seeds produce bit-level identical prompt tokens and initial noise tensors.

---

## 3. Google Drive Synchronization Workflow

To ensure seamless code replication between the local workstation and Google Colab, a dedicated synchronization script is deployed:

- **Script Path:** [`scripts/sync_to_gdrive.ps1`](file:///d:/Academics/IIIT%20Dharwad%20Exams/SEMESTER-7/AgeWeaver/scripts/sync_to_gdrive.ps1)
- **Source of Truth:** `D:\Academics\IIIT Dharwad Exams\SEMESTER-7\AgeWeaver`
- **Destination Target:** `G:\My Drive\AgeWeaver_Research\08_Code_Backups\AgeWeaver_Colab`
- **Safety Protocol:**
  - Strictly manual execution (zero unexpected background syncs).
  - Explicit exclusions: `.git/`, `.venv/`, `__pycache__/`, temporary caches.
  - Non-destructive mirroring: Destination files are never silently deleted.
  - SHA-256 cryptographic manifest generated upon every sync: [`reports/SYNC_MANIFEST_LATEST.md`](file:///d:/Academics/IIIT%20Dharwad%20Exams/SEMESTER-7/AgeWeaver/reports/SYNC_MANIFEST_LATEST.md).

---

## 4. Google Colab GPU Execution Instructions

To execute the full multi-identity benchmark on Google Colab:

1. **Mount Google Drive in Colab:**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
2. **Navigate to Synchronized Workspace:**
   ```bash
   cd "/content/drive/My Drive/AgeWeaver_Research/08_Code_Backups/AgeWeaver_Colab"
   ```
3. **Install Colab Requirements:**
   ```bash
   pip install -r requirements-colab.txt
   ```
4. **Execute Accelerated Benchmark Runner:**
   ```bash
   python scripts/run_colab_benchmark.py --config configs/default.yaml --device cuda --fp16
   ```
5. **Output Verification:**
   All generated images, multi-age grids, and updated CSV/JSONL records are saved directly back into Google Drive under `outputs/experiments/`.

