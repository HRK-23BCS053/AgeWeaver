# AgeWeaver Reproduction Guide

This guide provides step-by-step instructions to reproduce all experiments, tests, and demonstrations in the AgeWeaver research project.

## 1. System Requirements & Prerequisites

- **Operating System**: Windows 11 (or compatible Linux/macOS)
- **RAM**: 16 GB RAM recommended
- **Hardware**: Compatible with CPU-only execution (Intel Iris Xe or similar; no NVIDIA GPU required)
- **Python**: Python 3.11.x (verified on 3.11.9)
- **Model Cache**: Stable Diffusion v1-5 snapshot cached locally in `~/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5`
- **InsightFace Cache**: `buffalo_l` cached in `~/.insightface/models/buffalo_l`

## 2. Reusing the Existing Verified Environment

To execute using the existing verified Python environment in the sibling directory:

In PowerShell (from the `AgeWeaver` directory):
```powershell
# Set Python path to include AgeWeaver source code
$env:PYTHONPATH="src;."

# Define Python executable shortcut
$PYTHON = "..\MINI PROJECT-2\.venv\Scripts\python.exe"
```

## 3. Step-by-Step Reproduction Commands

### Step 3.1: Environment & Offline Model Check
Verify that all packages and model snapshots are present without triggering network downloads:
```powershell
& $PYTHON scripts/check_environment.py
```
Expected output:
- PyTorch 2.14.0+cpu, Diffusers 0.40.0, InsightFace 2.0
- RunwayML Stable Diffusion v1-5 snapshot verified
- InsightFace Buffalo_l ONNX models verified

### Step 3.2: Run the Automated Test Suite
Execute the 18 automated unit and regression tests:
```powershell
& $PYTHON -m unittest discover tests -v
```
Expected output:
- `Ran 18 tests in ~10s`
- `OK` (All tests pass)

### Step 3.3: Validate the 100-Celebrity Dataset Manifest
Audit the dataset manifest schema and integrity:
```powershell
& $PYTHON scripts/validate_dataset.py
```
Expected output:
- 100 unique identities parsed
- 0 schema errors found
- Summary audit counts printed

### Step 3.4: Run the Interactive CLI Demo
Run facial age progression on a synthetic benchmark or custom image:
```powershell
# Standard progression to age 65
& $PYTHON scripts/run_demo.py --target-age 65 --mode structured --steps 15

# With cyclic round-trip reconstruction
& $PYTHON scripts/run_demo.py --target-age 68 --mode structured --steps 15 --cyclic
```
Generated artifacts are saved in `outputs/demo_output.png` and `outputs/demo_cyclic_recon.png`.

### Step 3.5: Execute Comparative Research Experiments
Run the controlled multi-condition evaluation matrix comparing:
- Condition A: Generic Baseline
- Condition B: Structured Age-Aware Prompt
- Condition C: Proposed Adaptive Controller with Cyclic Evaluation
Across Target Age 45 and Target Age 68:
```powershell
& $PYTHON scripts/run_experiments.py
```
All trial records and metrics are logged to:
- `outputs/experiments/experiment_log.csv`
- `outputs/experiments/experiment_log.jsonl`
- Generated images in `outputs/experiments/`

### Step 3.6: Standalone Pairwise Evaluation
Evaluate identity similarity and age error between any two arbitrary images:
```powershell
& $PYTHON scripts/evaluate.py --source outputs/experiments/benchmark_source_identity.png --generated outputs/demo_output.png --target-age 65
```

