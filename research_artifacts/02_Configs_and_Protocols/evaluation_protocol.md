# AgeWeaver Evaluation Protocol

## 1. Evaluation Philosophy: The Identity–Aging Tension

Facial age progression must balance two antagonistic objectives:
1. **Identity Invariance**: The geometric, anatomical, and biometric signature of the subject must remain constant.
2. **Age Progression Expression**: Visual markers of biological senescence (wrinkles, skin texture laxity, nasolabial deepening, hair graying) must visibly appear.

> **Critical Evaluation Rule**: An identity cosine similarity of 0.98 is trivial to achieve by applying zero editing ($\sigma = 0.0$), but this fails completely at age progression. Conversely, heavy editing ($\sigma = 0.90$) may produce an aged face, but with an identity similarity of 0.20, indicating identity corruption.
>
> True research success lies in dominating the **Pareto frontier** between ArcFace Identity Similarity and Target-Age MAE.

## 2. Metric Formulations

### 2.1 ArcFace Identity Cosine Similarity ($S_{\text{id}}$)
- **Model**: InsightFace ResNet-50 (`w600k_r50.onnx`).
- **Feature Vector**: $\mathbf{e} \in \mathbb{R}^{512}$, normalized to $\|\mathbf{e}\|_2 = 1.0$.
- **Formula**:
  $$S_{\text{id}}(I_{\text{src}}, I_{\text{gen}}) = \mathbf{e}_{\text{src}}^\top \mathbf{e}_{\text{gen}}$$
- **Interpretation**:
  - $S_{\text{id}} \ge 0.65$: Exceptional identity preservation (same person with high biometric confidence).
  - $0.50 \le S_{\text{id}} < 0.65$: Moderate identity preservation (recognizable, subtle structural shifts).
  - $S_{\text{id}} < 0.50$: Identity drift or identity loss (perceived as a sibling or stranger).

### 2.2 Target-Age Mean Absolute Error ($E_{\text{age}}$)
- **Model**: InsightFace `genderage.onnx` regression head trained on multi-attribute face datasets.
- **Formula**:
  $$E_{\text{age}} = | \hat{y}_{\text{pred}} - y_{\text{target}} |$$
- **Interpretation**: Lower is better. Measures how faithfully the model adhered to the requested chronological age.

### 2.3 Cyclic Identity Similarity ($S_{\text{cyc}}$)
- **Concept**: Derived from FaceTT Section 5.4.
- **Process**:
  1. $I_{\text{src}} \xrightarrow{\text{age to } y_{\text{tgt}}} I_{\text{gen}}$
  2. $I_{\text{gen}} \xrightarrow{\text{reconstruct to } y_{\text{src}}} I_{\text{recon}}$
  3. $S_{\text{cyc}} = \mathbf{e}_{\text{src}}^\top \mathbf{e}_{\text{recon}}$
- **Interpretation**: Measures the bijectivity and consistency of the generative trajectory. A high $S_{\text{cyc}}$ confirms that the semantic aging transformation did not irreversibly destroy source facial geometry.

### 2.4 Face Detection Success Rate ($R_{\text{det}}$)
- Fraction of generated images containing a detectable primary face by `det_10g.onnx` with detection confidence $\ge 0.60$.
- Guard rail against generative hallucinations (e.g. unrecognizable blurs or abstract textures).

## 3. Failure Mode Handling

- **No Face Detected**: If either the source, generated, or cyclic image lacks a detectable face, the similarity score is marked as `None` / `NaN` with explicit reason `NO_FACE_DETECTED`. No arbitrary imputation is permitted.
- **Multiple Faces**: If background crowds or artifacts appear, the face analyzer automatically selects the face with the largest bounding box area.

