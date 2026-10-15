"""
About & Methodology View for AgeWeaver Streamlit UI.
Details research motivations, latent diffusion methodology, honest FaceTT gap analysis,
dataset curation with provenance, hardware optimization choices, and institutional context.
"""

import streamlit as st

def render_about_view():
    """Renders the About & Methodology destination."""
    st.markdown("""
    <div class="ageweaver-brand">
        <div class="ageweaver-logo-icon">AW</div>
        <div class="ageweaver-title-group">
            <h1>About AgeWeaver</h1>
            <p>Identity-Preserving Facial Age Progression • Architectural Blueprint & Methodology</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # =========================================================================
    # 1. RESEARCH OBJECTIVE & PROBLEM STATEMENT
    # =========================================================================
    st.markdown('<div class="aw-card">', unsafe_allow_html=True)
    st.markdown('<div class="aw-card-title"><span>🎯</span> Research Objective & Motivation</div>', unsafe_allow_html=True)
    st.markdown("""
    Facial age progression (also known as age synthesis or digital aging) aims to synthetically predict the future facial appearance of an individual while strictly preserving their unique facial geometry, bone structure, and identity. 

    Traditional generative approaches (such as conditional GANs) suffer from mode collapse, unnatural skin textures, and severe identity degradation across large age jumps. Recent diffusion-based approaches offer unprecedented textural realism, but naively prompting a latent diffusion model erodes facial identity, replacing the original subject with a generic individual fitting the text prompt.

    **AgeWeaver** investigates controllable, identity-preserving facial age progression using latent diffusion models (Stable Diffusion v1-5) and deep face recognition embeddings (ArcFace / InsightFace buffalo_l) in a strictly offline, training-free setting.
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # =========================================================================
    # 2. TECHNICAL METHODOLOGY
    # =========================================================================
    st.markdown('<div class="aw-card">', unsafe_allow_html=True)
    st.markdown('<div class="aw-card-title"><span>🧬</span> Technical Methodology & Experimental Paradigms</div>', unsafe_allow_html=True)

    st.markdown("""
    AgeWeaver formulates age progression as controlled latent image-to-image translation. Given a source portrait $x_{\\text{src}}$ at verified young chronological age $y_{\\text{src}}$:
    
    1. **Biometric Face Audit**: The image is processed through InsightFace buffalo_l (RetinaFace `det_10g.onnx`) to confirm single face presence, localize 5 facial landmarks, extract a 512-dimensional $L_2$-normalized ArcFace embedding $f(x_{\\text{src}})$, and estimate initial biometric age $\\hat{y}_{\\text{src}}$ via ResNet `genderage.onnx`.
    2. **Latent Encoding**: $x_{\\text{src}}$ is encoded into latent space $z_0 = \\mathcal{E}(x_{\\text{src}})$ using the VAE encoder.
    3. **Perturbation**: Gaussian noise is added to reach timestep $t = \\text{round}(\\sigma \\cdot T)$:
       $$z_t = \\sqrt{\\bar{\\alpha}_t} z_0 + \\sqrt{1 - \\bar{\\alpha}_t} \\epsilon, \\quad \\epsilon \\sim \\mathcal{N}(0, I)$$
    4. **Denoising with Conditioning**: Latent $z_t$ is iteratively denoised via DDIM across 15 steps under textual conditioning prompt $c(y_{\\text{tgt}})$ and classifier-free guidance $s$.
    
    We rigorously compare three experimental conditions across identical seeds and identities:
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        st.markdown("""
        <div class="aw-portrait-card" style="padding: 16px;">
            <div style="font-weight: 700; color: #CBD5E1; margin-bottom: 8px;">Condition A: Generic Baseline</div>
            <p style="font-size: 12px; color: #94A3B8; line-height: 1.5;">
                Naive age prompt (e.g. <i>"a portrait photo of a 70-year-old man"</i>). Fixed denoising strength (σ=0.45) and guidance (s=7.5). Serves as the experimental control.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="aw-portrait-card" style="padding: 16px; border-color: rgba(139, 92, 246, 0.4);">
            <div style="font-weight: 700; color: #C4B5FD; margin-bottom: 8px;">Condition B: Structured Prompt</div>
            <p style="font-size: 12px; color: #94A3B8; line-height: 1.5;">
                Decomposes conditioning into identity preservation anchors (<i>"identical facial bone structure, identical eyes and nose"</i>) paired with age bracket skin/hair morphological modifiers.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="aw-portrait-card" style="padding: 16px; border-color: rgba(6, 182, 212, 0.4);">
            <div style="font-weight: 700; color: #67E8F9; margin-bottom: 8px;">Condition C: Adaptive Controller</div>
            <p style="font-size: 12px; color: #94A3B8; line-height: 1.5;">
                Dynamically modulates denoising strength σ(Δy) and guidance s(Δy) proportional to the age jump Δy = y_tgt - y_src, combined with round-trip cyclic consistency verification.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # =========================================================================
    # 3. FACETT IMPLEMENTATION GAP DISCLOSURE (CRITICAL)
    # =========================================================================
    st.markdown('<div class="aw-card">', unsafe_allow_html=True)
    st.markdown('<div class="aw-card-title"><span>⚖️</span> FaceTT Implementation Gap & Research Honesty</div>', unsafe_allow_html=True)

    st.markdown("""
    In the computer vision literature, **FaceTT** (*He et al., 2024, "FaceTT: Towards Identity-Preserving Facial Age Progression via Texture Transfer"*) is a state-of-the-art framework requiring substantial GPU cluster resources for fine-tuning diffusion cross-attention layers.

    **Academic Integrity Disclosure**: AgeWeaver does **not** claim to be a 100% reproduction of FaceTT. The exact relationship and architectural gaps are outlined below:
    """, unsafe_allow_html=True)

    st.markdown("""
    | FaceTT Theoretical Component | FaceTT Paper Specification | AgeWeaver Operational Implementation | Technical Justification |
    | :--- | :--- | :--- | :--- |
    | **Adaptive Aging Controller (AAC)** | Trainable cross-attention module balancing identity and age attention maps dynamically | Heuristic scheduling: $\\sigma(\\Delta y) = 0.35 + 0.20 \\cdot \\frac{\\Delta y}{50}$ and $s(\\Delta y) = 6.5 + 2.0 \\cdot \\frac{\\Delta y}{50}$ | Running on 16GB RAM CPU without GPU precludes training UNet cross-attention modules. |
    | **Angular Inversion** | Spherical manifold latent inversion maintaining constant angular distance to source face | DDIM latent inversion / deterministic img2img latent perturbation | Inversion with spherical constraints requires iterative gradient steps per image. |
    | **Identity Loss Guidance** | Computes $\\nabla_{z_t} \\mathcal{L}_{\\text{id}}$ via ArcFace backpropagation through UNet during sampling | Training-free classifier-free guidance with explicit textual identity anchors | Backpropagating through a 1-billion-parameter UNet on CPU would take >15 minutes per image. |
    | **Cyclic Consistency (Sec 5.4)** | Inverts progressed image back to young age: $x_{\\text{prog}} \\to \\hat{x}_{\\text{src}}$ | Implemented in Condition C: computes round-trip ArcFace cosine similarity | Validates bijectivity of the progression trajectory without retraining. |
    """)

    st.markdown('</div>', unsafe_allow_html=True)

    # =========================================================================
    # 4. DATASET DESIGN & LICENSING
    # =========================================================================
    st.markdown('<div class="aw-card">', unsafe_allow_html=True)
    st.markdown('<div class="aw-card-title"><span>📂</span> Dataset Architecture & Provenance Integrity</div>', unsafe_allow_html=True)

    st.markdown("""
    - **Manifest Schema (`data/dataset_manifest.csv`)**: Contains 100 curated identity entries with documented birth years, public domain / CC-BY / GODL provenance URLs, and license metadata.
    - **Physical Image Storage (`data/images/`)**: Downloaded reference images are stored strictly in `data/images/`, which is added to `.gitignore`. **No copyrighted images or large model weights are ever committed to Git.**
    - **Archival Young Reference Auditing**: As mandated by faculty evaluation, reference portraits must depict the subject in their young adult years ($y_{\\text{src}} \\approx 20-35$ yrs) so that subsequent progressions represent genuine forward aging ($y_{\\text{tgt}} > y_{\\text{src}}$).
    - **Canonical Benchmark Example (Rajesh Khanna)**:
      - *Subject*: Rajesh Khanna (1942–2012), iconic superstar of Indian cinema.
      - *Provenance*: India Post commemorative philatelic stamp (~1970 archival photograph, superstar era).
      - *Documented Chronological Age*: **28 years old** (verified from 1942 birth year and ~1970 capture date).
      - *License*: Government Open Data License - India (GODL-India).
      - *Integrity Hash*: `bc2beae7d13cf7340b0805f1f9a1cb1703c14a2fc08f9a941574a7eb098e4d20`.
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # =========================================================================
    # 5. HARDWARE CONSTRAINTS & EXECUTION ENGINEERING
    # =========================================================================
    st.markdown('<div class="aw-card">', unsafe_allow_html=True)
    st.markdown('<div class="aw-card-title"><span>💻</span> Hardware Constraints & CPU Optimization</div>', unsafe_allow_html=True)

    st.markdown("""
    - **Execution Environment**: Windows 11 Home, 16.0 GB RAM, Intel Core i5 with Intel Iris Xe integrated graphics (no NVIDIA GPU / CUDA).
    - **Optimized Latency Strategies**:
      - **15-Step DDIM Scheduler**: Reduces diffusion inference steps from 50 to 15, cutting CPU inference time from ~5 minutes to ~80–110s with negligible perceptual loss.
      - **RAM Singleton Caching (`@st.cache_resource`)**: Stable Diffusion v1-5 and InsightFace buffalo_l ONNX sessions are loaded once into system RAM and reused across all trials.
      - **Thread Pool Optimization**: Configured `torch.set_num_threads` to leverage multi-core CPU efficiency without thrashing system memory.
      - **100% Offline Compatibility**: Zero reliance on external paid APIs (OpenAI, Replicate, etc.). The entire research pipeline runs self-contained on the local machine.
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # =========================================================================
    # 6. INSTITUTIONAL METADATA
    # =========================================================================
    st.markdown('<div class="aw-card">', unsafe_allow_html=True)
    st.markdown('<div class="aw-card-title"><span>🏛️</span> Academic & Project Information</div>', unsafe_allow_html=True)
    st.markdown("""
    - **Project**: AgeWeaver — Identity-Preserving Facial Age Progression via Latent Diffusion
    - **Institution**: Indian Institute of Information Technology (IIIT) Dharwad
    - **Curriculum**: Semester-7 Capstone / Mini-Project 2 Research Continuation
    - **Evaluation Deadline**: 15 October 2026
    - **Repository**: Private Git Workspace (`master` branch)
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
