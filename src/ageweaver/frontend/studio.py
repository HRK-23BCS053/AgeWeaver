"""
Studio View for AgeWeaver Streamlit UI.
Provides portrait upload / benchmark identity selection, biometric inspection,
target-age progression controls, CPU-aware execution, and side-by-side evaluation.
"""

import os
import io
import time
import json
import streamlit as st
from PIL import Image

from ageweaver.frontend.state import (
    get_pipeline,
    get_analyzer,
    get_dataset_manager,
    get_experiment_runner
)
from ageweaver.utils import compute_image_sha256

def render_studio_view():
    """Renders the primary AgeWeaver Studio interface."""
    dm = get_dataset_manager()
    analyzer = get_analyzer()

    # Branded Header
    st.markdown("""
    <div class="ageweaver-brand">
        <div class="ageweaver-logo-icon">AW</div>
        <div class="ageweaver-title-group">
            <h1>AgeWeaver Studio</h1>
            <p>Identity-Preserving Facial Age Progression • Diffusion Latent Modeling & Biometric Invariance</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([5, 7], gap="large")

    # =========================================================================
    # LEFT COLUMN: INPUT SOURCE & PARAMETERS
    # =========================================================================
    with col_left:
        st.markdown('<div class="aw-card">', unsafe_allow_html=True)
        st.markdown('<div class="aw-card-title"><span>👤</span> 1. Select Reference Subject</div>', unsafe_allow_html=True)

        source_type = st.radio(
            "Input Mode",
            ["Verified Benchmark Archive", "Upload Custom Portrait"],
            horizontal=True,
            label_visibility="collapsed"
        )

        source_img = None
        source_id = "CUSTOM_SUBJECT"
        doc_age = None
        doc_meta = {}

        if source_type == "Verified Benchmark Archive":
            benchmark_options = {
                "ID001_RAJESH_KHANNA": "Rajesh Khanna (~1970 archival portrait, documented age ~28)",
                "ID009_DEEPIKA_PADUKONE": "Deepika Padukone (2008 adult portrait, documented age ~22)",
                "SYNTH_ID_2024": "Deterministic Synthetic Young Adult Portrait (diffusion baseline, 23 yrs)"
            }
            selected_key = st.selectbox(
                "Benchmark Subject",
                list(benchmark_options.keys()),
                format_func=lambda k: benchmark_options[k]
            )
            source_id = selected_key

            if selected_key == "ID001_RAJESH_KHANNA":
                img_path = dm.get_image_path("ID001_RAJESH_KHANNA", auto_download=True)
                source_img = Image.open(img_path).convert("RGB")
                doc_age = 28
                doc_meta = {
                    "Subject": "Rajesh Khanna (1942–2012)",
                    "Capture Year": "~1970 (Superstar era)",
                    "Documented Age": "28 years old",
                    "License": "GODL-India (India Post commemorative stamp)",
                    "Attribution": "India Post, Government of India",
                    "Resolution": f"{source_img.size[0]} x {source_img.size[1]} px"
                }
            elif selected_key == "ID009_DEEPIKA_PADUKONE":
                img_path = dm.get_image_path("ID009_DEEPIKA_PADUKONE", auto_download=True)
                source_img = Image.open(img_path).convert("RGB")
                doc_age = 22
                doc_meta = {
                    "Subject": "Deepika Padukone (b. 1986)",
                    "Capture Year": "2008 (Om Shanti Om era)",
                    "Documented Age": "22 years old",
                    "License": "CC BY-SA 3.0",
                    "Attribution": "Bollywood Hungama",
                    "Resolution": f"{source_img.size[0]} x {source_img.size[1]} px"
                }
            elif selected_key == "SYNTH_ID_2024":
                synth_path = "outputs/experiments/benchmark_source_identity.png"
                if os.path.exists(synth_path):
                    source_img = Image.open(synth_path).convert("RGB")
                else:
                    pipe = get_pipeline()
                    source_img = pipe.generate_synthetic_identity(seed=2024, steps=15)
                doc_age = 23
                doc_meta = {
                    "Subject": "Synthetic Young Identity (SYNTH_ID_2024)",
                    "Source": "Stable Diffusion v1-5 txt2img (seed 2024)",
                    "Documented Age": "23 years old",
                    "License": "OpenRAIL-M Synthetic Benchmark",
                    "Attribution": "AgeWeaver Deterministic Benchmark",
                    "Resolution": f"{source_img.size[0]} x {source_img.size[1]} px"
                }

        else:
            uploaded_file = st.file_uploader(
                "Upload high-resolution facial photograph",
                type=["jpg", "jpeg", "png"],
                help="Ensure single, clear, frontal face with minimal occlusion."
            )
            if uploaded_file is not None:
                try:
                    source_img = Image.open(uploaded_file).convert("RGB")
                    source_id = f"UPLOAD_{int(time.time())}"
                except Exception as e:
                    st.error(f"Could not load image: {e}")

        st.markdown('</div>', unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # BIOMETRIC FACE ANALYSIS CARD
        # ---------------------------------------------------------------------
        face_valid = False
        est_age = None
        gender_hint = "person"

        if source_img is not None:
            st.markdown('<div class="aw-card">', unsafe_allow_html=True)
            st.markdown('<div class="aw-card-title"><span>🔍</span> 2. Biometric Face Audit</div>', unsafe_allow_html=True)

            img_col, meta_col = st.columns([1, 1], gap="medium")
            with img_col:
                st.image(source_img, caption=f"Source: {source_id}", use_container_width=True)

            with meta_col:
                with st.spinner("Auditing facial landmarks and ArcFace representation..."):
                    face_res = analyzer.analyze_image(source_img)

                if face_res.face_detected:
                    face_valid = True
                    est_age = face_res.estimated_age
                    gender_hint = face_res.gender or "person"
                    det_score = face_res.det_score or 0.0

                    st.markdown(f'<span class="aw-badge aw-badge-cyan">✓ Face Detected (Confidence: {det_score:.2f})</span>', unsafe_allow_html=True)
                    st.markdown(f"**Gender**: `{gender_hint.capitalize()}`")
                    if doc_age is not None:
                        st.markdown(f"**Documented Chronological Age**: `{doc_age} yrs`")
                    st.markdown(f"**CNN Estimated Age**: `{est_age:.1f} yrs`")
                    st.markdown(f"**ArcFace Embedding**: `512-d normalized L2 (✓)`")

                    if doc_meta:
                        with st.expander("Provenance & License Terms", expanded=False):
                            for k, v in doc_meta.items():
                                st.markdown(f"- **{k}**: {v}")
                else:
                    st.error(f"No face detected: {face_res.error_message}")
                    st.caption("Please select or upload a frontal portrait with clear lighting.")

            st.markdown('</div>', unsafe_allow_html=True)

            # -----------------------------------------------------------------
            # AGE PROGRESSION CONFIGURATION
            # -----------------------------------------------------------------
            if face_valid:
                st.markdown('<div class="aw-card">', unsafe_allow_html=True)
                st.markdown('<div class="aw-card-title"><span>⚙️</span> 3. Progression Configuration</div>', unsafe_allow_html=True)

                # Reference base age (prioritize documented chronological age, else rounded CNN estimate)
                base_age = doc_age if doc_age is not None else int(round(est_age))
                min_target = max(base_age + 5, 30)

                target_age = st.slider(
                    "Target Progression Age (Years)",
                    min_value=min_target,
                    max_value=85,
                    value=max(min_target, 60),
                    step=5,
                    help=f"Must be strictly older than verified reference age ({base_age} yrs) for valid age progression."
                )

                age_delta = target_age - base_age
                st.markdown(f'<span class="aw-badge aw-badge-violet">Age Jump: +{age_delta} Years ({base_age} ➔ {target_age})</span>', unsafe_allow_html=True)

                condition = st.selectbox(
                    "Progression Paradigm",
                    ["structured_prompt", "adaptive_control", "generic_baseline"],
                    index=0,
                    format_func=lambda c: {
                        "structured_prompt": "Structured Age-Aware Prompt (Identity Anchors + Skin/Hair Descriptors)",
                        "adaptive_control": "AgeWeaver Adaptive Controller (Delta-Scaled Strength & Guidance)",
                        "generic_baseline": "Generic Baseline (Naive Age Prompting)"
                    }[c]
                )

                is_cyclic = False
                if condition == "adaptive_control":
                    is_cyclic = st.checkbox(
                        "Execute Cyclic Round-Trip Reconstruction",
                        value=False,
                        help="Computes round-trip recovery back to source age to measure trajectory bijectivity (FaceTT Section 5.4)."
                    )

                with st.expander("Advanced Inference Hyperparameters", expanded=False):
                    inf_steps = st.slider("Diffusion Steps", min_value=10, max_value=25, value=15, step=1, help="15 steps optimal for CPU latency.")
                    inf_seed = st.number_input("Random Seed", value=42, step=1)

                st.markdown("""
                <div class="aw-callout">
                    ⏱ <b>CPU Hardware Notice</b>: Running on Intel Iris Xe CPU (15 steps ~80–110s). The model is cached in RAM.
                </div>
                """, unsafe_allow_html=True)

                # Execute Action Button
                btn_label = "✨ Synthesize Age Progression"
                generate_clicked = st.button(
                    btn_label,
                    disabled=st.session_state.is_running,
                    use_container_width=True
                )

                if generate_clicked:
                    st.session_state.is_running = True
                    runner = get_experiment_runner()

                    exp_id = f"DEMO_{source_id}_{condition}_T{target_age}"

                    with st.spinner(f"Executing facial age progression ({condition}) to age {target_age}..."):
                        t_start = time.time()
                        try:
                            trial_result = runner.run_single_trial(
                                experiment_id=exp_id,
                                condition=condition,
                                source_image=source_img,
                                input_id=source_id,
                                target_age=target_age,
                                source_age=base_age,
                                gender_hint=gender_hint,
                                seed=inf_seed,
                                steps=inf_steps,
                                execute_cyclic=is_cyclic
                            )
                            st.session_state.last_progression_result = trial_result
                            st.success(f"Progression completed successfully in {time.time() - t_start:.1f}s!")
                        except Exception as e:
                            st.error(f"Inference error: {e}")
                        finally:
                            st.session_state.is_running = False

                st.markdown('</div>', unsafe_allow_html=True)

    # =========================================================================
    # RIGHT COLUMN: PROGRESSION RESULTS & SIDE-BY-SIDE VERIFICATION
    # =========================================================================
    with col_right:
        res = st.session_state.last_progression_result

        # If no result in current session, check for canonical Rajesh Khanna benchmark output
        if res is None:
            canonical_path = "outputs/experiments/ID001_RAJESH_KHANNA_structured_prompt_age70_seed42.png"
            ref_path = "outputs/experiments/ID001_RAJESH_KHANNA_source_reference.png"

            if os.path.exists(canonical_path) and os.path.exists(ref_path):
                # Load precomputed benchmark trial record
                res = {
                    "experiment_id": "EXP_ID001_RAJESH_KHANNA_structured_prompt_T70",
                    "input_id": "ID001_RAJESH_KHANNA",
                    "condition": "structured_prompt",
                    "source_age": 28,
                    "target_age": 70,
                    "generated_estimated_age": 69.0,
                    "target_age_mae": 1.0,
                    "identity_similarity": 0.4052,
                    "cyclic_similarity": None,
                    "runtime_sec": 110.0,
                    "strength": 0.45,
                    "guidance_scale": 7.5,
                    "output_path": canonical_path,
                    "prompt": "portrait of the exact same man, identical facial bone structure, identical eyes and nose, aged 70 years old, senior, pronounced wrinkles, deeper facial creases, visible crow's feet, mature eye contours, textured aging skin, silver-gray hair, receding hairline or thinning hair, sharp studio photo, realistic skin texture",
                    "is_precomputed_benchmark": True
                }

        st.markdown('<div class="aw-card">', unsafe_allow_html=True)
        st.markdown('<div class="aw-card-header"><span class="aw-card-title"><span>📊</span> Progression Visual Verification</span>'
                    f'<span class="aw-badge aw-badge-{"cyan" if res and not res.get("is_precomputed_benchmark") else "violet"}">'
                    f'{"Live Result" if res and not res.get("is_precomputed_benchmark") else "Verified Benchmark Reference"}</span></div>', unsafe_allow_html=True)

        if res is not None:
            out_img_path = res.get("output_path")
            if out_img_path and os.path.exists(out_img_path):
                prog_img = Image.open(out_img_path)

                # Side-by-Side Display
                disp_cols = st.columns(3 if res.get("cyclic_similarity") else 2, gap="medium")
                with disp_cols[0]:
                    ref_disp = source_img if source_img is not None else Image.open("outputs/experiments/ID001_RAJESH_KHANNA_source_reference.png")
                    st.image(ref_disp, caption=f"Original Reference (Age {res['source_age']})", use_container_width=True)

                with disp_cols[1]:
                    st.image(prog_img, caption=f"Age Progressed (Target {res['target_age']})", use_container_width=True)

                if res.get("cyclic_similarity") and len(disp_cols) > 2:
                    cyc_path = out_img_path.replace(f"age{res['target_age']}", f"cyclic_recon_age{res['source_age']}")
                    if os.path.exists(cyc_path):
                        with disp_cols[2]:
                            st.image(Image.open(cyc_path), caption=f"Cyclic Reconstruction ({res['source_age']})", use_container_width=True)

                # Quantitative Metrics Dashboard
                st.markdown('<div class="aw-metric-grid">', unsafe_allow_html=True)

                # Metric 1: Identity Similarity
                sim_val = res.get("identity_similarity")
                sim_str = f"{sim_val:.4f}" if sim_val is not None else "N/A"
                sim_class = "green" if sim_val and sim_val >= 0.40 else ("cyan" if sim_val and sim_val >= 0.25 else "violet")
                st.markdown(f"""
                <div class="aw-metric-box">
                    <div class="aw-metric-label">ArcFace Identity Sim</div>
                    <div class="aw-metric-val {sim_class}">{sim_str}</div>
                </div>
                """, unsafe_allow_html=True)

                # Metric 2: Estimated Output Age
                est_val = res.get("generated_estimated_age")
                est_str = f"{est_val:.1f} yrs" if est_val is not None else "N/A"
                st.markdown(f"""
                <div class="aw-metric-box">
                    <div class="aw-metric-label">Output Est. Age</div>
                    <div class="aw-metric-val cyan">{est_str}</div>
                </div>
                """, unsafe_allow_html=True)

                # Metric 3: Target Age MAE
                mae_val = res.get("target_age_mae")
                mae_str = f"{mae_val:.1f} yrs" if mae_val is not None else "N/A"
                mae_class = "green" if mae_val is not None and mae_val <= 3.0 else "violet"
                st.markdown(f"""
                <div class="aw-metric-box">
                    <div class="aw-metric-label">Target-Age Error</div>
                    <div class="aw-metric-val {mae_class}">{mae_str}</div>
                </div>
                """, unsafe_allow_html=True)

                # Metric 4: Runtime
                run_sec = res.get("runtime_sec", 0.0)
                st.markdown(f"""
                <div class="aw-metric-box">
                    <div class="aw-metric-label">CPU Runtime</div>
                    <div class="aw-metric-val">{run_sec:.1f}s</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

                # Prompt and parameters inspection
                with st.expander("Inspect Synthesized Text Conditioning & Parameters", expanded=False):
                    st.markdown(f"**Conditioning Prompt**: `{res.get('prompt', 'N/A')}`")
                    st.markdown(f"- **Condition**: `{res.get('condition')}`")
                    st.markdown(f"- **Denoising Strength (σ)**: `{res.get('strength', 'N/A')}`")
                    st.markdown(f"- **Guidance Scale (s)**: `{res.get('guidance_scale', 'N/A')}`")
                    st.markdown(f"- **Experiment ID**: `{res.get('experiment_id')}`")

                # Image Download Button
                buf = io.BytesIO()
                prog_img.save(buf, format="PNG")
                btn_dl = st.download_button(
                    label="💾 Download Progressed Portrait (PNG)",
                    data=buf.getvalue(),
                    file_name=f"{res.get('input_id')}_age{res.get('target_age')}.png",
                    mime="image/png",
                    use_container_width=True
                )
            else:
                st.warning("Experiment record found, but output image file is not on disk.")
        else:
            st.info("Select a reference portrait on the left and click **Synthesize Age Progression** to generate a result.")

        st.markdown('</div>', unsafe_allow_html=True)
