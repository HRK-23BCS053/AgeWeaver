"""
Studio View for AgeWeaver Streamlit UI.
Provides portrait upload / benchmark identity selection, biometric inspection,
target-age progression controls, CPU-aware execution, and side-by-side evaluation.

Architectural Guarantees:
1. Benchmark results are strictly gated to their corresponding benchmark identity.
2. Uploaded custom portraits NEVER receive precomputed celebrity benchmark outputs or metrics.
3. When the input portrait changes, any stale progression results and metrics are immediately cleared.
4. Documented chronological age is kept strictly distinct from machine CNN age estimates.
"""

import os
import io
import time
import hashlib
import json
import streamlit as st
from PIL import Image
from typing import Optional, Dict, Any, Tuple

from ageweaver.frontend.state import (
    get_pipeline,
    get_analyzer,
    get_dataset_manager,
    get_experiment_runner
)
from ageweaver.utils import compute_image_sha256

def resolve_source_age_metadata(
    source_type: str,
    benchmark_id: Optional[str],
    user_provided_age: Optional[int],
    cnn_estimated_age: Optional[float]
) -> Dict[str, Any]:
    """
    Deterministically resolves documented ground-truth age vs CNN machine estimate.
    
    Guarantees:
    - Documented chronological age is only set if historically verified (benchmark archive)
      or explicitly entered by the user.
    - An unvalidated CNN biometric estimate is never labeled as documented chronological age.
    """
    doc_age = None
    is_documented = False
    
    if source_type == "Verified Benchmark Archive":
        if benchmark_id == "ID001_RAJESH_KHANNA":
            doc_age = 28
            is_documented = True
        elif benchmark_id == "ID009_DEEPIKA_PADUKONE":
            doc_age = 22
            is_documented = True
        elif benchmark_id == "SYNTH_ID_2024":
            doc_age = 23
            is_documented = True
    else:
        # Custom upload
        if user_provided_age is not None and user_provided_age > 0:
            doc_age = user_provided_age
            is_documented = True

    # Determine baseline age for progression slider
    if doc_age is not None:
        baseline_age = doc_age
        baseline_source_desc = f"documented chronological age ({doc_age} yrs)"
    elif cnn_estimated_age is not None:
        baseline_age = int(round(cnn_estimated_age))
        baseline_source_desc = f"CNN estimated biometric age ({cnn_estimated_age:.1f} yrs)"
    else:
        baseline_age = 25
        baseline_source_desc = "default baseline age (25 yrs)"

    return {
        "documented_age": doc_age,
        "is_documented": is_documented,
        "cnn_estimated_age": cnn_estimated_age,
        "baseline_age": baseline_age,
        "baseline_source_desc": baseline_source_desc
    }

def resolve_studio_result(
    source_type: str,
    selected_benchmark: Optional[str],
    current_source_id: Optional[str],
    current_sha256: Optional[str],
    session_result: Optional[Dict[str, Any]]
) -> Tuple[Optional[Dict[str, Any]], bool]:
    """
    Resolves the progression result to display in Studio right column.
    
    Returns:
        (result_dict, is_precomputed_benchmark)
        
    Guarantees:
    - If session_result is present but was computed for a DIFFERENT input_id or
      input_sha256, it is rejected (returns None).
    - Precomputed benchmark results are ONLY returned when source_type == 'Verified Benchmark Archive'
      AND selected_benchmark == 'ID001_RAJESH_KHANNA' (and files exist on disk).
    - For custom uploads (source_type == 'Upload Custom Portrait'), precomputed celebrity
      benchmark results are STRICTLY FORBIDDEN and will NEVER be returned.
    """
    # 1. If an active session result exists, check whether it matches the current input
    if session_result is not None:
        res_input_id = session_result.get("input_id")
        res_sha = session_result.get("input_sha256")
        
        # Check matching identity or image hash
        id_match = (res_input_id == current_source_id) if (res_input_id and current_source_id) else False
        sha_match = (res_sha == current_sha256) if (res_sha and current_sha256) else False
        
        if id_match or sha_match:
            # Genuine matching trial generated in active session
            return session_result, False
        else:
            # Stale result from an older image or different subject! Discard.
            session_result = None

    # 2. If no matching session result, check if we should show canonical benchmark preview
    # STRICT GUARD: ONLY for Verified Benchmark Archive and specifically ID001_RAJESH_KHANNA
    if source_type == "Verified Benchmark Archive" and selected_benchmark == "ID001_RAJESH_KHANNA":
        canonical_path = "outputs/experiments/ID001_RAJESH_KHANNA_structured_prompt_age70_seed42.png"
        ref_path = "outputs/experiments/ID001_RAJESH_KHANNA_source_reference.png"

        if os.path.exists(canonical_path) and os.path.exists(ref_path):
            precomputed = {
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
                "is_precomputed_benchmark": True,
                "benchmark_ref_path": ref_path
            }
            return precomputed, True

    # For custom uploads or benchmark identities without precomputed preview, return None (awaiting generation)
    return None, False

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
        source_id = None
        current_sha256 = None
        user_provided_age = None
        doc_meta = {}
        selected_benchmark_key = None

        if source_type == "Verified Benchmark Archive":
            benchmark_options = {
                "ID001_RAJESH_KHANNA": "Rajesh Khanna (~1970 archival portrait, documented age ~28)",
                "ID009_DEEPIKA_PADUKONE": "Deepika Padukone (2008 adult portrait, documented age ~22)",
                "SYNTH_ID_2024": "Deterministic Synthetic Young Adult Portrait (diffusion baseline, 23 yrs)"
            }
            selected_benchmark_key = st.selectbox(
                "Benchmark Subject",
                list(benchmark_options.keys()),
                format_func=lambda k: benchmark_options[k]
            )
            source_id = selected_benchmark_key

            if selected_benchmark_key == "ID001_RAJESH_KHANNA":
                img_path = dm.get_image_path("ID001_RAJESH_KHANNA", auto_download=True)
                source_img = Image.open(img_path).convert("RGB")
                current_sha256 = compute_image_sha256(source_img)
                doc_meta = {
                    "Subject": "Rajesh Khanna (1942–2012)",
                    "Capture Year": "~1970 (Superstar era)",
                    "Documented Age": "28 years old (Verified Ground Truth)",
                    "License": "GODL-India (India Post commemorative stamp)",
                    "Attribution": "India Post, Government of India",
                    "Resolution": f"{source_img.size[0]} x {source_img.size[1]} px"
                }
            elif selected_benchmark_key == "ID009_DEEPIKA_PADUKONE":
                img_path = dm.get_image_path("ID009_DEEPIKA_PADUKONE", auto_download=True)
                source_img = Image.open(img_path).convert("RGB")
                current_sha256 = compute_image_sha256(source_img)
                doc_meta = {
                    "Subject": "Deepika Padukone (b. 1986)",
                    "Capture Year": "2008 (Om Shanti Om era)",
                    "Documented Age": "22 years old (Verified Ground Truth)",
                    "License": "CC BY-SA 3.0",
                    "Attribution": "Bollywood Hungama",
                    "Resolution": f"{source_img.size[0]} x {source_img.size[1]} px"
                }
            elif selected_benchmark_key == "SYNTH_ID_2024":
                synth_path = "outputs/experiments/benchmark_source_identity.png"
                if os.path.exists(synth_path):
                    source_img = Image.open(synth_path).convert("RGB")
                else:
                    pipe = get_pipeline()
                    source_img = pipe.generate_synthetic_identity(seed=2024, steps=15)
                current_sha256 = compute_image_sha256(source_img)
                doc_meta = {
                    "Subject": "Synthetic Young Identity (SYNTH_ID_2024)",
                    "Source": "Stable Diffusion v1-5 txt2img (seed 2024)",
                    "Documented Age": "23 years old (Deterministic Baseline)",
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
                    current_sha256 = compute_image_sha256(source_img)
                    file_stem = os.path.splitext(uploaded_file.name)[0]
                    safe_stem = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in file_stem)
                    source_id = f"UPLOAD_{safe_stem}_{current_sha256[:8]}"
                    doc_meta = {
                        "File Name": uploaded_file.name,
                        "Format": uploaded_file.type,
                        "Integrity Hash": current_sha256,
                        "Resolution": f"{source_img.size[0]} x {source_img.size[1]} px"
                    }

                    user_provided_age = st.number_input(
                        "Subject's Actual Age When Photo Was Taken (Optional)",
                        min_value=0,
                        max_value=100,
                        value=0,
                        step=1,
                        help="Leave 0 if unknown; the CNN biometric estimate will then be used as the progression baseline."
                    )
                except Exception as e:
                    st.error(f"Could not load image: {e}")

        st.markdown('</div>', unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # INPUT STATE INVALIDATION CHECK
        # ---------------------------------------------------------------------
        # If the user changed the active source (different image or different benchmark),
        # clear any stale progression results immediately.
        prev_id = st.session_state.get("active_source_id")
        prev_sha = st.session_state.get("active_source_sha256")
        if source_id != prev_id or current_sha256 != prev_sha:
            st.session_state.active_source_id = source_id
            st.session_state.active_source_sha256 = current_sha256
            st.session_state.last_progression_result = None

        # ---------------------------------------------------------------------
        # BIOMETRIC FACE ANALYSIS CARD
        # ---------------------------------------------------------------------
        face_valid = False
        est_age = None
        gender_hint = "person"
        age_meta = None

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

                    # Resolve documented vs CNN machine age
                    age_meta = resolve_source_age_metadata(
                        source_type=source_type,
                        benchmark_id=selected_benchmark_key,
                        user_provided_age=user_provided_age,
                        cnn_estimated_age=est_age
                    )

                    st.markdown(f'<span class="aw-badge aw-badge-cyan">✓ Face Detected (Confidence: {det_score:.2f})</span>', unsafe_allow_html=True)
                    st.markdown(f"**Gender**: `{gender_hint.capitalize()}`")
                    
                    if age_meta["is_documented"]:
                        st.markdown(f"**Documented Chronological Age**: `{age_meta['documented_age']} yrs` (Verified)")
                    else:
                        st.markdown("**Documented Chronological Age**: `Not specified (Custom Upload)`")

                    st.markdown(f"**CNN Biometric Estimated Age**: `{est_age:.1f} yrs` *(Machine Estimate)*")
                    st.markdown(f"**ArcFace Representation**: `512-d normalized L2 (✓)`")

                    if doc_meta:
                        with st.expander("Provenance & Audit Metadata", expanded=False):
                            for k, v in doc_meta.items():
                                st.markdown(f"- **{k}**: {v}")
                else:
                    st.error(f"No face detected: {face_res.error_message}")
                    st.caption("Please select or upload a frontal portrait with clear lighting.")

            st.markdown('</div>', unsafe_allow_html=True)

            # -----------------------------------------------------------------
            # AGE PROGRESSION CONFIGURATION
            # -----------------------------------------------------------------
            if face_valid and age_meta:
                st.markdown('<div class="aw-card">', unsafe_allow_html=True)
                st.markdown('<div class="aw-card-title"><span>⚙️</span> 3. Progression Configuration</div>', unsafe_allow_html=True)

                base_age = age_meta["baseline_age"]
                min_target = max(base_age + 5, 25)

                target_age = st.slider(
                    "Target Progression Age (Years)",
                    min_value=min_target,
                    max_value=90,
                    value=max(min_target, min(base_age + 30, 75)),
                    step=5,
                    help=f"Must be strictly older than reference baseline ({base_age} yrs from {age_meta['baseline_source_desc']})."
                )

                age_delta = target_age - base_age
                st.markdown(
                    f'<span class="aw-badge aw-badge-violet">Age Jump: +{age_delta} Years ({base_age} ➔ {target_age} yrs • Based on {age_meta["baseline_source_desc"]})</span>',
                    unsafe_allow_html=True
                )

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

                    exp_id = f"RUN_{source_id}_{condition}_T{target_age}"

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
        # Resolve progression result: strictly gated to current input
        res, is_benchmark = resolve_studio_result(
            source_type=source_type,
            selected_benchmark=selected_benchmark_key,
            current_source_id=source_id,
            current_sha256=current_sha256,
            session_result=st.session_state.last_progression_result
        )

        if res is not None:
            out_img_path = res.get("output_path")
            has_out_file = bool(out_img_path and os.path.exists(out_img_path))

            st.markdown('<div class="aw-card">', unsafe_allow_html=True)
            
            badge_label = "Verified Benchmark Reference" if is_benchmark else "Live Generation Result"
            badge_class = "violet" if is_benchmark else "cyan"

            st.markdown(
                f'<div class="aw-card-header"><span class="aw-card-title"><span>📊</span> Progression Visual Verification</span>'
                f'<span class="aw-badge aw-badge-{badge_class}">{badge_label}</span></div>',
                unsafe_allow_html=True
            )

            if is_benchmark:
                st.markdown("""
                <div class="aw-callout">
                    ℹ️ <b>Archival Benchmark Preview</b>: Displaying precomputed verified benchmark trial <code>EXP_ID001_RAJESH_KHANNA_structured_prompt_T70</code>.
                    Configure parameters on the left and click <b>✨ Synthesize Age Progression</b> to execute a live trial.
                </div>
                """, unsafe_allow_html=True)

            if has_out_file:
                prog_img = Image.open(out_img_path)

                # Determine reference image to display
                if is_benchmark:
                    ref_disp = Image.open(res["benchmark_ref_path"])
                    ref_caption = f"Benchmark Reference (Rajesh Khanna, Doc. Age {res['source_age']})"
                else:
                    ref_disp = source_img
                    src_age_label = f"Doc. Age {res['source_age']}" if (age_meta and age_meta["is_documented"]) else f"Est. Age {res['source_age']}"
                    ref_caption = f"Uploaded Reference ({src_age_label})"

                # Side-by-Side Display
                disp_cols = st.columns(3 if res.get("cyclic_similarity") else 2, gap="medium")
                with disp_cols[0]:
                    st.image(ref_disp, caption=ref_caption, use_container_width=True)

                with disp_cols[1]:
                    st.image(prog_img, caption=f"Age Progressed Output (Target {res['target_age']})", use_container_width=True)

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
                    if not is_benchmark:
                        st.markdown(f"- **Input Image Hash**: `{current_sha256}`")

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
                st.warning(f"Experiment record found ({res.get('experiment_id')}), but output image file was not found on disk.")
            
            st.markdown('</div>', unsafe_allow_html=True)

        else:
            # Standby / Awaiting generation state
            st.markdown('<div class="aw-card">', unsafe_allow_html=True)
            
            if source_img is not None:
                st.markdown("""
                <div class="aw-card-header">
                    <span class="aw-card-title"><span>⏳</span> Awaiting Age Progression Synthesis</span>
                    <span class="aw-badge aw-badge-cyan">Portrait Loaded</span>
                </div>
                <div style="text-align: center; padding: 40px 16px; color: #94A3B8;">
                    <div style="font-size: 42px; margin-bottom: 12px;">🧬</div>
                    <div style="font-size: 16px; font-weight: 700; color: #F8FAFC; margin-bottom: 8px;">
                        Portrait Biometrically Audited & Ready
                    </div>
                    <p style="font-size: 13px; max-width: 440px; margin: 0 auto 20px auto; line-height: 1.6; color: #CBD5E1;">
                        Configure your desired target age and progression paradigm on the left, then click
                        <strong style="color: #A78BFA;">✨ Synthesize Age Progression</strong>.
                        The diffusion pipeline will execute on this specific portrait and compute genuine ArcFace and CNN metrics.
                    </p>
                    <div style="display: inline-flex; gap: 8px; flex-wrap: wrap; justify-content: center;">
                        <span class="aw-badge aw-badge-gray">No Celebrity Fallback</span>
                        <span class="aw-badge aw-badge-violet">CPU Latency ~80–110s</span>
                        <span class="aw-badge aw-badge-cyan">InsightFace ArcFace Ready</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="aw-card-header">
                    <span class="aw-card-title"><span>📤</span> Upload Portrait to Begin</span>
                    <span class="aw-badge aw-badge-gray">Awaiting Input</span>
                </div>
                <div style="text-align: center; padding: 48px 16px; color: #64748B;">
                    <div style="font-size: 38px; margin-bottom: 12px;">👤</div>
                    <p style="font-size: 14px; margin: 0; color: #94A3B8;">Upload a frontal photograph on the left to start biometric face audit and age progression.</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)
