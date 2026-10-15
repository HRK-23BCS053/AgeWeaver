"""
Research Metrics & Empirical Benchmarking View for AgeWeaver Streamlit UI.
Provides scientific comparative analysis across progression paradigms:
- Generic Baseline (A)
- Structured Age-Aware Prompt (B)
- AgeWeaver Adaptive Controller (C)
Distinguishes CNN-estimated ages from documented chronological ages,
visualizes the Pareto identity-progression tradeoff, and honestly details
FaceTT implementation gaps.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from typing import List, Dict, Any, Tuple

from ageweaver.frontend.experiments_view import load_experiment_records

# Visual palette constants
DARK_BG = "#0D1326"
AXIS_BG = "#131B35"
TEXT_COLOR = "#F8FAFC"
MUTED_TEXT = "#94A3B8"
SPINE_COLOR = "#334155"
GRID_COLOR = "#1E293B"

COLOR_BASELINE = "#94A3B8"     # Gray / Silver
COLOR_STRUCTURED = "#8B5CF6"   # Electric Violet
COLOR_ADAPTIVE = "#06B6D4"     # Cyan
COLOR_IDEAL = "#34D399"        # Mint Green

def apply_plot_theme(fig, ax):
    """Applies high-contrast dark midnight-navy theme to matplotlib axes."""
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(AXIS_BG)
    ax.tick_params(colors=MUTED_TEXT, labelsize=10)
    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)
        spine.set_linewidth(1.0)
    ax.grid(True, linestyle="--", alpha=0.3, color=GRID_COLOR)

def render_metrics_view():
    """Renders the Research Metrics destination."""
    st.markdown("""
    <div class="ageweaver-brand">
        <div class="ageweaver-logo-icon">RM</div>
        <div class="ageweaver-title-group">
            <h1>Research Metrics & Comparative Evaluation</h1>
            <p>Empirical Quantitative Benchmarks • Identity Preservation vs. Aging Fidelity Pareto Frontier</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    records = load_experiment_records()
    if not records:
        st.warning("No experiment records found in outputs/experiments/experiment_log.jsonl.")
        return

    df_all = pd.DataFrame(records)

    # Filter out any unmeasured trials
    df_valid = df_all.dropna(subset=["identity_similarity", "target_age_mae"]).copy()

    # =========================================================================
    # COHORT SELECTION
    # =========================================================================
    st.markdown('<div class="aw-card">', unsafe_allow_html=True)
    st.markdown('<div class="aw-card-title"><span>📊</span> Experimental Cohort Selection</div>', unsafe_allow_html=True)

    cohort_options = {
        "verified_young": "Verified Young Archival Benchmark (Rajesh Khanna, documented age ~28, source_age=28)",
        "synthetic": "Deterministic Synthetic Identity Benchmark (SYNTH_ID_2024, source_age=23)",
        "all_trials": "All Completed Experimental Trials Combined (21 trials)"
    }

    sel_cohort = st.selectbox(
        "Select Evaluation Benchmark Cohort",
        list(cohort_options.keys()),
        format_func=lambda k: cohort_options[k]
    )

    if sel_cohort == "verified_young":
        df = df_valid[(df_valid["input_id"] == "ID001_RAJESH_KHANNA") & (df_valid["source_age"] == 28)].copy()
        cohort_desc = "Verified archival young portrait of Rajesh Khanna (~1970, India Post archival release, documented age ~28)."
    elif sel_cohort == "synthetic":
        df = df_valid[df_valid["input_id"] == "SYNTH_ID_2024"].copy()
        cohort_desc = "Deterministic synthetic young adult portrait generated via Stable Diffusion v1-5 (seed 2024, source age ~23)."
    else:
        df = df_valid.copy()
        cohort_desc = "All 21 recorded physical trials across all identities and conditions."

    st.markdown(f'<p style="color: #94A3B8; font-size: 13px; margin-top: 4px;">{cohort_desc}</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if df.empty:
        st.warning("No records available for the selected cohort.")
        return

    # =========================================================================
    # EXECUTIVE SUMMARY CARDS
    # =========================================================================
    cond_stats = df.groupby("condition").agg(
        mean_sim=("identity_similarity", "mean"),
        std_sim=("identity_similarity", "std"),
        mean_mae=("target_age_mae", "mean"),
        std_mae=("target_age_mae", "std"),
        mean_runtime=("runtime_sec", "mean"),
        count=("experiment_id", "count")
    ).reset_index()

    st.markdown('<div class="aw-metric-grid">', unsafe_allow_html=True)

    # Card 1: Top Identity Preservation
    top_sim_row = cond_stats.sort_values(by="mean_sim", ascending=False).iloc[0]
    st.markdown(f"""
    <div class="aw-metric-box">
        <div class="aw-metric-label">Highest Identity Preserved</div>
        <div class="aw-metric-val violet">{top_sim_row['mean_sim']:.4f}</div>
        <div style="font-size: 11px; color: #94A3B8; margin-top: 4px;">{top_sim_row['condition']}</div>
    </div>
    """, unsafe_allow_html=True)

    # Card 2: Lowest Target Age MAE
    best_mae_row = cond_stats.sort_values(by="mean_mae", ascending=True).iloc[0]
    st.markdown(f"""
    <div class="aw-metric-box">
        <div class="aw-metric-label">Lowest Target Age MAE</div>
        <div class="aw-metric-val green">{best_mae_row['mean_mae']:.1f} yrs</div>
        <div style="font-size: 11px; color: #94A3B8; margin-top: 4px;">{best_mae_row['condition']}</div>
    </div>
    """, unsafe_allow_html=True)

    # Card 3: Adaptive Control Balance
    adap_rows = cond_stats[cond_stats["condition"] == "adaptive_control"]
    if not adap_rows.empty:
        adap_row = adap_rows.iloc[0]
        st.markdown(f"""
        <div class="aw-metric-box">
            <div class="aw-metric-label">Adaptive Control Sim</div>
            <div class="aw-metric-val cyan">{adap_row['mean_sim']:.4f}</div>
            <div style="font-size: 11px; color: #94A3B8; margin-top: 4px;">MAE: {adap_row['mean_mae']:.1f} yrs</div>
        </div>
        """, unsafe_allow_html=True)

    # Card 4: Mean CPU Runtime
    mean_cpu_time = df["runtime_sec"].mean()
    st.markdown(f"""
    <div class="aw-metric-box">
        <div class="aw-metric-label">Mean CPU Inversion</div>
        <div class="aw-metric-val">{mean_cpu_time:.1f}s</div>
        <div style="font-size: 11px; color: #94A3B8; margin-top: 4px;">15-step DDIM on Intel Iris Xe</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # =========================================================================
    # SCIENTIFIC COMPARISON CHARTS
    # =========================================================================
    tab1, tab2, tab3 = st.tabs([
        "📈 Primary Paradigm Comparison",
        "🎯 Pareto Tradeoff & Calibration",
        "📋 Condition Aggregate Table"
    ])

    with tab1:
        chart_col1, chart_col2 = st.columns(2, gap="medium")

        # Plot 1: ArcFace Identity Cosine Similarity
        with chart_col1:
            st.markdown('<div class="aw-card">', unsafe_allow_html=True)
            st.markdown('<div class="aw-card-title"><span>🛡️</span> ArcFace Identity Preservation (Higher is Better)</div>', unsafe_allow_html=True)

            fig1, ax1 = plt.subplots(figsize=(6, 4.2), dpi=120)
            apply_plot_theme(fig1, ax1)

            conditions = ["generic_baseline", "structured_prompt", "adaptive_control"]
            labels = ["Generic Baseline\n(Condition A)", "Structured Prompt\n(Condition B)", "Adaptive Control\n(Condition C)"]
            colors = [COLOR_BASELINE, COLOR_STRUCTURED, COLOR_ADAPTIVE]

            means = []
            stds = []
            for c in conditions:
                subset = df[df["condition"] == c]["identity_similarity"]
                means.append(subset.mean() if not subset.empty else 0.0)
                stds.append(subset.std() if len(subset) > 1 else 0.0)

            x = np.arange(len(conditions))
            bars = ax1.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.85, edgecolor=TEXT_COLOR, width=0.55)

            for bar, mean_val in zip(bars, means):
                if mean_val > 0:
                    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{mean_val:.4f}",
                             ha="center", va="bottom", color=TEXT_COLOR, fontsize=10, fontweight="bold")

            ax1.set_xticks(x)
            ax1.set_xticklabels(labels, color=TEXT_COLOR)
            ax1.set_ylabel("ArcFace Cosine Similarity ($S_{id}$)", color=TEXT_COLOR)
            ax1.set_ylim(0, max(max(means) + 0.15, 0.6))
            ax1.set_title("Identity Similarity $S_{id}$ across Paradigms", color=TEXT_COLOR, fontsize=12, pad=12, fontweight="bold")

            st.pyplot(fig1)
            plt.close(fig1)

            st.caption("Cosine similarity of 512-d ArcFace facial embeddings between input reference and progressed outputs.")
            st.markdown('</div>', unsafe_allow_html=True)

        # Plot 2: Target Age MAE
        with chart_col2:
            st.markdown('<div class="aw-card">', unsafe_allow_html=True)
            st.markdown('<div class="aw-card-title"><span>🎯</span> Target Age MAE (Lower is Better)</div>', unsafe_allow_html=True)

            fig2, ax2 = plt.subplots(figsize=(6, 4.2), dpi=120)
            apply_plot_theme(fig2, ax2)

            mae_means = []
            mae_stds = []
            for c in conditions:
                subset = df[df["condition"] == c]["target_age_mae"]
                mae_means.append(subset.mean() if not subset.empty else 0.0)
                mae_stds.append(subset.std() if len(subset) > 1 else 0.0)

            bars2 = ax2.bar(x, mae_means, yerr=mae_stds, capsize=5, color=colors, alpha=0.85, edgecolor=TEXT_COLOR, width=0.55)

            for bar, mean_val in zip(bars2, mae_means):
                if mean_val >= 0:
                    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4, f"{mean_val:.1f} y",
                             ha="center", va="bottom", color=TEXT_COLOR, fontsize=10, fontweight="bold")

            ax2.set_xticks(x)
            ax2.set_xticklabels(labels, color=TEXT_COLOR)
            ax2.set_ylabel("Mean Absolute Error (Years)", color=TEXT_COLOR)
            ax2.set_ylim(0, max(max(mae_means) + 5, 20))
            ax2.set_title("Target-Age Error $|\hat{y} - y_{tgt}|$ across Paradigms", color=TEXT_COLOR, fontsize=12, pad=12, fontweight="bold")

            st.pyplot(fig2)
            plt.close(fig2)

            st.caption("Discrepancy between requested target chronological age and ResNet CNN estimated age.")
            st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        col_pareto1, col_pareto2 = st.columns(2, gap="medium")

        # Plot 3: Pareto Frontier (Identity Similarity vs Progression Delta)
        with col_pareto1:
            st.markdown('<div class="aw-card">', unsafe_allow_html=True)
            st.markdown('<div class="aw-card-title"><span>⚖️</span> Identity vs. Age Progression Tradeoff</div>', unsafe_allow_html=True)

            fig3, ax3 = plt.subplots(figsize=(6, 4.2), dpi=120)
            apply_plot_theme(fig3, ax3)

            marker_map = {"generic_baseline": "o", "structured_prompt": "s", "adaptive_control": "^"}
            color_map = {"generic_baseline": COLOR_BASELINE, "structured_prompt": COLOR_STRUCTURED, "adaptive_control": COLOR_ADAPTIVE}
            name_map = {"generic_baseline": "Generic Baseline", "structured_prompt": "Structured Prompt", "adaptive_control": "Adaptive Control"}

            for cond_name in conditions:
                sub = df[df["condition"] == cond_name]
                if not sub.empty:
                    # Age delta = target_age - source_age
                    deltas = sub["target_age"] - sub["source_age"]
                    ax3.scatter(
                        deltas,
                        sub["identity_similarity"],
                        color=color_map[cond_name],
                        marker=marker_map[cond_name],
                        s=90,
                        alpha=0.9,
                        edgecolor="#FFFFFF",
                        linewidth=1.0,
                        label=name_map[cond_name]
                    )

            ax3.set_xlabel("Progression Age Jump ($\Delta y = y_{tgt} - y_{src}$ Years)", color=TEXT_COLOR)
            ax3.set_ylabel("ArcFace Identity Similarity ($S_{id}$)", color=TEXT_COLOR)
            ax3.set_title("Tradeoff: Identity Decay vs. Progression Jump", color=TEXT_COLOR, fontsize=12, pad=12, fontweight="bold")
            ax3.legend(facecolor=DARK_BG, edgecolor=SPINE_COLOR, labelcolor=TEXT_COLOR, loc="upper right", fontsize=9)
            ax3.set_ylim(0, 0.65)

            st.pyplot(fig3)
            plt.close(fig3)

            st.caption("Demonstrates the biological and diffusion tradeoff: larger age jumps require higher denoising strength, which naturally erodes facial geometry.")
            st.markdown('</div>', unsafe_allow_html=True)

        # Plot 4: Age Calibration Curve
        with col_pareto2:
            st.markdown('<div class="aw-card">', unsafe_allow_html=True)
            st.markdown('<div class="aw-card-title"><span>📐</span> Age Calibration Curve (Target vs Output)</div>', unsafe_allow_html=True)

            fig4, ax4 = plt.subplots(figsize=(6, 4.2), dpi=120)
            apply_plot_theme(fig4, ax4)

            # Diagonal line: perfect calibration y = x
            min_a = min(df["target_age"].min(), 35)
            max_a = max(df["target_age"].max(), 75)
            ax4.plot([min_a, max_a], [min_a, max_a], linestyle="--", color=COLOR_IDEAL, label="Perfect Calibration ($y = x$)", alpha=0.7, lw=1.8)

            for cond_name in conditions:
                sub = df[df["condition"] == cond_name]
                if not sub.empty:
                    ax4.scatter(
                        sub["target_age"],
                        sub["generated_estimated_age"],
                        color=color_map[cond_name],
                        marker=marker_map[cond_name],
                        s=85,
                        alpha=0.9,
                        edgecolor="#FFFFFF",
                        linewidth=1.0,
                        label=name_map[cond_name]
                    )

            ax4.set_xlabel("Requested Target Age ($y_{tgt}$ Years)", color=TEXT_COLOR)
            ax4.set_ylabel("CNN Estimated Output Age ($\hat{y}$ Years)", color=TEXT_COLOR)
            ax4.set_title("Target Age Calibration ($\hat{y}$ vs. $y_{tgt}$)", color=TEXT_COLOR, fontsize=12, pad=12, fontweight="bold")
            ax4.legend(facecolor=DARK_BG, edgecolor=SPINE_COLOR, labelcolor=TEXT_COLOR, loc="upper left", fontsize=9)

            st.pyplot(fig4)
            plt.close(fig4)

            st.caption("Points closer to the green dashed line indicate accurate age representation in generated facial morphology.")
            st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="aw-card">', unsafe_allow_html=True)
        st.markdown('<div class="aw-card-title"><span>📊</span> Aggregate Statistical Ledger</div>', unsafe_allow_html=True)

        table_df = cond_stats.copy()
        table_df.columns = ["Condition", "Mean Identity Sim", "Std Identity Sim", "Mean Target MAE", "Std Target MAE", "Mean Runtime (s)", "Trials"]
        table_df["Condition"] = table_df["Condition"].map(name_map).fillna(table_df["Condition"])
        table_df["Mean Identity Sim"] = table_df["Mean Identity Sim"].round(4)
        table_df["Std Identity Sim"] = table_df["Std Identity Sim"].round(4)
        table_df["Mean Target MAE"] = table_df["Mean Target MAE"].round(2)
        table_df["Std Target MAE"] = table_df["Std Target MAE"].round(2)
        table_df["Mean Runtime (s)"] = table_df["Mean Runtime (s)"].round(1)

        st.dataframe(table_df, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # =========================================================================
    # DOCUMENTED CHRONOLOGICAL AGE VS CNN ESTIMATION AUDIT
    # =========================================================================
    st.markdown('<div class="aw-card">', unsafe_allow_html=True)
    st.markdown('<div class="aw-card-title"><span>🔍</span> Documented Chronological Age vs. CNN Biometric Age Audit</div>', unsafe_allow_html=True)

    st.markdown("""
    A foundational methodological requirement in age-progression research is distinguishing **documented ground-truth chronological age**
    (established via historical archival metadata) from **automated CNN age estimates** (which carry domain shift biases).
    """, unsafe_allow_html=True)

    audit_data = [
        {
            "Subject ID": "ID001_RAJESH_KHANNA",
            "Provenance Source": "India Post Commemorative Stamp (~1970 archival release)",
            "Documented Chronological Age": "28 years old (b. Dec 1942)",
            "InsightFace CNN Estimated Age": "41.0 years old",
            "Discrepancy Rationale": "Commemorative stamp engraving texture, half-tone printing grain, and vintage color balance bias the off-the-shelf ResNet estimator upward by +13 years."
        },
        {
            "Subject ID": "ID009_DEEPIKA_PADUKONE",
            "Provenance Source": "Bollywood Hungama Press Archive (2008 adult portrait)",
            "Documented Chronological Age": "22 years old (b. Jan 1986)",
            "InsightFace CNN Estimated Age": "23.5 years old",
            "Discrepancy Rationale": "High-resolution modern digital sensor photography with even illumination aligns tightly with the CNN estimator's training distribution (error < 2 yrs)."
        },
        {
            "Subject ID": "SYNTH_ID_2024",
            "Provenance Source": "Stable Diffusion v1-5 Synthetic Prompt (Seed 2024)",
            "Documented Chronological Age": "23.0 years old (prompt target)",
            "InsightFace CNN Estimated Age": "23.0 years old",
            "Discrepancy Rationale": "Digital-native diffusion output directly captures young facial landmarks matching training distribution."
        }
    ]

    st.dataframe(pd.DataFrame(audit_data), use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # =========================================================================
    # HONEST RESEARCH & METRIC GAP DISCLOSURE
    # =========================================================================
    st.markdown('<div class="aw-card">', unsafe_allow_html=True)
    st.markdown('<div class="aw-card-title"><span>⚠️</span> Implementation Metric Disclosure & FaceTT Comparison</div>', unsafe_allow_html=True)

    st.markdown("""
    To maintain scientific integrity, AgeWeaver explicitly distinguishes between metrics genuinely computed on CPU
    and theoretical constructs described in literature (such as FaceTT):
    """, unsafe_allow_html=True)

    metric_gap_table = [
        {
            "Metric / Component": "ArcFace Identity Cosine Similarity (S_id)",
            "Status in AgeWeaver": "✅ Fully Implemented",
            "Operational Details": "Extracts 512-d L2-normalized embeddings using InsightFace buffalo_l (w600k_r50.onnx) and computes cosine similarity against source face."
        },
        {
            "Metric / Component": "CNN Estimated Age (y_hat) & Target MAE",
            "Status in AgeWeaver": "✅ Fully Implemented",
            "Operational Details": "Computed using InsightFace genderage.onnx ResNet regression model. MAE is calculated as |y_hat - y_tgt|."
        },
        {
            "Metric / Component": "Cyclic Round-Trip Consistency (S_cyc)",
            "Status in AgeWeaver": "✅ Fully Implemented (Condition C)",
            "Operational Details": "Progressed image is inverted back to source age (y_tgt -> y_src) to test trajectory bijectivity (FaceTT Section 5.4)."
        },
        {
            "Metric / Component": "Adaptive Aging Controller (AAC)",
            "Status in AgeWeaver": "⚡ Heuristic Approximation",
            "Operational Details": "FaceTT learns cross-attention weights via GPU training. AgeWeaver approximates this via closed-form delta-scaled denoising strength and guidance curves on CPU."
        },
        {
            "Metric / Component": "Angular Inversion & Identity Loss Guidance",
            "Status in AgeWeaver": "❌ Research Gap (Documented Limitation)",
            "Operational Details": "Requires computing backpropagation loss gradients through diffusion UNet at every sampling step (grad_z_t L_id). Omitted due to CPU memory and latency constraints."
        }
    ]

    st.dataframe(pd.DataFrame(metric_gap_table), use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)
