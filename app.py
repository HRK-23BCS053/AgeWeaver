"""
AgeWeaver — Premium Research Interface for Identity-Preserving Facial Age Progression.
Entrypoint for Streamlit application.

Navigation:
- Studio: Biometric inspection, target-age selection, CPU-optimized synthesis, side-by-side comparison.
- Experiments: Auditable visual trial gallery and downloadable CSV ledger.
- Research Metrics: Empirical identity preservation vs age error Pareto trade-off and calibration curves.
- About: Mathematical foundations, honest FaceTT gap analysis, dataset provenance, and IIIT Dharwad context.
"""

import os
import sys

# Ensure src and root are in sys.path
sys.path.insert(0, os.path.abspath("src"))
sys.path.insert(0, os.path.abspath("."))

import streamlit as st

# Configure page layout and browser title
st.set_page_config(
    page_title="AgeWeaver Studio",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

from ageweaver.frontend.styles import inject_styles
from ageweaver.frontend.state import init_session_state
from ageweaver.frontend.studio import render_studio_view
from ageweaver.frontend.experiments_view import render_experiments_view
from ageweaver.frontend.metrics_view import render_metrics_view
from ageweaver.frontend.about_view import render_about_view

def main():
    # Inject dark midnight-navy theme and glassmorphic styles
    inject_styles()

    # Initialize state
    init_session_state()

    # =========================================================================
    # SIDEBAR NAVIGATION
    # =========================================================================
    with st.sidebar:
        st.markdown("""
        <div style="padding: 10px 0 18px 0; border-bottom: 1px solid rgba(139, 92, 246, 0.2); margin-bottom: 20px;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="width: 38px; height: 38px; border-radius: 10px; background: linear-gradient(135deg, #7C3AED 0%, #06B6D4 100%); display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 18px; color: #FFFFFF; box-shadow: 0 0 16px rgba(124, 58, 237, 0.4);">AW</div>
                <div>
                    <div style="font-size: 18px; font-weight: 800; color: #F8FAFC; letter-spacing: -0.02em;">AgeWeaver</div>
                    <div style="font-size: 11px; color: #94A3B8; font-weight: 500;">Research Interface • v2.0</div>
                </div>
            </div>
            <div style="margin-top: 10px;">
                <span class="aw-badge aw-badge-violet">IIIT Dharwad • Sem-7</span>
                <span class="aw-badge aw-badge-cyan">CPU Mode</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        destination = st.radio(
            "Navigation Menu",
            [
                "🧬 Studio",
                "📁 Experiments",
                "📈 Research Metrics",
                "ℹ️ About & Methodology"
            ],
            index=0,
            label_visibility="collapsed"
        )

        st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)

        # Hardware & Engine Status Card in Sidebar
        st.markdown("""
        <div class="aw-card" style="padding: 14px; margin-bottom: 14px;">
            <div style="font-size: 11px; text-transform: uppercase; color: #94A3B8; font-weight: 700; margin-bottom: 8px; letter-spacing: 0.05em;">System Environment</div>
            <div style="font-size: 12px; color: #CBD5E1; line-height: 1.6;">
                <div>• <b>Hardware</b>: Intel Iris Xe CPU</div>
                <div>• <b>RAM</b>: 16 GB System Memory</div>
                <div>• <b>PyTorch</b>: CPU Optimized (DDIM 15)</div>
                <div>• <b>Generative Base</b>: SD v1-5 (local)</div>
                <div>• <b>Biometrics</b>: InsightFace buffalo_l</div>
                <div>• <b>Storage</b>: 100% Offline / Local</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Quick Links / Footer
        st.markdown("""
        <div style="padding: 10px; font-size: 11px; color: #64748B; text-align: center;">
            AgeWeaver Research Project<br>
            IIIT Dharwad • October 2026
        </div>
        """, unsafe_allow_html=True)

    # =========================================================================
    # VIEW ROUTING
    # =========================================================================
    if destination == "🧬 Studio":
        render_studio_view()
    elif destination == "📁 Experiments":
        render_experiments_view()
    elif destination == "📈 Research Metrics":
        render_metrics_view()
    elif destination == "ℹ️ About & Methodology":
        render_about_view()

if __name__ == "__main__":
    main()
