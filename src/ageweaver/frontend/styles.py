"""
Visual identity and custom CSS for AgeWeaver Streamlit application.
Palette:
- Dark Midnight-Navy Base: #0A0E1A, #0D1326, #131B35
- Electric Violet Primary: #7C3AED, #8B5CF6, #A78BFA
- Cyan Secondary Highlight: #06B6D4, #22D3EE, #38BDF8
- Silver-White Typography: #F8FAFC, #E2E8F0, #94A3B8
- Glassmorphic Cards: rgba(19, 27, 53, 0.7) with rgba(139, 92, 246, 0.2) borders
"""

CUSTOM_CSS = """
<style>
/* Global Imports */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* Main Background and Typography */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0A0E1A;
    color: #F8FAFC;
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

[data-testid="stHeader"] {
    background: rgba(10, 14, 26, 0.85);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid rgba(139, 92, 246, 0.15);
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #0D1326;
    border-right: 1px solid rgba(139, 92, 246, 0.15);
}

[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #F8FAFC;
}

/* AgeWeaver Brand Header */
.ageweaver-brand {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 12px 0 24px 0;
    border-bottom: 1px solid rgba(139, 92, 246, 0.2);
    margin-bottom: 24px;
}

.ageweaver-logo-icon {
    width: 44px;
    height: 44px;
    border-radius: 12px;
    background: linear-gradient(135deg, #7C3AED 0%, #06B6D4 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 0 24px rgba(124, 58, 237, 0.4);
    font-weight: 800;
    font-size: 22px;
    color: #FFFFFF;
}

.ageweaver-title-group h1 {
    font-size: 24px;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin: 0;
    background: linear-gradient(135deg, #FFFFFF 30%, #A78BFA 70%, #22D3EE 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.ageweaver-title-group p {
    font-size: 13px;
    color: #94A3B8;
    margin: 2px 0 0 0;
    font-weight: 500;
}

/* Glassmorphic Container Cards */
.aw-card {
    background: rgba(19, 27, 53, 0.65);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(139, 92, 246, 0.18);
    border-radius: 16px;
    padding: 22px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
    transition: all 0.25s ease-in-out;
}

.aw-card:hover {
    border-color: rgba(6, 182, 212, 0.35);
    box-shadow: 0 8px 36px rgba(124, 58, 237, 0.18);
}

.aw-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(148, 163, 184, 0.1);
}

.aw-card-title {
    font-size: 16px;
    font-weight: 700;
    color: #F8FAFC;
    display: flex;
    align-items: center;
    gap: 8px;
    letter-spacing: -0.01em;
}

/* Metric Display Pills */
.aw-metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 12px;
    margin: 16px 0;
}

.aw-metric-box {
    background: rgba(13, 19, 38, 0.8);
    border: 1px solid rgba(139, 92, 246, 0.16);
    border-radius: 12px;
    padding: 12px 14px;
    text-align: left;
}

.aw-metric-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94A3B8;
    margin-bottom: 4px;
    font-weight: 600;
}

.aw-metric-val {
    font-size: 19px;
    font-weight: 700;
    color: #F8FAFC;
    font-family: 'JetBrains Mono', monospace;
}

.aw-metric-val.violet {
    color: #A78BFA;
}

.aw-metric-val.cyan {
    color: #22D3EE;
}

.aw-metric-val.green {
    color: #34D399;
}

/* Pill Badges */
.aw-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.02em;
}

.aw-badge-violet {
    background: rgba(124, 58, 237, 0.18);
    color: #C4B5FD;
    border: 1px solid rgba(139, 92, 246, 0.35);
}

.aw-badge-cyan {
    background: rgba(6, 182, 212, 0.16);
    color: #67E8F9;
    border: 1px solid rgba(6, 182, 212, 0.35);
}

.aw-badge-gray {
    background: rgba(148, 163, 184, 0.12);
    color: #CBD5E1;
    border: 1px solid rgba(148, 163, 184, 0.25);
}

/* Portrait Comparison Layout */
.aw-compare-container {
    display: flex;
    gap: 20px;
    margin: 16px 0;
}

.aw-portrait-card {
    flex: 1;
    background: rgba(13, 19, 38, 0.85);
    border: 1px solid rgba(139, 92, 246, 0.2);
    border-radius: 14px;
    overflow: hidden;
    position: relative;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}

.aw-portrait-header {
    padding: 10px 14px;
    background: rgba(19, 27, 53, 0.95);
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid rgba(139, 92, 246, 0.15);
    font-size: 13px;
    font-weight: 600;
    color: #E2E8F0;
}

/* Button & Action Overrides */
div.stButton > button:first-child {
    background: linear-gradient(135deg, #7C3AED 0%, #6366F1 50%, #06B6D4 100%);
    color: #FFFFFF;
    font-weight: 700;
    font-size: 15px;
    padding: 12px 24px;
    border-radius: 12px;
    border: none;
    box-shadow: 0 4px 20px rgba(124, 58, 237, 0.35);
    transition: all 0.2s ease-in-out;
    width: 100%;
}

div.stButton > button:first-child:hover {
    background: linear-gradient(135deg, #8B5CF6 0%, #7C3AED 50%, #22D3EE 100%);
    box-shadow: 0 6px 28px rgba(124, 58, 237, 0.55);
    transform: translateY(-1px);
}

div.stButton > button:first-child:active {
    transform: translateY(1px);
}

/* Navigation Radio Tabs Styling */
[data-testid="stSidebarNav"] {
    display: none;
}

/* Slider, Selectbox & Input Customization */
.stSlider [data-baseweb="slider"] {
    margin-top: 10px;
}

[data-testid="stSelectbox"] > div, [data-testid="stTextInput"] > div {
    background-color: rgba(19, 27, 53, 0.8) !important;
    border: 1px solid rgba(139, 92, 246, 0.25) !important;
    border-radius: 10px !important;
    color: #F8FAFC !important;
}

/* Progress and Spinner */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #7C3AED, #06B6D4);
}

/* Table and DataFrame Customization */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(139, 92, 246, 0.18);
    border-radius: 12px;
    overflow: hidden;
}

/* Callout / Alert Box */
.aw-callout {
    background: rgba(124, 58, 237, 0.1);
    border-left: 3px solid #8B5CF6;
    padding: 14px 18px;
    border-radius: 0 10px 10px 0;
    margin: 14px 0;
    font-size: 13px;
    color: #E2E8F0;
    line-height: 1.6;
}

.aw-callout.cyan {
    background: rgba(6, 182, 212, 0.1);
    border-left-color: #06B6D4;
}
</style>
"""

def inject_styles():
    """Injects custom CSS into the Streamlit session."""
    import streamlit as st
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
