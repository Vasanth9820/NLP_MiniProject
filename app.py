"""
app.py
------
AI Content & Plagiarism Detection System — Streamlit Application
Clean UI with Dark / Light mode toggle.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.document_parser import load_documents
from src.preprocessing import preprocess_text
from src.plagiarism import (
    compute_document_similarity,
    find_all_sentence_matches,
    get_similarity_label,
)
from src.ai_detector import get_detector, AI_DETECTION_DISCLAIMER
from src.report_generator import generate_text_report, generate_html_report

# ──────────────────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Plagiarism Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Session state defaults
# ──────────────────────────────────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "results" not in st.session_state:
    st.session_state.results = {}


# ──────────────────────────────────────────────────────────────────────────────
# Theme palette
# ──────────────────────────────────────────────────────────────────────────────
def _palette(dark: bool) -> dict:
    if dark:
        return dict(
            bg="#0f1117",
            surface="#1a1d2e",
            card="#22263a",
            card_hover="#272b40",
            border="#2d3255",
            accent="#4f8ef7",
            accent2="#7c3aed",
            text="#e8eaed",
            text_muted="#8b92a8",
            text_inverse="#0f1117",
            success="#22c55e",
            warning="#f59e0b",
            danger="#ef4444",
            info="#38bdf8",
            sidebar_bg="#13162b",
            input_bg="#1e2236",
            shadow="rgba(0,0,0,0.4)",
            plotly_template="plotly_dark",
            badge_low_bg="#052e16", badge_low_fg="#4ade80",
            badge_mod_bg="#451a03", badge_mod_fg="#fb923c",
            badge_high_bg="#431407", badge_high_fg="#f97316",
            badge_vhigh_bg="#450a0a", badge_vhigh_fg="#f87171",
        )
    else:
        return dict(
            bg="#f1f5f9",
            surface="#ffffff",
            card="#ffffff",
            card_hover="#f8fafc",
            border="#e2e8f0",
            accent="#2563eb",
            accent2="#7c3aed",
            text="#1e293b",
            text_muted="#64748b",
            text_inverse="#ffffff",
            success="#16a34a",
            warning="#d97706",
            danger="#dc2626",
            info="#0284c7",
            sidebar_bg="#f8fafc",
            input_bg="#f1f5f9",
            shadow="rgba(0,0,0,0.08)",
            plotly_template="plotly_white",
            badge_low_bg="#dcfce7", badge_low_fg="#15803d",
            badge_mod_bg="#fef3c7", badge_mod_fg="#92400e",
            badge_high_bg="#ffedd5", badge_high_fg="#9a3412",
            badge_vhigh_bg="#fee2e2", badge_vhigh_fg="#991b1b",
        )


# ──────────────────────────────────────────────────────────────────────────────
# CSS injection
# ──────────────────────────────────────────────────────────────────────────────
def inject_css(p: dict) -> None:
    st.markdown(
        f"""
        <style>
        /* ── Google Font ── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        /* ── Root overrides ── */
        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif !important;
        }}

        /* ── App background ── */
        .stApp {{
            background-color: {p['bg']} !important;
            color: {p['text']} !important;
        }}

        /* ── Main content area ── */
        .block-container {{
            padding: 1.5rem 2rem 2rem 2rem !important;
            max-width: 1280px !important;
        }}

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {{
            background: {p['sidebar_bg']} !important;
            border-right: 1px solid {p['border']} !important;
        }}
        [data-testid="stSidebar"] * {{
            color: {p['text']} !important;
        }}
        [data-testid="stSidebarContent"] {{
            padding: 1.5rem 1rem !important;
        }}

        /* ── Sidebar header ── */
        .sidebar-logo {{
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 0.5rem 0 1rem 0;
            border-bottom: 1px solid {p['border']};
            margin-bottom: 1.2rem;
        }}
        .sidebar-logo .logo-icon {{
            font-size: 1.8rem;
            line-height: 1;
        }}
        .sidebar-logo .logo-text {{
            font-size: 1.05rem;
            font-weight: 700;
            color: {p['accent']} !important;
            line-height: 1.2;
        }}
        .sidebar-logo .logo-sub {{
            font-size: 0.7rem;
            color: {p['text_muted']} !important;
        }}

        /* ── Section headers in sidebar ── */
        .sidebar-section {{
            font-size: 0.65rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: {p['text_muted']} !important;
            margin: 1.2rem 0 0.5rem 0;
        }}

        /* ── Inputs & sliders ── */
        .stTextArea textarea, .stTextInput input {{
            background: {p['input_bg']} !important;
            border: 1px solid {p['border']} !important;
            border-radius: 8px !important;
            color: {p['text']} !important;
            font-size: 0.875rem !important;
        }}
        .stTextArea textarea:focus, .stTextInput input:focus {{
            border-color: {p['accent']} !important;
            box-shadow: 0 0 0 3px {p['accent']}22 !important;
        }}
        .stSlider [data-testid="stTickBarMin"],
        .stSlider [data-testid="stTickBarMax"] {{
            color: {p['text_muted']} !important;
        }}

        /* ── File uploader ── */
        [data-testid="stFileUploader"] {{
            border: 2px dashed {p['border']} !important;
            border-radius: 10px !important;
            background: {p['input_bg']} !important;
            transition: border-color 0.2s;
        }}
        [data-testid="stFileUploader"]:hover {{
            border-color: {p['accent']} !important;
        }}
        [data-testid="stFileUploaderDropzone"] p,
        [data-testid="stFileUploaderDropzone"] span {{
            color: {p['text_muted']} !important;
        }}

        /* ── Primary button ── */
        .stButton > button[kind="primary"] {{
            background: linear-gradient(135deg, {p['accent']}, {p['accent2']}) !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
            padding: 0.6rem 1.2rem !important;
            transition: all 0.2s ease !important;
            box-shadow: 0 4px 15px {p['accent']}44 !important;
        }}
        .stButton > button[kind="primary"]:hover {{
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 20px {p['accent']}66 !important;
        }}

        /* ── Secondary button ── */
        .stButton > button:not([kind="primary"]) {{
            background: {p['card']} !important;
            color: {p['text']} !important;
            border: 1px solid {p['border']} !important;
            border-radius: 10px !important;
            font-weight: 500 !important;
            font-size: 0.85rem !important;
            transition: all 0.2s ease !important;
        }}
        .stButton > button:not([kind="primary"]):hover {{
            border-color: {p['accent']} !important;
            color: {p['accent']} !important;
        }}

        /* ── Tabs ── */
        [data-testid="stTabs"] [data-testid="stTab"] {{
            font-weight: 500 !important;
            font-size: 0.875rem !important;
            color: {p['text_muted']} !important;
            border-radius: 8px 8px 0 0 !important;
            padding: 0.6rem 1.2rem !important;
            transition: all 0.2s !important;
        }}
        [data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"] {{
            color: {p['accent']} !important;
            border-bottom: 2px solid {p['accent']} !important;
            background: {p['surface']}88 !important;
        }}
        [data-testid="stTabContent"] {{
            background: {p['surface']} !important;
            border: 1px solid {p['border']} !important;
            border-radius: 0 12px 12px 12px !important;
            padding: 1.5rem !important;
        }}

        /* ── Expander ── */
        [data-testid="stExpander"] {{
            background: {p['card']} !important;
            border: 1px solid {p['border']} !important;
            border-radius: 10px !important;
            margin-bottom: 0.5rem !important;
            overflow: hidden !important;
        }}
        [data-testid="stExpander"] summary {{
            color: {p['text']} !important;
            font-weight: 500 !important;
            font-size: 0.875rem !important;
        }}

        /* ── Dataframe / tables ── */
        [data-testid="stDataFrame"] {{
            border: 1px solid {p['border']} !important;
            border-radius: 10px !important;
            overflow: hidden !important;
        }}
        [data-testid="stDataFrame"] th {{
            background: {p['accent']}18 !important;
            color: {p['text']} !important;
            font-weight: 600 !important;
        }}

        /* ── Metric widget ── */
        [data-testid="stMetric"] {{
            background: {p['card']} !important;
            border: 1px solid {p['border']} !important;
            border-radius: 12px !important;
            padding: 1rem 1.2rem !important;
            box-shadow: 0 2px 8px {p['shadow']} !important;
        }}
        [data-testid="stMetricLabel"] p {{
            color: {p['text_muted']} !important;
            font-size: 0.8rem !important;
            font-weight: 500 !important;
            letter-spacing: 0.03em !important;
        }}
        [data-testid="stMetricValue"] {{
            color: {p['text']} !important;
            font-size: 1.6rem !important;
            font-weight: 700 !important;
        }}

        /* ── Alerts / callouts ── */
        [data-testid="stAlert"] {{
            border-radius: 10px !important;
            border-left-width: 4px !important;
        }}
        .stSuccess {{
            background: {p['badge_low_bg']}88 !important;
        }}
        .stWarning {{
            background: {p['badge_mod_bg']}88 !important;
        }}
        .stError {{
            background: {p['badge_vhigh_bg']}88 !important;
        }}
        .stInfo {{
            background: {p['accent']}11 !important;
        }}

        /* ── Checkboxes ── */
        [data-testid="stCheckbox"] span {{
            color: {p['text']} !important;
        }}

        /* ── Divider ── */
        hr {{
            border-color: {p['border']} !important;
        }}

        /* ── Scrollbar ── */
        ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
        ::-webkit-scrollbar-track {{ background: {p['bg']}; }}
        ::-webkit-scrollbar-thumb {{ background: {p['border']}; border-radius: 4px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: {p['accent']}; }}

        /* ── Custom component classes ── */
        .page-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.5rem 0 1.5rem 0;
            border-bottom: 1px solid {p['border']};
            margin-bottom: 1.5rem;
        }}
        .page-title {{
            font-size: 1.75rem;
            font-weight: 800;
            background: linear-gradient(135deg, {p['accent']}, {p['accent2']});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 0;
            line-height: 1.2;
        }}
        .page-subtitle {{
            font-size: 0.875rem;
            color: {p['text_muted']};
            margin: 0.2rem 0 0 0;
        }}
        .theme-toggle {{
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 6px 14px;
            background: {p['card']};
            border: 1px solid {p['border']};
            border-radius: 50px;
            font-size: 0.8rem;
            font-weight: 500;
            color: {p['text_muted']};
            cursor: pointer;
        }}

        /* ── Stat card ── */
        .stat-card {{
            background: {p['card']};
            border: 1px solid {p['border']};
            border-radius: 14px;
            padding: 1.2rem 1.4rem;
            box-shadow: 0 2px 8px {p['shadow']};
            transition: box-shadow 0.2s, transform 0.2s;
            text-align: center;
            height: 100%;
        }}
        .stat-card:hover {{
            box-shadow: 0 4px 16px {p['shadow']};
            transform: translateY(-2px);
        }}
        .stat-value {{
            font-size: 2rem;
            font-weight: 800;
            color: {p['accent']};
            line-height: 1.1;
            margin-bottom: 0.3rem;
        }}
        .stat-label {{
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: {p['text_muted']};
        }}
        .stat-icon {{
            font-size: 1.4rem;
            margin-bottom: 0.5rem;
        }}

        /* ── Section card ── */
        .section-card {{
            background: {p['card']};
            border: 1px solid {p['border']};
            border-radius: 14px;
            padding: 1.4rem 1.6rem;
            box-shadow: 0 2px 8px {p['shadow']};
            margin-bottom: 1rem;
        }}
        .section-title {{
            font-size: 1rem;
            font-weight: 700;
            color: {p['text']};
            margin: 0 0 1rem 0;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        /* ── Similarity badge ── */
        .badge-low {{
            background: {p['badge_low_bg']};
            color: {p['badge_low_fg']};
            padding: 2px 10px;
            border-radius: 50px;
            font-size: 0.75rem;
            font-weight: 600;
            white-space: nowrap;
        }}
        .badge-moderate {{
            background: {p['badge_mod_bg']};
            color: {p['badge_mod_fg']};
            padding: 2px 10px;
            border-radius: 50px;
            font-size: 0.75rem;
            font-weight: 600;
            white-space: nowrap;
        }}
        .badge-high {{
            background: {p['badge_high_bg']};
            color: {p['badge_high_fg']};
            padding: 2px 10px;
            border-radius: 50px;
            font-size: 0.75rem;
            font-weight: 600;
            white-space: nowrap;
        }}
        .badge-very-high {{
            background: {p['badge_vhigh_bg']};
            color: {p['badge_vhigh_fg']};
            padding: 2px 10px;
            border-radius: 50px;
            font-size: 0.75rem;
            font-weight: 600;
            white-space: nowrap;
        }}

        /* ── AI prediction card ── */
        .pred-card {{
            background: {p['card']};
            border: 1px solid {p['border']};
            border-radius: 14px;
            padding: 1.4rem;
            box-shadow: 0 2px 8px {p['shadow']};
            margin-bottom: 1rem;
            transition: all 0.2s;
        }}
        .pred-card:hover {{
            box-shadow: 0 4px 16px {p['shadow']};
        }}
        .pred-filename {{
            font-size: 0.8rem;
            color: {p['text_muted']};
            font-weight: 500;
            margin-bottom: 0.3rem;
        }}
        .pred-label-ai {{
            font-size: 1.1rem;
            font-weight: 700;
            color: {p['danger']};
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .pred-label-human {{
            font-size: 1.1rem;
            font-weight: 700;
            color: {p['success']};
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .pred-label-uncertain {{
            font-size: 1.1rem;
            font-weight: 700;
            color: {p['warning']};
            display: flex;
            align-items: center;
            gap: 6px;
        }}

        /* ── Doc info row ── */
        .doc-row {{
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 0.75rem 1rem;
            background: {p['card']};
            border: 1px solid {p['border']};
            border-radius: 10px;
            margin-bottom: 0.5rem;
        }}
        .doc-icon {{
            font-size: 1.4rem;
            flex-shrink: 0;
        }}
        .doc-name {{
            font-weight: 600;
            font-size: 0.9rem;
            color: {p['text']};
            flex: 1;
            min-width: 0;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        .doc-meta {{
            font-size: 0.75rem;
            color: {p['text_muted']};
            white-space: nowrap;
        }}

        /* ── Sentence match pair ── */
        .sent-box {{
            background: {p['input_bg']};
            border: 1px solid {p['border']};
            border-radius: 8px;
            padding: 0.75rem 1rem;
            font-size: 0.85rem;
            font-style: italic;
            color: {p['text']};
            line-height: 1.5;
        }}
        .sent-label {{
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: {p['text_muted']};
            margin-bottom: 0.3rem;
        }}

        /* ── Progress bar custom ── */
        .stProgress > div > div > div > div {{
            background: linear-gradient(90deg, {p['accent']}, {p['accent2']}) !important;
            border-radius: 4px !important;
        }}
        .stProgress > div > div {{
            background: {p['border']} !important;
            border-radius: 4px !important;
        }}

        /* ── How it works flow ── */
        .flow-step {{
            background: {p['card']};
            border: 1px solid {p['border']};
            border-radius: 10px;
            padding: 0.9rem 1.1rem;
            font-size: 0.82rem;
            color: {p['text_muted']};
            line-height: 1.6;
        }}
        .flow-step strong {{
            color: {p['accent']};
            font-size: 0.78rem;
            letter-spacing: 0.05em;
            display: block;
            margin-bottom: 0.3rem;
            text-transform: uppercase;
        }}
        .flow-arrow {{
            text-align: center;
            color: {p['text_muted']};
            font-size: 1.2rem;
            padding: 0.1rem 0;
        }}

        /* ── Disclaimer box ── */
        .disclaimer {{
            background: {p['badge_mod_bg']};
            border-left: 4px solid {p['warning']};
            border-radius: 0 8px 8px 0;
            padding: 0.8rem 1rem;
            font-size: 0.8rem;
            color: {p['text']};
            line-height: 1.5;
        }}

        /* ── Download buttons ── */
        .stDownloadButton > button {{
            background: {p['card']} !important;
            color: {p['accent']} !important;
            border: 1.5px solid {p['accent']} !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            font-size: 0.875rem !important;
            transition: all 0.2s !important;
        }}
        .stDownloadButton > button:hover {{
            background: {p['accent']} !important;
            color: white !important;
        }}

        /* ── Footer ── */
        .app-footer {{
            text-align: center;
            font-size: 0.72rem;
            color: {p['text_muted']};
            padding: 2rem 0 1rem 0;
            border-top: 1px solid {p['border']};
            margin-top: 2rem;
        }}

        /* ── Hide Streamlit branding ── */
        #MainMenu {{ visibility: hidden; }}
        footer {{ visibility: hidden; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Plotly theme helper
# ──────────────────────────────────────────────────────────────────────────────
def _plotly_layout(p: dict, title: str, **kwargs) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=14, color=p["text"], family="Inter")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=p["text"], family="Inter", size=12),
        xaxis=dict(
            gridcolor=p["border"], linecolor=p["border"],
            tickfont=dict(color=p["text_muted"]),
            title_font=dict(color=p["text_muted"]),
        ),
        yaxis=dict(
            gridcolor=p["border"], linecolor=p["border"],
            tickfont=dict(color=p["text_muted"]),
            title_font=dict(color=p["text_muted"]),
        ),
        showlegend=kwargs.pop("showlegend", True),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=p["text_muted"]),
        ),
        margin=dict(l=30, r=20, t=50, b=40),
        height=kwargs.pop("height", 380),
        **kwargs,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Badge helper
# ──────────────────────────────────────────────────────────────────────────────
def _badge_html(status: str) -> str:
    cls = {
        "Low": "badge-low",
        "Moderate": "badge-moderate",
        "High": "badge-high",
        "Very High": "badge-very-high",
    }.get(status, "badge-low")
    dot = {"Low": "🟢", "Moderate": "🟡", "High": "🟠", "Very High": "🔴"}.get(status, "")
    return f'<span class="{cls}">{dot} {status}</span>'


# ──────────────────────────────────────────────────────────────────────────────
# Apply theme
# ──────────────────────────────────────────────────────────────────────────────
dark = st.session_state.dark_mode
p = _palette(dark)
inject_css(p)


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo
    st.markdown(
        """
        <div class="sidebar-logo">
            <div class="logo-icon">🔍</div>
            <div>
                <div class="logo-text">PlagiarismAI</div>
                <div class="logo-sub">NLP Detection System</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Theme toggle ──
    st.markdown('<div class="sidebar-section">Appearance</div>', unsafe_allow_html=True)
    col_light, col_dark = st.columns(2)
    with col_light:
        if st.button(
            "☀️ Light" if dark else "☀️ Light ✓",
            use_container_width=True,
            type="secondary",
        ):
            st.session_state.dark_mode = False
            st.rerun()
    with col_dark:
        if st.button(
            "🌙 Dark ✓" if dark else "🌙 Dark",
            use_container_width=True,
            type="secondary",
        ):
            st.session_state.dark_mode = True
            st.rerun()

    st.markdown("---")

    # ── Upload ──
    st.markdown('<div class="sidebar-section">📂 Documents</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload files (TXT · PDF · DOCX)",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-section">✏️ Paste Text</div>', unsafe_allow_html=True)
    pasted_text = st.text_area(
        "Paste text",
        height=110,
        placeholder="Type or paste your text here…",
        label_visibility="collapsed",
    )
    paste_label = st.text_input("Label", value="Pasted Text", label_visibility="collapsed")

    st.markdown("---")

    # ── Settings ──
    st.markdown('<div class="sidebar-section">⚙️ Settings</div>', unsafe_allow_html=True)

    sim_threshold = st.slider(
        "Similarity alert threshold (%)",
        0, 100, 60, 5,
        help="Pairs above this % are flagged.",
    )
    sent_threshold = st.slider(
        "Sentence match threshold (%)",
        40, 100, 70, 5,
        help="Sentences above this % are shown.",
    )
    run_ai = st.checkbox("Enable AI content detection", value=True)

    st.markdown("---")

    # ── Actions ──
    analyse_btn = st.button("🔍  Analyse Documents", type="primary", use_container_width=True)
    if st.button("↺  Clear & Reset", use_container_width=True):
        st.session_state.analysis_done = False
        st.session_state.results = {}
        st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# PAGE HEADER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="page-header">
        <div>
            <h1 class="page-title">AI Content & Plagiarism Detector</h1>
            <p class="page-subtitle">
                NLP-powered document similarity analysis and AI-generated text detection
            </p>
        </div>
        <div style="font-size:0.78rem;color:{p['text_muted']};text-align:right;">
            {'🌙 Dark Mode' if dark else '☀️ Light Mode'}<br>
            <span style="font-size:0.68rem;">Toggle in sidebar</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# HOW IT WORKS (collapsed by default)
# ──────────────────────────────────────────────────────────────────────────────
with st.expander("ℹ️  How It Works", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"""
            <div class="section-title">📄 Plagiarism Detection</div>
            <div class="flow-step"><strong>Step 1 — Preprocessing</strong>
            Lowercase → remove URLs & punctuation → tokenise → remove stopwords → lemmatise</div>
            <div class="flow-arrow">↓</div>
            <div class="flow-step"><strong>Step 2 — TF-IDF Vectorisation</strong>
            Each document is converted to a vector of weighted word importance scores.</div>
            <div class="flow-arrow">↓</div>
            <div class="flow-step"><strong>Step 3 — Cosine Similarity</strong>
            The angle between two document vectors is measured (0–100%).
            Higher score = more textual overlap.</div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="section-title">🤖 AI Content Detection</div>
            <div class="flow-step"><strong>Step 1 — Feature Extraction</strong>
            Text is preprocessed and converted to TF-IDF features (unigrams + bigrams + trigrams).</div>
            <div class="flow-arrow">↓</div>
            <div class="flow-step"><strong>Step 2 — Logistic Regression</strong>
            A trained classifier predicts the probability of AI-generated vs. human-written.</div>
            <div class="flow-arrow">↓</div>
            <div class="flow-step"><strong>Step 3 — Confidence Score</strong>
            The higher-probability class becomes the prediction.
            Results below 55% confidence are marked "Uncertain".</div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown(
        f'<div class="disclaimer">⚠️ {AI_DETECTION_DISCLAIMER.replace("⚠️ **Disclaimer:** ", "")}</div>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# ANALYSIS TRIGGER
# ──────────────────────────────────────────────────────────────────────────────
if analyse_btn:
    docs: list[dict] = []
    if uploaded_files:
        with st.spinner("Parsing uploaded documents…"):
            docs.extend(load_documents(uploaded_files))
    if pasted_text.strip():
        docs.append({"name": paste_label, "text": pasted_text.strip(), "error": None})

    for err_doc in [d for d in docs if d["error"]]:
        st.error(f"❌ **{err_doc['name']}** — {err_doc['error']}")

    valid_docs = [d for d in docs if not d["error"] and d["text"].strip()]

    if not valid_docs:
        st.warning("⚠️ No valid documents found. Upload files or paste text in the sidebar.")
        st.stop()

    results: dict = dict(
        doc_names=[d["name"] for d in valid_docs],
        doc_texts=[d["text"] for d in valid_docs],
        matrix_df=pd.DataFrame(),
        pairwise_df=pd.DataFrame(),
        sentence_matches=[],
        ai_results=[],
        sim_threshold=sim_threshold,
        sent_threshold=sent_threshold,
    )

    if len(valid_docs) >= 2:
        with st.spinner("Computing similarity scores…"):
            try:
                results["matrix_df"], results["pairwise_df"] = compute_document_similarity(
                    results["doc_texts"], results["doc_names"]
                )
            except Exception as exc:
                st.error(f"Similarity error: {exc}")
        with st.spinner("Finding sentence-level matches…"):
            try:
                results["sentence_matches"] = find_all_sentence_matches(
                    results["doc_texts"], results["doc_names"],
                    threshold=sent_threshold / 100,
                )
            except Exception as exc:
                st.warning(f"Sentence matching skipped: {exc}")
    else:
        st.info("ℹ️ Upload at least 2 documents for similarity analysis.")

    if run_ai:
        detector = get_detector()
        ai_list = []
        with st.spinner("Running AI content detection…"):
            for doc in valid_docs:
                r = detector.predict(doc["text"])
                r["name"] = doc["name"]
                ai_list.append(r)
        results["ai_results"] = ai_list

    st.session_state.results = results
    st.session_state.analysis_done = True
    st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# RESULTS
# ──────────────────────────────────────────────────────────────────────────────
if st.session_state.analysis_done and st.session_state.results:
    res = st.session_state.results
    doc_names: list[str] = res["doc_names"]
    doc_texts: list[str] = res["doc_texts"]
    pairwise_df: pd.DataFrame = res["pairwise_df"]
    matrix_df: pd.DataFrame = res["matrix_df"]
    sentence_matches: list[dict] = res["sentence_matches"]
    ai_results: list[dict] = res["ai_results"]
    sim_threshold = res["sim_threshold"]

    tab_ov, tab_sim, tab_ai, tab_rep = st.tabs(
        ["📊  Overview", "📄  Similarity", "🤖  AI Detection", "📋  Report"]
    )

    # ══════════════════════════════════════════════════════════
    # TAB — OVERVIEW
    # ══════════════════════════════════════════════════════════
    with tab_ov:
        # ── Stat cards ──
        highest = pairwise_df["Similarity (%)"].max() if not pairwise_df.empty else None
        average = pairwise_df["Similarity (%)"].mean() if not pairwise_df.empty else None
        comparisons = len(pairwise_df)

        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown(
                f"""<div class="stat-card">
                    <div class="stat-icon">📂</div>
                    <div class="stat-value">{len(doc_names)}</div>
                    <div class="stat-label">Documents</div>
                </div>""", unsafe_allow_html=True,
            )
        with s2:
            st.markdown(
                f"""<div class="stat-card">
                    <div class="stat-icon">🔀</div>
                    <div class="stat-value">{comparisons}</div>
                    <div class="stat-label">Comparisons</div>
                </div>""", unsafe_allow_html=True,
            )
        with s3:
            val = f"{highest:.1f}%" if highest is not None else "N/A"
            st.markdown(
                f"""<div class="stat-card">
                    <div class="stat-icon">📈</div>
                    <div class="stat-value">{val}</div>
                    <div class="stat-label">Highest Similarity</div>
                </div>""", unsafe_allow_html=True,
            )
        with s4:
            val = f"{average:.1f}%" if average is not None else "N/A"
            st.markdown(
                f"""<div class="stat-card">
                    <div class="stat-icon">📉</div>
                    <div class="stat-value">{val}</div>
                    <div class="stat-label">Avg Similarity</div>
                </div>""", unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Document list ──
        st.markdown('<div class="section-title">📂 Analysed Documents</div>', unsafe_allow_html=True)
        ext_icons = {"pdf": "📕", "docx": "📘", "txt": "📄"}
        for name, text in zip(doc_names, doc_texts):
            ext = name.rsplit(".", 1)[-1].lower() if "." in name else "txt"
            icon = ext_icons.get(ext, "📄")
            words = len(text.split())
            chars = len(text)
            st.markdown(
                f"""<div class="doc-row">
                    <div class="doc-icon">{icon}</div>
                    <div class="doc-name">{name}</div>
                    <div class="doc-meta">{words:,} words · {chars:,} chars</div>
                </div>""",
                unsafe_allow_html=True,
            )

        # ── Bar chart ──
        if not pairwise_df.empty:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-title">📊 Similarity by Document Pair</div>', unsafe_allow_html=True)
            color_map = {
                "Low": "#22c55e", "Moderate": "#f59e0b",
                "High": "#f97316", "Very High": "#ef4444",
            }
            labels = [
                f"{r['Document A'][:12]}… vs {r['Document B'][:12]}…"
                if len(r['Document A']) > 12 or len(r['Document B']) > 12
                else f"{r['Document A']} vs {r['Document B']}"
                for _, r in pairwise_df.iterrows()
            ]
            fig = go.Figure(go.Bar(
                x=labels,
                y=pairwise_df["Similarity (%)"],
                marker_color=[color_map.get(s, "#94a3b8") for s in pairwise_df["Status"]],
                text=[f"{v:.1f}%" for v in pairwise_df["Similarity (%)"]],
                textposition="outside",
                textfont=dict(color=p["text"], size=11),
            ))
            fig.update_layout(**_plotly_layout(p, "", showlegend=False, height=340))
            fig.update_yaxes(range=[0, 115], title_text="Similarity (%)")
            fig.update_xaxes(title_text="Document Pairs")
            st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════════════════════
    # TAB — SIMILARITY
    # ══════════════════════════════════════════════════════════
    with tab_sim:
        if pairwise_df.empty:
            st.info("ℹ️ Similarity analysis requires at least 2 documents.")
        else:
            # Legend
            leg_cols = st.columns(4)
            legend = [
                ("🟢 Low", "0–30%", "badge-low"),
                ("🟡 Moderate", "30–60%", "badge-moderate"),
                ("🟠 High", "60–80%", "badge-high"),
                ("🔴 Very High", "80–100%", "badge-very-high"),
            ]
            for col, (lbl, rng, cls) in zip(leg_cols, legend):
                with col:
                    st.markdown(
                        f'<div style="text-align:center"><span class="{cls}">{lbl}</span>'
                        f'<br><span style="font-size:0.7rem;color:{p["text_muted"]}">{rng}</span></div>',
                        unsafe_allow_html=True,
                    )
            st.markdown("<br>", unsafe_allow_html=True)

            # ── Pairwise table ──
            st.markdown('<div class="section-title">🔢 Pairwise Similarity</div>', unsafe_allow_html=True)
            st.markdown(
                f'<p style="font-size:0.8rem;color:{p["text_muted"]};margin:-0.5rem 0 0.8rem 0;">'
                f'Scores reflect textual overlap only and do not constitute evidence of plagiarism.</p>',
                unsafe_allow_html=True,
            )
            display_df = pairwise_df.copy()
            display_df["Status"] = display_df["Status"].apply(
                lambda s: {"Low": "🟢 Low", "Moderate": "🟡 Moderate",
                            "High": "🟠 High", "Very High": "🔴 Very High"}.get(s, s)
            )
            st.dataframe(
                display_df.style.format({"Similarity (%)": "{:.2f}%"}),
                use_container_width=True,
                hide_index=True,
            )

            # ── Heatmap ──
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-title">🌡️ Similarity Heatmap</div>', unsafe_allow_html=True)
            hm = go.Figure(go.Heatmap(
                z=matrix_df.values,
                x=matrix_df.columns.tolist(),
                y=matrix_df.index.tolist(),
                colorscale=[[0, "#22c55e"], [0.3, "#f59e0b"], [0.6, "#f97316"], [1, "#ef4444"]],
                zmin=0, zmax=100,
                text=np.round(matrix_df.values, 1),
                texttemplate="%{text}%",
                textfont=dict(size=12, color="white"),
                colorbar=dict(
                    title="Similarity (%)",
                    tickfont=dict(color=p["text"]),
                    title_font=dict(color=p["text"]),
                ),
            ))
            hm.update_layout(
                **_plotly_layout(p, "", showlegend=False,
                                  height=max(300, 100 * len(doc_names)))
            )
            st.plotly_chart(hm, use_container_width=True)

            # ── Flagged pairs ──
            flagged = pairwise_df[pairwise_df["Similarity (%)"] >= sim_threshold]
            if not flagged.empty:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    f'<div class="section-title">⚠️ Flagged Pairs (≥ {sim_threshold}%)</div>',
                    unsafe_allow_html=True,
                )
                for _, row in flagged.iterrows():
                    score = int(row["Similarity (%)"])
                    badge = _badge_html(row["Status"])
                    st.markdown(
                        f"""<div class="doc-row" style="margin-bottom:0.4rem;">
                            <div style="flex:1;font-weight:600;font-size:0.9rem;color:{p['text']};">
                                {row['Document A']} <span style="color:{p['text_muted']}">↔</span> {row['Document B']}
                            </div>
                            <div>{badge}</div>
                            <div style="font-size:1rem;font-weight:700;color:{p['danger']};min-width:60px;text-align:right;">
                                {row['Similarity (%)']:.1f}%
                            </div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                    st.progress(score)
                    st.markdown("")

            # ── Sentence matches ──
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-title">🔡 Sentence-Level Matches</div>', unsafe_allow_html=True)
            if not sentence_matches:
                st.success("✅ No highly similar sentence pairs found above the threshold.")
            else:
                st.markdown(
                    f'<p style="font-size:0.82rem;color:{p["text_muted"]};margin:-0.5rem 0 0.8rem 0;">'
                    f'Found <strong>{len(sentence_matches)}</strong> matching pair(s). Showing top 25.</p>',
                    unsafe_allow_html=True,
                )
                for idx, match in enumerate(sentence_matches[:25], 1):
                    score = match["similarity"]
                    badge = _badge_html(get_similarity_label(score / 100))
                    with st.expander(
                        f"Match {idx}  ·  {match['doc_a']} ↔ {match['doc_b']}  ·  {score:.1f}%",
                        expanded=False,
                    ):
                        mc1, mc2 = st.columns(2)
                        with mc1:
                            st.markdown(
                                f'<div class="sent-label">{match["doc_a"]}</div>'
                                f'<div class="sent-box">"{match["sentence_a"]}"</div>',
                                unsafe_allow_html=True,
                            )
                        with mc2:
                            st.markdown(
                                f'<div class="sent-label">{match["doc_b"]}</div>'
                                f'<div class="sent-box">"{match["sentence_b"]}"</div>',
                                unsafe_allow_html=True,
                            )
                        st.markdown(
                            f'<div style="display:flex;align-items:center;gap:10px;margin-top:0.6rem;">'
                            f'{badge}'
                            f'<span style="font-size:0.8rem;color:{p["text_muted"]};">Similarity: <strong>{score:.1f}%</strong></span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

    # ══════════════════════════════════════════════════════════
    # TAB — AI DETECTION
    # ══════════════════════════════════════════════════════════
    with tab_ai:
        detector = get_detector()

        # Model status banner
        if detector.is_ready:
            st.success(f"✅ AI detector model loaded — `models/ai_detector.pkl`")
        else:
            st.warning(detector.get_model_status_message())

        st.markdown(
            f'<div class="disclaimer">{AI_DETECTION_DISCLAIMER.replace("⚠️ **Disclaimer:** ", "")}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        if not ai_results:
            st.info("ℹ️ Enable AI detection in the sidebar and re-analyse.")
        else:
            for res in ai_results:
                label = res.get("label", "Uncertain")
                ai_prob = res.get("ai_prob", 0.0)
                human_prob = res.get("human_prob", 0.0)
                confidence = res.get("confidence", 0.0)
                model_ready = res.get("model_ready", False)

                # Determine label class and icon
                if "AI" in label:
                    label_class = "pred-label-ai"
                    icon = "🤖"
                elif "Human" in label:
                    label_class = "pred-label-human"
                    icon = "✍️"
                else:
                    label_class = "pred-label-uncertain"
                    icon = "❓"

                st.markdown(
                    f"""<div class="pred-card">
                        <div class="pred-filename">📄 {res['name']}</div>
                        <div class="{label_class}">{icon} {label}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

                if model_ready:
                    pc1, pc2, pc3 = st.columns([2, 2, 3])
                    with pc1:
                        st.metric("Confidence", f"{confidence:.1f}%")
                    with pc2:
                        st.metric("AI Probability", f"{ai_prob:.1f}%")
                    with pc3:
                        st.metric("Human Probability", f"{human_prob:.1f}%")

                    # Probability donut + bar
                    bar_fig = go.Figure()
                    bar_fig.add_trace(go.Bar(
                        name="AI-generated",
                        x=["AI-generated"],
                        y=[ai_prob],
                        marker_color=p["danger"],
                        text=[f"{ai_prob:.1f}%"],
                        textposition="outside",
                        textfont=dict(color=p["text"]),
                    ))
                    bar_fig.add_trace(go.Bar(
                        name="Human-written",
                        x=["Human-written"],
                        y=[human_prob],
                        marker_color=p["success"],
                        text=[f"{human_prob:.1f}%"],
                        textposition="outside",
                        textfont=dict(color=p["text"]),
                    ))
                    bar_fig.update_layout(
                        **_plotly_layout(p, f"Probability Breakdown — {res['name']}",
                                         showlegend=False, height=280)
                    )
                    bar_fig.update_yaxes(range=[0, 120])
                    st.plotly_chart(bar_fig, use_container_width=True)
                else:
                    st.warning("Model not available — train the model first (`python train_model.py`).")

                st.markdown("---")

            # ── AI summary chart (if multiple docs) ──
            if len(ai_results) > 1 and all(r.get("model_ready") for r in ai_results):
                st.markdown('<div class="section-title">📊 AI Probability Summary</div>', unsafe_allow_html=True)
                names = [r["name"] for r in ai_results]
                ai_probs = [r["ai_prob"] for r in ai_results]
                hum_probs = [r["human_prob"] for r in ai_results]
                summary_fig = go.Figure()
                summary_fig.add_trace(go.Bar(
                    name="AI-generated", x=names, y=ai_probs,
                    marker_color=p["danger"], opacity=0.85,
                ))
                summary_fig.add_trace(go.Bar(
                    name="Human-written", x=names, y=hum_probs,
                    marker_color=p["success"], opacity=0.85,
                ))
                summary_fig.update_layout(
                    **_plotly_layout(p, "AI vs Human Probability by Document",
                                     showlegend=True, height=340),
                    barmode="group",
                )
                summary_fig.update_yaxes(range=[0, 115], title_text="Probability (%)")
                st.plotly_chart(summary_fig, use_container_width=True)

    # ══════════════════════════════════════════════════════════
    # TAB — REPORT
    # ══════════════════════════════════════════════════════════
    with tab_rep:
        st.markdown('<div class="section-title">📋 Downloadable Report</div>', unsafe_allow_html=True)
        st.markdown(
            f'<p style="font-size:0.82rem;color:{p["text_muted"]};margin:-0.5rem 0 1rem 0;">'
            f'Download a full analysis report in TXT or HTML format.</p>',
            unsafe_allow_html=True,
        )

        txt_report = generate_text_report(doc_names, pairwise_df, sentence_matches, ai_results)
        html_report = generate_html_report(doc_names, pairwise_df, sentence_matches, ai_results)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "⬇️  Download TXT Report",
                data=txt_report.encode("utf-8"),
                file_name=f"report_{ts}.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with dl2:
            st.download_button(
                "⬇️  Download HTML Report",
                data=html_report.encode("utf-8"),
                file_name=f"report_{ts}.html",
                mime="text/html",
                use_container_width=True,
            )

        st.markdown("---")
        st.markdown('<div class="section-title">📄 Report Preview</div>', unsafe_allow_html=True)
        st.code(txt_report, language="text")

# ──────────────────────────────────────────────────────────────────────────────
# WELCOME SCREEN (no results yet)
# ──────────────────────────────────────────────────────────────────────────────
else:
    st.markdown(
        f"""
        <div style="text-align:center;padding:3rem 1rem 1rem 1rem;">
            <div style="font-size:3.5rem;margin-bottom:0.5rem;">🔍</div>
            <h2 style="font-size:1.4rem;font-weight:700;color:{p['text']};margin-bottom:0.5rem;">
                Ready to Analyse
            </h2>
            <p style="color:{p['text_muted']};font-size:0.9rem;max-width:500px;margin:0 auto;">
                Upload documents or paste text in the sidebar, then click
                <strong style="color:{p['accent']};">🔍 Analyse Documents</strong> to get started.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)

    feature_cards = [
        ("📂", "Multi-Format Upload", "Supports TXT, PDF, and DOCX files. Upload several documents at once for batch comparison."),
        ("📊", "Similarity Detection", "TF-IDF + Cosine Similarity measures textual overlap at both document and sentence level."),
        ("🤖", "AI Content Analysis", "A trained ML classifier estimates AI vs. human authorship with a confidence score."),
    ]
    for col, (icon, title, desc) in zip([f1, f2, f3], feature_cards):
        with col:
            st.markdown(
                f"""<div class="section-card" style="text-align:center;height:100%;">
                    <div style="font-size:2rem;margin-bottom:0.6rem;">{icon}</div>
                    <div style="font-weight:700;font-size:0.95rem;color:{p['text']};margin-bottom:0.5rem;">{title}</div>
                    <div style="font-size:0.82rem;color:{p['text_muted']};line-height:1.5;">{desc}</div>
                </div>""",
                unsafe_allow_html=True,
            )


# ──────────────────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    f"""<div class="app-footer">
        AI Content & Plagiarism Detection System &nbsp;·&nbsp; College NLP Mini Project
        &nbsp;·&nbsp; Results are indicative only and should not replace human review.
    </div>""",
    unsafe_allow_html=True,
)
