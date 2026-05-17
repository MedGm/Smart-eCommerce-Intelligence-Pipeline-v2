"""
Smart eCommerce Intelligence Pipeline — BI Dashboard
Multi-page Streamlit app with enforced dark theme, interactive Plotly charts,
and MCP-based data access (responsible architecture).

Dossier tools: Streamlit, Plotly, Altair, Seaborn.
"""

import html
import io
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.mcp.architecture import MCPClient
from src.scoring.topk import compute_score, topk_overall

try:
    import altair as alt

    HAS_ALTAIR = True
except ImportError:
    HAS_ALTAIR = False

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


# ── Design Tokens ─────────────────────────────────────────────
C = {
    "primary": "#D6A85F",
    "secondary": "#B7653C",
    "accent": "#4E9C98",
    "success": "#7FC18A",
    "warning": "#E2B96C",
    "surface": "#0C1117",
    "card": "#121922",
    "border": "rgba(214,168,95,0.14)",
    "text": "#F2EAD9",
    "muted": "#AE9F8A",
    "palette": [
        "#D6A85F",
        "#4E9C98",
        "#B7653C",
        "#7FC18A",
        "#E2B96C",
        "#7E6A55",
        "#B3844E",
        "#78A9A0",
        "#6E8AC9",
        "#D89F70",
    ],
}


# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Smart eCommerce Intelligence",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Custom CSS ────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

/* Night-market editorial theme */
.stApp, [data-testid="stAppViewContainer"],
[data-testid="stHeader"], [data-testid="stToolbar"] {
    font-family: 'IBM Plex Sans', sans-serif;
    color: #F2EAD9;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(214,168,95,0.14), transparent 26%),
        radial-gradient(circle at top right, rgba(78,156,152,0.10), transparent 28%),
        linear-gradient(180deg, #0B1016 0%, #101823 100%);
}

[data-testid="stHeader"] {
    background: rgba(12, 17, 23, 0.85);
    border-bottom: 1px solid rgba(214,168,95,0.08);
}

[data-testid="stToolbar"] {
    display: none;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, rgba(12,17,23,0.98) 0%, rgba(16,24,35,0.98) 100%);
    border-right: 1px solid rgba(214,168,95,0.08);
}
section[data-testid="stSidebar"] .stRadio > div {
    gap: 6px;
}
section[data-testid="stSidebar"] .stRadio label {
    font-size: 13px;
    font-weight: 500;
    padding: 12px 16px;
    border-radius: 14px;
    border: 1px solid transparent;
    transition: all 0.18s ease;
    margin: 0;
    color: #E6DCC9 !important;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(214, 168, 95, 0.08);
    border-color: rgba(214, 168, 95, 0.16);
}
section[data-testid="stSidebar"] .stRadio label[data-checked="true"],
section[data-testid="stSidebar"] input[type="radio"]:checked + div {
    background: rgba(78, 156, 152, 0.12);
    border-color: rgba(78, 156, 152, 0.22);
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: linear-gradient(180deg, rgba(18,25,34,0.96) 0%, rgba(16,22,30,0.96) 100%);
    border-radius: 18px;
    padding: 18px 22px;
    border: 1px solid rgba(214,168,95,0.10);
    box-shadow: 0 18px 40px rgba(0, 0, 0, 0.28);
    transition: transform 0.2s ease, border-color 0.2s ease;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    border-color: rgba(214, 168, 95, 0.22);
}
div[data-testid="stMetric"] label {
    color: #AE9F8A !important;
    font-size: 11.5px !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600 !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-weight: 700 !important;
    font-size: 26px !important;
    color: #F2EAD9 !important;
    font-family: 'Cormorant Garamond', serif !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid rgba(214,168,95,0.08);
    padding-bottom: 0;
}
.stTabs [data-baseweb="tab"] {
    padding: 10px 18px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 600;
    color: #AE9F8A;
    background: rgba(18, 25, 34, 0.82);
    border: 1px solid rgba(214,168,95,0.10);
}
.stTabs [aria-selected="true"] {
    background: rgba(214, 168, 95, 0.12) !important;
    color: #F2EAD9 !important;
    border: 1px solid rgba(214, 168, 95, 0.22) !important;
}

/* Headers */
.page-title {
    font-size: 52px;
    font-weight: 700;
    line-height: 0.95;
    color: #F2EAD9;
    margin-bottom: 6px;
    letter-spacing: -0.8px;
    font-family: 'Cormorant Garamond', serif;
}
.page-subtitle {
    font-size: 15px;
    color: #B7A894;
    margin-bottom: 28px;
    max-width: 780px;
}
.section-header {
    font-size: 12px;
    font-weight: 700;
    color: #AE9F8A;
    text-transform: uppercase;
    letter-spacing: 1.8px;
    margin: 30px 0 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(214,168,95,0.08);
}

/* Dataframes */
.stDataFrame {
    border-radius: 18px;
    overflow: hidden;
    border: 1px solid rgba(214,168,95,0.10);
}

div[data-testid="stPlotlyChart"],
div[data-testid="stDataFrame"],
div[data-testid="stTable"],
div[data-testid="stAltairChart"],
div[data-testid="stPyplot"] {
    background: linear-gradient(180deg, rgba(18,25,34,0.96) 0%, rgba(16,22,30,0.96) 100%);
    border: 1px solid rgba(214,168,95,0.10);
    border-radius: 22px;
    padding: 12px;
    box-shadow: 0 18px 40px rgba(0, 0, 0, 0.20);
}

/* Buttons */
.stButton > button {
    background: #D6A85F !important;
    color: #0C1117 !important;
    border: 1px solid #D6A85F !important;
    border-radius: 999px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 10px 22px !important;
    letter-spacing: 0.3px;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: #E2B96C !important;
    border-color: #E2B96C !important;
    box-shadow: 0 8px 20px rgba(214, 168, 95, 0.18) !important;
}

.stDownloadButton > button {
    background: transparent !important;
    color: #F2EAD9 !important;
    border: 1px solid rgba(214,168,95,0.22) !important;
    border-radius: 999px !important;
}

.stDownloadButton > button:hover {
    background: rgba(78, 156, 152, 0.10) !important;
    border-color: rgba(78, 156, 152, 0.28) !important;
}

/* Hide Streamlit branding */
#MainMenu, footer { visibility: hidden; }

/* Expander */
.streamlit-expanderHeader {
    font-size: 13px;
    font-weight: 600;
    color: #E6DCC9;
}

/* Select / Slider */
.stSelectbox label, .stSlider label, .stMultiSelect label {
    font-size: 12px !important;
    color: #AE9F8A !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}

div[data-baseweb="select"] > div,
.stSlider > div > div,
div[data-testid="stTextInputRootElement"] > div {
    background: rgba(18,25,34,0.94) !important;
    border-radius: 14px !important;
    border-color: rgba(214,168,95,0.12) !important;
}

.editorial-hero {
    position: relative;
    overflow: hidden;
    padding: 28px 30px;
    margin: 0 0 26px;
    border-radius: 28px;
    background:
        linear-gradient(135deg, rgba(18,25,34,0.96) 0%, rgba(19,28,39,0.98) 100%);
    border: 1px solid rgba(214,168,95,0.10);
    box-shadow: 0 20px 46px rgba(0, 0, 0, 0.28);
}

.editorial-hero::after {
    content: "";
    position: absolute;
    inset: auto -80px -80px auto;
    width: 220px;
    height: 220px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(214,168,95,0.18) 0%, rgba(214,168,95,0) 68%);
}

.hero-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    color: #AE9F8A;
    margin-bottom: 12px;
}

.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 46px;
    line-height: 0.95;
    color: #F2EAD9;
    margin: 0 0 10px;
}

.hero-copy {
    max-width: 760px;
    font-size: 15px;
    line-height: 1.65;
    color: #D1C5B6;
    margin-bottom: 18px;
}

.hero-grid {
    display: grid;
    grid-template-columns: 1.2fr 1fr;
    gap: 14px;
}

.spotlight-card, .rank-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(214,168,95,0.08);
    border-radius: 20px;
    padding: 18px 20px;
    backdrop-filter: blur(6px);
}

.spotlight-label, .mono-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    color: #AE9F8A;
    margin-bottom: 10px;
}

.spotlight-title {
    font-size: 24px;
    line-height: 1.05;
    color: #F2EAD9;
    font-family: 'Cormorant Garamond', serif;
    margin-bottom: 10px;
}

.spotlight-meta {
    font-size: 13px;
    color: #D1C5B6;
    line-height: 1.7;
}

.muted-note {
    font-size: 13px;
    color: #AE9F8A;
    margin-top: 6px;
}

.rank-row {
    display: grid;
    grid-template-columns: 44px 1fr auto;
    gap: 12px;
    align-items: center;
    padding: 10px 0;
    border-bottom: 1px dashed rgba(214,168,95,0.10);
}

.rank-row:last-child {
    border-bottom: none;
    padding-bottom: 0;
}

.rank-badge {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #D6A85F;
    color: #0C1117;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
}

.rank-name {
    font-size: 15px;
    font-weight: 600;
    color: #F2EAD9;
}

.rank-sub {
    font-size: 12px;
    color: #AE9F8A;
}

.rank-score {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    color: #D6A85F;
}

.insight-banner {
    padding: 18px 20px;
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(18,25,34,0.98), rgba(22,30,42,0.96));
    border: 1px solid rgba(78,156,152,0.18);
    color: #D1C5B6;
    margin-bottom: 18px;
}

.insight-banner strong {
    color: #F2EAD9;
}

.overview-hero {
    position: relative;
    overflow: hidden;
    padding: 32px;
    border-radius: 28px;
    background:
        radial-gradient(circle at top right, rgba(214,168,95,0.14), transparent 30%),
        linear-gradient(135deg, rgba(18,25,34,0.98) 0%, rgba(14,21,29,0.98) 100%);
    border: 1px solid rgba(214,168,95,0.12);
    box-shadow: 0 20px 48px rgba(0, 0, 0, 0.26);
    margin-bottom: 24px;
}

.overview-kicker {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.8px;
    color: #AE9F8A;
    margin-bottom: 12px;
}

.overview-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 48px;
    line-height: 0.96;
    color: #F2EAD9;
    max-width: 760px;
    margin-bottom: 14px;
}

.overview-copy {
    color: #D1C5B6;
    font-size: 15px;
    line-height: 1.7;
    max-width: 760px;
}

.overview-grid {
    display: grid;
    grid-template-columns: 1.25fr 0.95fr;
    gap: 16px;
    margin-top: 22px;
}

.overview-panel {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(214,168,95,0.10);
    border-radius: 22px;
    padding: 18px 20px;
}

.overview-panel-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 1.6px;
    text-transform: uppercase;
    color: #AE9F8A;
    margin-bottom: 12px;
}

.overview-panel-copy {
    color: #D1C5B6;
    font-size: 14px;
    line-height: 1.7;
}

.overview-mini-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px;
}

.overview-mini-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(214,168,95,0.08);
    border-radius: 18px;
    padding: 14px 16px;
}

.overview-mini-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    color: #8E806E;
    margin-bottom: 8px;
}

.overview-mini-value {
    font-family: 'Cormorant Garamond', serif;
    font-size: 28px;
    line-height: 1;
    color: #F2EAD9;
    margin-bottom: 6px;
}

.overview-mini-copy {
    color: #AE9F8A;
    font-size: 12px;
    line-height: 1.5;
}

.overview-note-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 14px;
    margin: 10px 0 24px;
}

.overview-note-card {
    padding: 18px 20px;
    border-radius: 20px;
    background: linear-gradient(180deg, rgba(18,25,34,0.96), rgba(16,22,30,0.96));
    border: 1px solid rgba(214,168,95,0.10);
}

.overview-note-card strong {
    color: #F2EAD9;
}

.overview-note-card .note-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #8E806E;
    margin-bottom: 8px;
}

.overview-note-card .note-copy {
    font-size: 13px;
    line-height: 1.7;
    color: #D1C5B6;
}

.ranking-shell {
    display: grid;
    grid-template-columns: 1.2fr 0.88fr;
    gap: 16px;
    margin-bottom: 22px;
}

.ranking-card {
    background: linear-gradient(180deg, rgba(18,25,34,0.98), rgba(16,22,30,0.98));
    border: 1px solid rgba(214,168,95,0.10);
    border-radius: 24px;
    padding: 22px 24px;
    box-shadow: 0 18px 42px rgba(0, 0, 0, 0.22);
}

.ranking-kicker {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 1.6px;
    text-transform: uppercase;
    color: #AE9F8A;
    margin-bottom: 12px;
}

.ranking-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 44px;
    line-height: 0.98;
    color: #F2EAD9;
    margin-bottom: 12px;
}

.ranking-copy {
    font-size: 15px;
    line-height: 1.7;
    color: #D1C5B6;
    max-width: 760px;
}

.ranking-summary-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px;
    margin-top: 18px;
}

.ranking-mini {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(214,168,95,0.08);
    border-radius: 18px;
    padding: 14px 16px;
}

.ranking-mini-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    color: #8E806E;
    margin-bottom: 6px;
}

.ranking-mini-value {
    font-family: 'Cormorant Garamond', serif;
    font-size: 28px;
    line-height: 1;
    color: #F2EAD9;
    margin-bottom: 6px;
}

.ranking-mini-copy {
    color: #AE9F8A;
    font-size: 12px;
    line-height: 1.5;
}

.ranking-notes {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 14px;
    margin: 10px 0 24px;
}

.ranking-note {
    padding: 18px 20px;
    border-radius: 20px;
    background: linear-gradient(180deg, rgba(18,25,34,0.96), rgba(16,22,30,0.96));
    border: 1px solid rgba(214,168,95,0.10);
}

.ranking-note-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #8E806E;
    margin-bottom: 8px;
}

.ranking-note-copy {
    color: #D1C5B6;
    font-size: 13px;
    line-height: 1.7;
}

.ranking-note-copy strong {
    color: #F2EAD9;
}

.ranking-shortlist {
    display: grid;
    gap: 12px;
}

.product-spot-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(214,168,95,0.08);
    border-radius: 20px;
    padding: 16px 18px;
}

.product-spot-topline {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
}

.product-spot-rank {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #D6A85F;
    color: #0C1117;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
}

.product-spot-score {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: #D6A85F;
}

.product-spot-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 26px;
    line-height: 1;
    color: #F2EAD9;
    margin-bottom: 8px;
}

.product-spot-copy {
    color: #D1C5B6;
    font-size: 13px;
    line-height: 1.6;
    margin-bottom: 12px;
}

.pill-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.signal-pill {
    display: inline-flex;
    align-items: center;
    padding: 6px 10px;
    border-radius: 999px;
    border: 1px solid rgba(214,168,95,0.14);
    background: rgba(255,255,255,0.03);
    color: #D1C5B6;
    font-size: 11px;
}

.shop-shell {
    display: grid;
    grid-template-columns: 1.15fr 0.9fr;
    gap: 16px;
    margin-bottom: 22px;
}

.shop-card {
    background: linear-gradient(180deg, rgba(18,25,34,0.98), rgba(16,22,30,0.98));
    border: 1px solid rgba(214,168,95,0.10);
    border-radius: 24px;
    padding: 22px 24px;
    box-shadow: 0 18px 42px rgba(0, 0, 0, 0.22);
}

.shop-kicker {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 1.6px;
    text-transform: uppercase;
    color: #AE9F8A;
    margin-bottom: 12px;
}

.shop-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 42px;
    line-height: 0.98;
    color: #F2EAD9;
    margin-bottom: 12px;
}

.shop-copy {
    font-size: 15px;
    line-height: 1.7;
    color: #D1C5B6;
}

.shop-mini-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px;
    margin-top: 18px;
}

.shop-mini {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(214,168,95,0.08);
    border-radius: 18px;
    padding: 14px 16px;
}

.shop-mini-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    color: #8E806E;
    margin-bottom: 6px;
}

.shop-mini-value {
    font-family: 'Cormorant Garamond', serif;
    font-size: 28px;
    line-height: 1;
    color: #F2EAD9;
    margin-bottom: 6px;
}

.shop-mini-copy {
    color: #AE9F8A;
    font-size: 12px;
    line-height: 1.5;
}

.shop-note-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 14px;
    margin: 10px 0 24px;
}

.shop-note {
    padding: 18px 20px;
    border-radius: 20px;
    background: linear-gradient(180deg, rgba(18,25,34,0.96), rgba(16,22,30,0.96));
    border: 1px solid rgba(214,168,95,0.10);
}

.shop-note-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #8E806E;
    margin-bottom: 8px;
}

.shop-note-copy {
    color: #D1C5B6;
    font-size: 13px;
    line-height: 1.7;
}

.shop-note-copy strong {
    color: #F2EAD9;
}

.shop-dossier-list {
    display: grid;
    gap: 12px;
}

.shop-dossier {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(214,168,95,0.08);
    border-radius: 20px;
    padding: 16px 18px;
}

.shop-dossier-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
}

.shop-dossier-name {
    font-family: 'Cormorant Garamond', serif;
    font-size: 28px;
    line-height: 1;
    color: #F2EAD9;
}

.shop-dossier-score {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: #D6A85F;
}

.shop-dossier-copy {
    color: #D1C5B6;
    font-size: 13px;
    line-height: 1.6;
}

.model-shell {
    display: grid;
    grid-template-columns: 1.15fr 0.9fr;
    gap: 16px;
    margin-bottom: 22px;
}

.model-card {
    background: linear-gradient(180deg, rgba(18,25,34,0.98), rgba(16,22,30,0.98));
    border: 1px solid rgba(214,168,95,0.10);
    border-radius: 24px;
    padding: 22px 24px;
    box-shadow: 0 18px 42px rgba(0, 0, 0, 0.22);
}

.model-kicker {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 1.6px;
    text-transform: uppercase;
    color: #AE9F8A;
    margin-bottom: 12px;
}

.model-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 42px;
    line-height: 0.98;
    color: #F2EAD9;
    margin-bottom: 12px;
}

.model-copy {
    font-size: 15px;
    line-height: 1.7;
    color: #D1C5B6;
}

.model-mini-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px;
    margin-top: 18px;
}

.model-mini {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(214,168,95,0.08);
    border-radius: 18px;
    padding: 14px 16px;
}

.model-mini-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    color: #8E806E;
    margin-bottom: 6px;
}

.model-mini-value {
    font-family: 'Cormorant Garamond', serif;
    font-size: 28px;
    line-height: 1;
    color: #F2EAD9;
    margin-bottom: 6px;
}

.model-mini-copy {
    color: #AE9F8A;
    font-size: 12px;
    line-height: 1.5;
}

.model-note-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 14px;
    margin: 10px 0 24px;
}

.model-note {
    padding: 18px 20px;
    border-radius: 20px;
    background: linear-gradient(180deg, rgba(18,25,34,0.96), rgba(16,22,30,0.96));
    border: 1px solid rgba(214,168,95,0.10);
}

.model-note-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #8E806E;
    margin-bottom: 8px;
}

.model-note-copy {
    color: #D1C5B6;
    font-size: 13px;
    line-height: 1.7;
}

.model-note-copy strong {
    color: #F2EAD9;
}

.model-driver-list {
    display: grid;
    gap: 10px;
}

.model-driver {
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 12px;
    align-items: center;
    padding: 12px 14px;
    border-radius: 16px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(214,168,95,0.08);
}

.model-driver-name {
    color: #F2EAD9;
    font-size: 14px;
    font-weight: 600;
}

.model-driver-value {
    color: #D6A85F;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
}

.segment-shell {
    display: grid;
    grid-template-columns: 1.15fr 0.9fr;
    gap: 16px;
    margin-bottom: 22px;
}

.segment-card {
    background: linear-gradient(180deg, rgba(18,25,34,0.98), rgba(16,22,30,0.98));
    border: 1px solid rgba(214,168,95,0.10);
    border-radius: 24px;
    padding: 22px 24px;
    box-shadow: 0 18px 42px rgba(0, 0, 0, 0.22);
}

.segment-kicker {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 1.6px;
    text-transform: uppercase;
    color: #AE9F8A;
    margin-bottom: 12px;
}

.segment-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 42px;
    line-height: 0.98;
    color: #F2EAD9;
    margin-bottom: 12px;
}

.segment-copy {
    font-size: 15px;
    line-height: 1.7;
    color: #D1C5B6;
}

.segment-mini-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px;
    margin-top: 18px;
}

.segment-mini {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(214,168,95,0.08);
    border-radius: 18px;
    padding: 14px 16px;
}

.segment-mini-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    color: #8E806E;
    margin-bottom: 6px;
}

.segment-mini-value {
    font-family: 'Cormorant Garamond', serif;
    font-size: 28px;
    line-height: 1;
    color: #F2EAD9;
    margin-bottom: 6px;
}

.segment-mini-copy {
    color: #AE9F8A;
    font-size: 12px;
    line-height: 1.5;
}

.segment-note-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 14px;
    margin: 10px 0 24px;
}

.segment-note {
    padding: 18px 20px;
    border-radius: 20px;
    background: linear-gradient(180deg, rgba(18,25,34,0.96), rgba(16,22,30,0.96));
    border: 1px solid rgba(214,168,95,0.10);
}

.segment-note-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #8E806E;
    margin-bottom: 8px;
}

.segment-note-copy {
    color: #D1C5B6;
    font-size: 13px;
    line-height: 1.7;
}

.segment-note-copy strong {
    color: #F2EAD9;
}

.segment-driver-list {
    display: grid;
    gap: 10px;
}

.segment-driver {
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 12px;
    align-items: center;
    padding: 12px 14px;
    border-radius: 16px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(214,168,95,0.08);
}

.segment-driver-name {
    color: #F2EAD9;
    font-size: 14px;
    font-weight: 600;
}

.segment-driver-value {
    color: #D6A85F;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
}

.rules-shell {
    display: grid;
    grid-template-columns: 1.15fr 0.9fr;
    gap: 16px;
    margin-bottom: 22px;
}

.rules-card {
    background: linear-gradient(180deg, rgba(18,25,34,0.98), rgba(16,22,30,0.98));
    border: 1px solid rgba(214,168,95,0.10);
    border-radius: 24px;
    padding: 22px 24px;
    box-shadow: 0 18px 42px rgba(0, 0, 0, 0.22);
}

.rules-kicker {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 1.6px;
    text-transform: uppercase;
    color: #AE9F8A;
    margin-bottom: 12px;
}

.rules-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 42px;
    line-height: 0.98;
    color: #F2EAD9;
    margin-bottom: 12px;
}

.rules-copy {
    font-size: 15px;
    line-height: 1.7;
    color: #D1C5B6;
}

.rules-mini-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px;
    margin-top: 18px;
}

.rules-mini {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(214,168,95,0.08);
    border-radius: 18px;
    padding: 14px 16px;
}

.rules-mini-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    color: #8E806E;
    margin-bottom: 6px;
}

.rules-mini-value {
    font-family: 'Cormorant Garamond', serif;
    font-size: 28px;
    line-height: 1;
    color: #F2EAD9;
    margin-bottom: 6px;
}

.rules-mini-copy {
    color: #AE9F8A;
    font-size: 12px;
    line-height: 1.5;
}

.rules-note-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 14px;
    margin: 10px 0 24px;
}

.rules-note {
    padding: 18px 20px;
    border-radius: 20px;
    background: linear-gradient(180deg, rgba(18,25,34,0.96), rgba(16,22,30,0.96));
    border: 1px solid rgba(214,168,95,0.10);
}

.rules-note-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #8E806E;
    margin-bottom: 8px;
}

.rules-note-copy {
    color: #D1C5B6;
    font-size: 13px;
    line-height: 1.7;
}

.rules-note-copy strong {
    color: #F2EAD9;
}

.rules-highlight-list {
    display: grid;
    gap: 10px;
}

.rules-highlight {
    padding: 12px 14px;
    border-radius: 16px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(214,168,95,0.08);
}

.rules-highlight-main {
    color: #F2EAD9;
    font-size: 13px;
    line-height: 1.6;
    margin-bottom: 6px;
}

.rules-highlight-meta {
    color: #D6A85F;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
}

.llm-shell {
    display: grid;
    grid-template-columns: 1.15fr 0.9fr;
    gap: 16px;
    margin-bottom: 22px;
}

.llm-card {
    background: linear-gradient(180deg, rgba(18,25,34,0.98), rgba(16,22,30,0.98));
    border: 1px solid rgba(214,168,95,0.10);
    border-radius: 24px;
    padding: 22px 24px;
    box-shadow: 0 18px 42px rgba(0, 0, 0, 0.22);
}

.llm-kicker {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 1.6px;
    text-transform: uppercase;
    color: #AE9F8A;
    margin-bottom: 12px;
}

.llm-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 42px;
    line-height: 0.98;
    color: #F2EAD9;
    margin-bottom: 12px;
}

.llm-copy {
    font-size: 15px;
    line-height: 1.7;
    color: #D1C5B6;
}

.llm-mini-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px;
    margin-top: 18px;
}

.llm-mini {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(214,168,95,0.08);
    border-radius: 18px;
    padding: 14px 16px;
}

.llm-mini-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    color: #8E806E;
    margin-bottom: 6px;
}

.llm-mini-value {
    font-family: 'Cormorant Garamond', serif;
    font-size: 28px;
    line-height: 1;
    color: #F2EAD9;
    margin-bottom: 6px;
}

.llm-mini-copy {
    color: #AE9F8A;
    font-size: 12px;
    line-height: 1.5;
}

.llm-note-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 14px;
    margin: 10px 0 24px;
}

.llm-note {
    padding: 18px 20px;
    border-radius: 20px;
    background: linear-gradient(180deg, rgba(18,25,34,0.96), rgba(16,22,30,0.96));
    border: 1px solid rgba(214,168,95,0.10);
}

.llm-note-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #8E806E;
    margin-bottom: 8px;
}

.llm-note-copy {
    color: #D1C5B6;
    font-size: 13px;
    line-height: 1.7;
}

.llm-note-copy strong {
    color: #F2EAD9;
}

.llm-action-list {
    display: grid;
    gap: 10px;
}

.llm-action {
    padding: 12px 14px;
    border-radius: 16px;
    background: rgba(214,168,95,0.88);
    border: 1px solid rgba(214,168,95,0.42);
}

.llm-action-main {
    color: #0C1117;
    font-size: 13px;
    font-weight: 600;
    line-height: 1.6;
    margin-bottom: 6px;
}

.llm-action-meta {
    color: #1F2937;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
}

.st-key-llm_exec_summary button,
.st-key-llm_strategy button,
.st-key-llm_profile button,
.st-key-llm_exec_summary button p,
.st-key-llm_strategy button p,
.st-key-llm_profile button p,
.st-key-llm_exec_summary button span,
.st-key-llm_strategy button span,
.st-key-llm_profile button span {
    color: #0C1117 !important;
}

@media (max-width: 980px) {
    .rules-shell,
    .rules-note-grid {
        grid-template-columns: 1fr;
    }

    .rules-title {
        font-size: 36px;
    }
}

@media (max-width: 980px) {
    .llm-shell,
    .llm-note-grid {
        grid-template-columns: 1fr;
    }

    .llm-title {
        font-size: 36px;
    }
}

@media (max-width: 980px) {
    .segment-shell,
    .segment-note-grid {
        grid-template-columns: 1fr;
    }

    .segment-title {
        font-size: 36px;
    }
}

@media (max-width: 980px) {
    .model-shell,
    .model-note-grid {
        grid-template-columns: 1fr;
    }

    .model-title {
        font-size: 36px;
    }
}

@media (max-width: 980px) {
    .shop-shell,
    .shop-note-grid {
        grid-template-columns: 1fr;
    }

    .shop-title {
        font-size: 36px;
    }
}

@media (max-width: 980px) {
    .ranking-shell,
    .ranking-notes {
        grid-template-columns: 1fr;
    }

    .ranking-title {
        font-size: 38px;
    }
}

@media (max-width: 980px) {
    .overview-grid,
    .overview-note-grid {
        grid-template-columns: 1fr;
    }

    .overview-title {
        font-size: 40px;
    }
}

div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li,
div[data-testid="stMarkdownContainer"] label {
    color: #D1C5B6;
}

.stCaption {
    color: #AE9F8A !important;
}

.stChatMessage {
    background: rgba(18,25,34,0.88);
    border: 1px solid rgba(214,168,95,0.08);
    border-radius: 18px;
}

@media (max-width: 980px) {
    .hero-grid {
        grid-template-columns: 1fr;
    }

    .hero-title {
        font-size: 38px;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


# ── MCP Client ────────────────────────────────────────────────
mcp = MCPClient()


@st.cache_data(ttl=60)
def load_csv(name: str) -> pd.DataFrame:
    content = mcp.get_analytics(name)
    if content is None:
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(content))


@st.cache_data(ttl=60)
def load_json(name: str) -> dict:
    content = mcp.get_analytics(name)
    if content is None:
        return {}
    return json.loads(content)


@st.cache_data(ttl=60)
def load_features() -> pd.DataFrame:
    from src.config import processed_dir

    p = processed_dir() / "features.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@st.cache_data(ttl=60)
def build_topk_view(mode: str, max_per_shop_ratio: float, k_overall: int = 50) -> pd.DataFrame:
    """Build dashboard ranking view from live features (strict or diversified)."""
    df = load_features()
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    if "score" not in df.columns:
        df["score"] = compute_score(df)

    if mode == "Strict (score-only)":
        topk = df.nlargest(k_overall, "score")
    else:
        topk = topk_overall(df, k=k_overall, max_per_shop_ratio=max_per_shop_ratio)

    return topk.sort_values("score", ascending=False).reset_index(drop=True)


@st.cache_data(ttl=60)
def load_processed_json(name: str) -> dict:
    from src.config import processed_dir

    path = processed_dir() / name
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


@st.cache_data(ttl=300)
def get_llm_summary() -> str:
    try:
        from src.llm.summarizer import run as llm_run

        return llm_run()
    except Exception as e:
        return f"LLM summary unavailable: {e}"


def apply_theme(fig, title="", height=400):
    """Apply consistent editorial styling to Plotly figures."""
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, color=C["text"], family="Cormorant Garamond"),
        ),
        height=height,
        plot_bgcolor=C["card"],
        paper_bgcolor=C["card"],
        font=dict(color=C["text"], family="IBM Plex Sans, sans-serif", size=12),
        xaxis=dict(
            gridcolor="rgba(214,168,95,0.08)",
            zerolinecolor="rgba(214,168,95,0.12)",
            tickfont=dict(color=C["muted"]),
            title_font=dict(color=C["muted"]),
        ),
        yaxis=dict(
            gridcolor="rgba(214,168,95,0.08)",
            zerolinecolor="rgba(214,168,95,0.12)",
            tickfont=dict(color=C["muted"]),
            title_font=dict(color=C["muted"]),
        ),
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def compact_number(value):
    if pd.isna(value):
        return "n/a"
    value = float(value)
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return f"{value:.0f}"


def format_currency(value):
    if pd.isna(value):
        return "Price unavailable"
    return f"${float(value):,.2f}"


def safe_text(value, fallback="Unknown"):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return fallback
    text = str(value).strip()
    return text if text else fallback


def short_text(value, limit=120):
    text = safe_text(value, "No description available")
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def format_percent(value):
    if pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def parse_jsonl_loose(raw_text: str) -> list[dict]:
    """Parse JSONL defensively, including lines with concatenated JSON objects."""
    if not raw_text:
        return []
    records: list[dict] = []
    decoder = json.JSONDecoder()

    for line in raw_text.splitlines():
        text = line.strip()
        if not text:
            continue

        # Common corruption case: multiple objects stuck on one line.
        text = text.replace("}{", "}\n{")
        for chunk in text.splitlines():
            chunk = chunk.strip()
            if not chunk:
                continue

            # Try full-line parse first.
            try:
                obj = json.loads(chunk)
                if isinstance(obj, dict):
                    records.append(obj)
                continue
            except json.JSONDecodeError:
                pass

            # Fallback: scan for embedded JSON objects inside noisy text.
            idx = 0
            while idx < len(chunk):
                brace = chunk.find("{", idx)
                if brace < 0:
                    break
                try:
                    obj, end = decoder.raw_decode(chunk, brace)
                    if isinstance(obj, dict):
                        records.append(obj)
                    idx = end
                except json.JSONDecodeError:
                    idx = brace + 1

    return records


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
    <div style="padding: 24px 0 16px; text-align: center;">
        <div style="font-size: 13px; font-weight: 700; letter-spacing: 2px;
             text-transform: uppercase; color: #D6A85F;">Smart eCommerce</div>
        <div style="font-size: 11px; color: #AE9F8A; margin-top: 2px;">
            Intelligence Pipeline</div>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Product Rankings",
            "Shop Analysis",
            "ML Models",
            "Segmentation",
            "Association Rules",
            "LLM Insights",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        f"""<div style="text-align:center; padding: 8px 0;">
        <div style="font-size: 10px; color: #8E806E; text-transform: uppercase;
             letter-spacing: 1px; margin-bottom: 6px;">Architecture</div>
        <div style="font-size: 11px; color: #AE9F8A;">MCP Host / Client / Server</div>
        <div style="font-size: 11px; color: #8E806E; margin-top: 2px;">
            {len(mcp.list_analytics())} analytics files loaded</div>
        </div>""",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown('<div class="page-title">Dashboard Overview</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">A high-contrast control room for the current catalog, spanning validated analytics outputs, shop mix, pricing spread, and product quality signals.</div>',
        unsafe_allow_html=True,
    )

    df = load_features()
    if df.empty:
        st.warning("No data available. Run the pipeline first.")
        st.stop()

    product_count = len(df)
    shop_count = int(df["shop_name"].nunique()) if "shop_name" in df.columns else 0
    category_count = int(df["category"].nunique()) if "category" in df.columns else 0
    avg_price = (
        float(df["price"].dropna().mean())
        if "price" in df.columns and df["price"].notna().any()
        else np.nan
    )
    avg_rating = (
        float(df["rating"].dropna().mean())
        if "rating" in df.columns and df["rating"].notna().any()
        else np.nan
    )
    price_coverage = float(df["price"].notna().mean()) if "price" in df.columns else 0.0
    rating_coverage = float(df["rating"].fillna(0).gt(0).mean()) if "rating" in df.columns else 0.0
    in_stock_rate = (
        float(df["is_in_stock"].fillna(False).mean()) if "is_in_stock" in df.columns else 0.0
    )
    discounted_rate = (
        float(df["discount_pct"].fillna(0).gt(0).mean()) if "discount_pct" in df.columns else 0.0
    )
    top_shop = (
        safe_text(df["shop_name"].value_counts().idxmax())
        if "shop_name" in df.columns
        else "Unknown"
    )
    top_category = (
        safe_text(df["category"].fillna("uncategorized").value_counts().idxmax())
        if "category" in df.columns
        else "uncategorized"
    )
    platform_mix = (
        df["source_platform"].value_counts(normalize=True).round(3).to_dict()
        if "source_platform" in df.columns
        else {}
    )
    lead_platform = max(platform_mix, key=platform_mix.get) if platform_mix else "Unknown"
    review_mass = int(df["review_count"].fillna(0).sum()) if "review_count" in df.columns else 0
    dq = load_processed_json("dq_counters.json")
    run_meta = load_processed_json("run_metadata.json")
    audit_delta = load_json("category_audit_before_after_delta.json")

    st.markdown(
        f"""
        <div class="overview-hero">
            <div class="overview-kicker">Overview / validated pipeline state</div>
            <div class="overview-title">A live command deck for catalog health, commercial coverage, and model-ready product signals.</div>
            <div class="overview-copy">
                The pipeline currently exposes <strong>{product_count}</strong> products across <strong>{shop_count}</strong> shops and <strong>{category_count}</strong> categories.
                This page is meant to answer one question immediately: <strong>is the current catalog rich enough, clean enough, and balanced enough to trust the downstream rankings and ML outputs?</strong>
            </div>
            <div class="overview-grid">
                <div class="overview-panel">
                    <div class="overview-panel-title">Market reading</div>
                    <div class="overview-panel-copy">
                        The catalog is currently led by <strong>{html.escape(top_shop)}</strong>, with <strong>{html.escape(top_category)}</strong>
                        as the largest visible category. Platform mix is currently dominated by <strong>{html.escape(lead_platform)}</strong>,
                        and the dataset carries <strong>{compact_number(review_mass)}</strong> review signals into downstream scoring.
                    </div>
                </div>
                <div class="overview-panel">
                    <div class="overview-panel-title">Coverage diagnostics</div>
                    <div class="overview-mini-grid">
                        <div class="overview-mini-card">
                            <div class="overview-mini-label">Price coverage</div>
                            <div class="overview-mini-value">{format_percent(price_coverage)}</div>
                            <div class="overview-mini-copy">Rows with usable price values for scoring and price analytics.</div>
                        </div>
                        <div class="overview-mini-card">
                            <div class="overview-mini-label">Rating coverage</div>
                            <div class="overview-mini-value">{format_percent(rating_coverage)}</div>
                            <div class="overview-mini-copy">Rows with non-zero ratings available for demand quality signals.</div>
                        </div>
                        <div class="overview-mini-card">
                            <div class="overview-mini-label">In-stock signal</div>
                            <div class="overview-mini-value">{format_percent(in_stock_rate)}</div>
                            <div class="overview-mini-copy">Products currently marked as available in the feature dataset.</div>
                        </div>
                        <div class="overview-mini-card">
                            <div class="overview-mini-label">Discounted rows</div>
                            <div class="overview-mini-value">{format_percent(discounted_rate)}</div>
                            <div class="overview-mini-copy">Products carrying a measurable discount signal right now.</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # KPI row
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Products", f"{product_count:,}")
    k2.metric("Shops", shop_count)
    k3.metric("Avg Price", f"${avg_price:.2f}" if pd.notna(avg_price) else "—")
    k4.metric("Avg Rating", f"{avg_rating:.1f}" if pd.notna(avg_rating) else "—")
    k5.metric("Categories", category_count)

    st.markdown(
        f"""
        <div class="overview-note-grid">
            <div class="overview-note-card">
                <div class="note-title">Catalog balance</div>
                <div class="note-copy">The current shop leader is <strong>{html.escape(top_shop)}</strong>. Large shop skew is useful to track here because it can dominate downstream rankings if left unexamined.</div>
            </div>
            <div class="overview-note-card">
                <div class="note-title">Metadata strength</div>
                <div class="note-copy">Price coverage sits at <strong>{format_percent(price_coverage)}</strong> while rating coverage sits at <strong>{format_percent(rating_coverage)}</strong>. These are the fastest indicators of dataset usefulness.</div>
            </div>
            <div class="overview-note-card">
                <div class="note-title">Commercial pulse</div>
                <div class="note-copy">A total of <strong>{compact_number(review_mass)}</strong> reviews are available in the current feature set, providing the strongest behavioral signal in the scoring pipeline.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Transparent data-quality + audit representation (no mock values)
    st.markdown(
        '<div class="section-header">Data Quality and Audit Truth</div>', unsafe_allow_html=True
    )

    if dq:
        rows_total = int(dq.get("rows_total", 0))
        category_found = int(dq.get("category_found", 0))
        category_missing = int(dq.get("category_missing", 0))
        category_found_rate = (category_found / rows_total) if rows_total else np.nan
        evidence_high = int(dq.get("category_evidence_strength_high", 0))

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Rows (validated)", f"{rows_total:,}")
        d2.metric("Category Found", f"{category_found:,}")
        d3.metric("Category Missing", f"{category_missing:,}")
        d4.metric(
            "Found Rate",
            f"{category_found_rate * 100:.1f}%" if pd.notna(category_found_rate) else "—",
        )

        with st.expander("Category Evidence Distribution", expanded=False):
            evidence_rows = []
            for key, value in dq.items():
                if key.startswith("category_evidence_strength_") and isinstance(
                    value, (int, float)
                ):
                    evidence_rows.append(
                        {
                            "Evidence": key.replace("category_evidence_strength_", ""),
                            "Count": int(value),
                        }
                    )
            if evidence_rows:
                ev_df = pd.DataFrame(evidence_rows).sort_values("Count", ascending=False)
                fig = px.bar(
                    ev_df,
                    x="Evidence",
                    y="Count",
                    color="Evidence",
                    color_discrete_sequence=C["palette"],
                )
                fig.update_layout(showlegend=False, xaxis_title="Evidence strength")
                fig = apply_theme(fig, "Category evidence strength", 320)
                st.plotly_chart(fig, width="stretch")
            st.caption(
                f"High-evidence rows: {evidence_high:,} / {rows_total:,}"
                if rows_total
                else "Evidence counters unavailable"
            )
    else:
        st.info(
            "`dq_counters.json` not found in processed outputs; run preprocessing to populate DQ metrics."
        )

    if audit_delta:
        after_counts = audit_delta.get("status_counts_on_same_20_rows", {}).get("after", {}) or {}
        non_found_after = (
            audit_delta.get("active_root_cause_counts_non_found", {}).get("after", {}) or {}
        )
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Audit Rows", int(audit_delta.get("rows", 0)))
        a2.metric("Audit Found", int(after_counts.get("found", 0)))
        a3.metric("Audit Missing", int(after_counts.get("missing", 0)))
        a4.metric(
            "Shopify extraction_failed",
            int((audit_delta.get("shopify_extraction_failed_count") or {}).get("after", 0)),
        )

        if non_found_after:
            root_df = pd.DataFrame(
                [{"Root Cause": key, "Count": int(value)} for key, value in non_found_after.items()]
            ).sort_values("Count", ascending=False)
            fig = px.bar(
                root_df,
                x="Count",
                y="Root Cause",
                orientation="h",
                color="Count",
                color_continuous_scale=[[0, C["accent"]], [1, C["primary"]]],
            )
            fig.update_layout(coloraxis_showscale=False)
            fig = apply_theme(fig, "Active non-found root causes (20-row audit)", 320)
            st.plotly_chart(fig, width="stretch")
        with st.expander("Audit delta payload", expanded=False):
            st.json(audit_delta)
    else:
        st.info("`category_audit_before_after_delta.json` not found in analytics outputs.")

    if run_meta:
        st.caption(
            "Run metadata — "
            f"schema {run_meta.get('schema_version', 'n/a')}, "
            f"extraction {run_meta.get('extraction_version', 'n/a')}, "
            f"timestamp {run_meta.get('run_ts_utc', 'n/a')}, "
            f"rows_output {run_meta.get('rows_output', 'n/a')}"
        )

    st.markdown('<div class="section-header">Market structure</div>', unsafe_allow_html=True)

    # Row 1: Platform + Shop
    r1a, r1b = st.columns(2)
    with r1a:
        if "source_platform" in df.columns:
            pf = df["source_platform"].value_counts().reset_index()
            pf.columns = ["Platform", "Products"]
            fig = px.pie(
                pf,
                values="Products",
                names="Platform",
                hole=0.6,
                color_discrete_sequence=[C["primary"], C["accent"]],
            )
            fig.update_traces(textposition="inside", textinfo="percent+label", textfont_size=12)
            fig = apply_theme(fig, "Platform Distribution", 340)
            st.plotly_chart(fig, width="stretch")

    with r1b:
        if "shop_name" in df.columns:
            sc = df["shop_name"].value_counts().reset_index()
            sc.columns = ["Shop", "Count"]
            fig = px.bar(
                sc,
                x="Count",
                y="Shop",
                orientation="h",
                color="Count",
                color_continuous_scale=[[0, C["accent"]], [1, C["primary"]]],
            )
            fig.update_layout(showlegend=False, coloraxis_showscale=False)
            fig = apply_theme(fig, "Products per Shop", 340)
            st.plotly_chart(fig, width="stretch")

    # Row 2: Categories + Price
    st.markdown(
        '<div class="section-header">Category and pricing anatomy</div>', unsafe_allow_html=True
    )
    r2a, r2b = st.columns(2)
    with r2a:
        if "category" in df.columns:
            cc = df["category"].value_counts().head(12).reset_index()
            cc.columns = ["Category", "Count"]
            if HAS_ALTAIR:
                chart = (
                    alt.Chart(cc)
                    .mark_bar(cornerRadiusEnd=4)
                    .encode(
                        x=alt.X("Count:Q", title="Products"),
                        y=alt.Y("Category:N", sort="-x", title=None),
                        color=alt.Color(
                            "Count:Q",
                            scale=alt.Scale(range=[C["primary"], C["accent"]]),
                            legend=None,
                        ),
                    )
                    .properties(title="Top Categories", height=340)
                    .configure_view(strokeWidth=0)
                    .configure(background=C["card"])
                    .configure_axis(
                        gridColor="rgba(214,168,95,0.08)",
                        labelColor=C["muted"],
                        titleColor=C["muted"],
                    )
                    .configure_title(color=C["text"], fontSize=14)
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                fig = px.bar(
                    cc,
                    x="Count",
                    y="Category",
                    orientation="h",
                    color="Count",
                    color_continuous_scale=[[0, C["primary"]], [1, C["success"]]],
                )
                fig.update_layout(coloraxis_showscale=False)
                fig = apply_theme(fig, "Top Categories", 340)
                st.plotly_chart(fig, width="stretch")

    with r2b:
        if "price" in df.columns:
            prices = df["price"].dropna()
            if not prices.empty:
                fig = go.Figure()
                fig.add_trace(
                    go.Histogram(
                        x=prices,
                        nbinsx=30,
                        marker=dict(color=C["primary"], line=dict(width=0.5, color=C["surface"])),
                        opacity=0.9,
                        name="Count",
                    )
                )
                fig = apply_theme(fig, "Price Distribution", 340)
                fig.update_layout(xaxis_title="Price ($)", yaxis_title="Products", bargap=0.05)
                st.plotly_chart(fig, width="stretch")

    # Row 3: Rating + Discount
    st.markdown(
        '<div class="section-header">Quality and monetization signals</div>', unsafe_allow_html=True
    )
    r3a, r3b = st.columns(2)
    with r3a:
        if "rating" in df.columns:
            rated = df["rating"].dropna()
            rated = rated[rated > 0]
            if not rated.empty:
                fig = go.Figure()
                fig.add_trace(
                    go.Histogram(
                        x=rated,
                        nbinsx=20,
                        marker=dict(color=C["warning"], line=dict(width=0.5, color=C["surface"])),
                    )
                )
                fig = apply_theme(fig, f"Rating Distribution ({len(rated)} rated)", 300)
                fig.update_layout(xaxis_title="Rating", yaxis_title="Products")
                st.plotly_chart(fig, width="stretch")

    with r3b:
        if "discount_pct" in df.columns:
            disc = df["discount_pct"].dropna()
            disc = disc[disc > 0]
            if not disc.empty:
                fig = go.Figure()
                fig.add_trace(
                    go.Histogram(
                        x=disc,
                        nbinsx=20,
                        marker=dict(color=C["secondary"], line=dict(width=0.5, color=C["surface"])),
                    )
                )
                fig = apply_theme(fig, f"Discount Distribution ({len(disc)} discounted)", 300)
                fig.update_layout(xaxis_title="Discount %", yaxis_title="Products")
                st.plotly_chart(fig, width="stretch")

    # Correlation heatmap
    if HAS_SEABORN:
        st.markdown(
            '<div class="section-header">Feature interaction map</div>', unsafe_allow_html=True
        )
        with st.expander("Feature Correlation Matrix"):
            num = df.select_dtypes(include=[np.number]).columns.tolist()
            keep = [c for c in num if c not in ["shop_product_count", "category_frequency"]][:12]
            if len(keep) > 3:
                corr = df[keep].corr()
                fig_s, ax = plt.subplots(figsize=(10, 5.5))
                sns.heatmap(
                    corr,
                    annot=True,
                    fmt=".2f",
                    cmap="RdBu_r",
                    ax=ax,
                    square=True,
                    linewidths=0.5,
                    cbar_kws={"shrink": 0.75},
                    annot_kws={"size": 9},
                )
                ax.set_title("Feature Correlation Matrix", fontsize=13, pad=12, color=C["text"])
                ax.tick_params(colors=C["muted"], labelsize=9)
                fig_s.patch.set_facecolor(C["card"])
                ax.set_facecolor(C["card"])
                plt.tight_layout()
                st.pyplot(fig_s)
                plt.close(fig_s)


# ══════════════════════════════════════════════════════════════
# PRODUCT RANKINGS
# ══════════════════════════════════════════════════════════════
elif page == "Product Rankings":
    st.markdown('<div class="page-title">Market Radar</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">A deeper buying board for the strongest products in the current catalog, balancing demand signals, review gravity, availability, pricing context, and discount momentum.</div>',
        unsafe_allow_html=True,
    )

    mode_cols = st.columns([1.2, 1.2, 2.6])
    with mode_cols[0]:
        ranking_mode = st.selectbox(
            "Ranking mode",
            ["Diversified (balanced shops)", "Strict (score-only)"],
            index=0,
        )
    with mode_cols[1]:
        if ranking_mode == "Diversified (balanced shops)":
            shop_cap_ratio = st.slider("Max share / shop", 0.2, 0.8, 0.4, 0.05)
        else:
            shop_cap_ratio = 1.0

    topk = build_topk_view(ranking_mode, shop_cap_ratio, k_overall=50)
    if topk.empty:
        st.warning("No ranking data available. Run preprocess/features first.")
        st.stop()

    if "score" in topk.columns:
        topk = topk.sort_values("score", ascending=False).reset_index(drop=True)

    st.markdown('<div class="section-header">Ranking workspace</div>', unsafe_allow_html=True)
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        cats = ["All"] + (
            sorted(topk["category"].dropna().unique().tolist())
            if "category" in topk.columns
            else []
        )
        sel_cat = st.selectbox("Category", cats)
    with fc2:
        shops = ["All"] + (
            sorted(topk["shop_name"].dropna().unique().tolist())
            if "shop_name" in topk.columns
            else []
        )
        sel_shop = st.selectbox("Shop", shops)
    with fc3:
        if "price" in topk.columns and topk["price"].notna().any():
            pmin, pmax = float(topk["price"].min()), float(topk["price"].max())
            price_range = st.slider("Price range ($)", pmin, pmax, (pmin, pmax))
        else:
            price_range = None
    with fc4:
        k = st.slider("Show top", 5, 100, 25)

    fd = topk.copy()
    if sel_cat != "All" and "category" in fd.columns:
        fd = fd[fd["category"] == sel_cat]
    if sel_shop != "All" and "shop_name" in fd.columns:
        fd = fd[fd["shop_name"] == sel_shop]
    if price_range and "price" in fd.columns:
        fd = fd[(fd["price"] >= price_range[0]) & (fd["price"] <= price_range[1])]
    if "score" in fd.columns:
        fd = fd.sort_values("score", ascending=False)
    fd = fd.head(k).reset_index(drop=True)

    if fd.empty:
        st.warning("No products match the current filter combination.")
        st.stop()

    leader = fd.iloc[0]
    leader_title = html.escape(safe_text(leader.get("title"), "No product title"))
    leader_shop = html.escape(safe_text(leader.get("shop_name"), "Unknown shop"))
    leader_category = html.escape(safe_text(leader.get("category"), "uncategorized"))
    leader_score = (
        float(leader["score"]) if "score" in leader and pd.notna(leader["score"]) else 0.0
    )
    leader_description = html.escape(short_text(leader.get("description"), 170))
    avg_score = float(fd["score"].mean()) if "score" in fd.columns else 0.0
    price_coverage = float(fd["price"].notna().mean() * 100) if "price" in fd.columns else 0.0
    shop_count = int(fd["shop_name"].nunique()) if "shop_name" in fd.columns else 0
    category_count = int(fd["category"].nunique()) if "category" in fd.columns else 0
    review_total = int(fd["review_count"].fillna(0).sum()) if "review_count" in fd.columns else 0
    avg_rating = (
        float(fd["rating"].dropna().mean())
        if "rating" in fd.columns and fd["rating"].notna().any()
        else np.nan
    )
    discount_share = (
        float(fd["discount_pct"].fillna(0).gt(0).mean()) if "discount_pct" in fd.columns else 0.0
    )
    in_stock_share = (
        float(fd["is_in_stock"].fillna(False).mean()) if "is_in_stock" in fd.columns else 0.0
    )
    score_gap = (
        float(fd.iloc[0]["score"] - fd.iloc[1]["score"])
        if len(fd) > 1 and "score" in fd.columns
        else 0.0
    )
    lead_platform = (
        safe_text(fd["source_platform"].value_counts().idxmax())
        if "source_platform" in fd.columns and not fd["source_platform"].dropna().empty
        else "Unknown"
    )

    if "shop_name" in fd.columns and "score" in fd.columns:
        dominant_shop = (
            fd.groupby("shop_name")["score"].mean().sort_values(ascending=False).index[0]
        )
    else:
        dominant_shop = "Unknown"

    shortlist_html = []
    for idx, (_, row) in enumerate(fd.head(4).iterrows(), start=1):
        shortlist_html.append(
            f'<div class="product-spot-card">'
            f'<div class="product-spot-topline">'
            f'<div class="product-spot-rank">{idx}</div>'
            f'<div class="product-spot-score">score {float(row.get("score", 0.0)):.3f}</div>'
            f"</div>"
            f'<div class="product-spot-title">{html.escape(safe_text(row.get("title"), "Untitled product"))}</div>'
            f'<div class="product-spot-copy">{html.escape(short_text(row.get("description"), 120))}</div>'
            f'<div class="pill-row">'
            f'<span class="signal-pill">{html.escape(safe_text(row.get("shop_name"), "Unknown shop"))}</span>'
            f'<span class="signal-pill">{html.escape(safe_text(row.get("category"), "uncategorized"))}</span>'
            f'<span class="signal-pill">{html.escape(format_currency(row.get("price")))}</span>'
            f'<span class="signal-pill">reviews {compact_number(row.get("review_count", 0) if pd.notna(row.get("review_count", 0)) else 0)}</span>'
            f"</div>"
            f"</div>"
        )

    st.markdown(
        f"""
        <div class="ranking-shell">
            <div class="ranking-card">
                <div class="ranking-kicker">Product rankings / active filter state</div>
                <div class="ranking-title">A ranking room built for shortlisting, not just chart-watching.</div>
                <div class="ranking-copy">
                    The current leader is <strong>{leader_title}</strong> from <strong>{leader_shop}</strong>, anchored in <strong>{leader_category}</strong> with a composite score of <strong>{leader_score:.3f}</strong>.
                    The current filter set keeps <strong>{len(fd)}</strong> products visible out of <strong>{len(topk)}</strong>, with <strong>{html.escape(safe_text(dominant_shop))}</strong> currently leading the filtered shop mix.
                </div>
                <div class="ranking-summary-grid">
                    <div class="ranking-mini">
                        <div class="ranking-mini-label">Score spread</div>
                        <div class="ranking-mini-value">{score_gap:.3f}</div>
                        <div class="ranking-mini-copy">Gap between rank #1 and rank #2. Useful for seeing whether the winner is clear or marginal.</div>
                    </div>
                    <div class="ranking-mini">
                        <div class="ranking-mini-label">Review gravity</div>
                        <div class="ranking-mini-value">{compact_number(review_total)}</div>
                        <div class="ranking-mini-copy">Total review mass available inside the current filtered leaderboard.</div>
                    </div>
                    <div class="ranking-mini">
                        <div class="ranking-mini-label">Lead platform</div>
                        <div class="ranking-mini-value">{html.escape(lead_platform)}</div>
                        <div class="ranking-mini-copy">Platform most represented in the current shortlist.</div>
                    </div>
                    <div class="ranking-mini">
                        <div class="ranking-mini-label">Price coverage</div>
                        <div class="ranking-mini-value">{price_coverage:.1f}%</div>
                        <div class="ranking-mini-copy">Visible products with usable price data for direct commercial comparison.</div>
                    </div>
                </div>
            </div>
            <div class="ranking-card">
                <div class="ranking-kicker">Curated top four</div>
                <div class="ranking-shortlist">{"".join(shortlist_html)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Average Score", f"{avg_score:.3f}")
    mc2.metric("Price Coverage", f"{price_coverage:.1f}%")
    mc3.metric("Visible Shops", shop_count)
    mc4.metric("Visible Categories", category_count)

    st.markdown(
        f"""
        <div class="ranking-notes">
            <div class="ranking-note">
                <div class="ranking-note-title">Commercial reading</div>
                <div class="ranking-note-copy">The current shortlist averages <strong>{avg_score:.3f}</strong> in score and <strong>{avg_rating:.2f}</strong> in rating{"" if pd.notna(avg_rating) else " where ratings exist"}.</div>
            </div>
            <div class="ranking-note">
                <div class="ranking-note-title">Availability posture</div>
                <div class="ranking-note-copy"><strong>{in_stock_share * 100:.1f}%</strong> of visible products are marked in stock, while <strong>{discount_share * 100:.1f}%</strong> currently carry a discount signal.</div>
            </div>
            <div class="ranking-note">
                <div class="ranking-note-title">Filter impact</div>
                <div class="ranking-note-copy">This current view spans <strong>{shop_count}</strong> shops and <strong>{category_count}</strong> categories, which makes it useful either as a broad leaderboard or a niche buying slice.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        f"Mode: {ranking_mode}. Showing {len(fd)} ranked products from the current filtered view."
    )

    if "score" in fd.columns and len(fd) > 0:
        leaderboard_tab, opportunity_tab, comparison_tab, export_tab = st.tabs(
            ["Leaderboard", "Opportunity Map", "Compare Views", "Decision Table"]
        )

        with leaderboard_tab:
            chart_col, rank_col = st.columns([1.25, 1.0])

            with chart_col:
                display_df = fd.head(20).copy()
                if "title" in display_df.columns:
                    display_df["label"] = display_df["title"].str[:42]
                else:
                    display_df["label"] = display_df.index.astype(str)
                fig = px.bar(
                    display_df,
                    x="score",
                    y="label",
                    orientation="h",
                    color="score",
                    color_continuous_scale=[
                        [0, C["secondary"]],
                        [0.55, C["accent"]],
                        [1, C["primary"]],
                    ],
                    hover_data=[
                        c
                        for c in ["shop_name", "price", "category", "review_count"]
                        if c in display_df.columns
                    ],
                )
                fig.update_layout(
                    coloraxis_showscale=False,
                    yaxis={"categoryorder": "total ascending"},
                    yaxis_title=None,
                    xaxis_title="Composite score",
                )
                fig = apply_theme(fig, f"Top {min(20, len(fd))} products by score", 520)
                st.plotly_chart(fig, width="stretch")

            with rank_col:
                spotlight_cols = [
                    c
                    for c in [
                        "title",
                        "shop_name",
                        "category",
                        "price",
                        "score",
                        "rating",
                        "review_count",
                    ]
                    if c in fd.columns
                ]
                spotlight_df = (
                    fd[spotlight_cols].head(8).copy() if spotlight_cols else fd.head(8).copy()
                )
                if "price" in spotlight_df.columns:
                    spotlight_df["price"] = spotlight_df["price"].map(
                        lambda value: format_currency(value) if pd.notna(value) else ""
                    )
                if "score" in spotlight_df.columns:
                    spotlight_df["score"] = spotlight_df["score"].map(
                        lambda value: f"{float(value):.3f}"
                    )
                if "rating" in spotlight_df.columns:
                    spotlight_df["rating"] = spotlight_df["rating"].map(
                        lambda value: f"{float(value):.2f}" if pd.notna(value) else ""
                    )
                st.dataframe(spotlight_df, height=520)

        with opportunity_tab:
            oc1, oc2 = st.columns([1.15, 0.85])

            with oc1:
                scatter_df = fd.copy()
                scatter_df["review_count"] = scatter_df.get("review_count", 0).fillna(0)
                scatter_df["rating"] = scatter_df.get("rating", 0).fillna(0)
                scatter_df["bubble_size"] = scatter_df["rating"].clip(lower=0.5) + 0.5
                color_col = "source_platform" if "source_platform" in scatter_df.columns else None
                fig = px.scatter(
                    scatter_df,
                    x="review_count",
                    y="score",
                    size="bubble_size",
                    color=color_col,
                    color_discrete_sequence=C["palette"],
                    hover_name="title" if "title" in scatter_df.columns else None,
                    hover_data=[
                        c
                        for c in ["shop_name", "category", "price", "rating"]
                        if c in scatter_df.columns
                    ],
                )
                fig.update_layout(xaxis_title="Review count", yaxis_title="Score")
                fig = apply_theme(fig, "Demand vs score", 470)
                st.plotly_chart(fig, width="stretch")

            with oc2:
                if "category" in fd.columns:
                    cat_scores = (
                        fd.groupby("category")["score"]
                        .mean()
                        .sort_values(ascending=False)
                        .head(8)
                        .reset_index()
                    )
                    fig = px.bar(
                        cat_scores,
                        x="score",
                        y="category",
                        orientation="h",
                        color="score",
                        color_continuous_scale=[[0, C["secondary"]], [1, C["primary"]]],
                    )
                    fig.update_layout(
                        coloraxis_showscale=False, xaxis_title="Average score", yaxis_title=None
                    )
                    fig = apply_theme(fig, "Best categories in view", 470)
                    st.plotly_chart(fig, width="stretch")

        with comparison_tab:
            comp1, comp2 = st.columns(2)

            with comp1:
                if "shop_name" in fd.columns:
                    shop_scores = (
                        fd.groupby("shop_name")
                        .agg(
                            avg_score=("score", "mean"),
                            products=("product_id", "count")
                            if "product_id" in fd.columns
                            else ("score", "count"),
                            avg_reviews=("review_count", "mean")
                            if "review_count" in fd.columns
                            else ("score", "mean"),
                        )
                        .reset_index()
                        .sort_values("avg_score", ascending=False)
                    )
                    fig = px.bar(
                        shop_scores,
                        x="avg_score",
                        y="shop_name",
                        orientation="h",
                        color="products",
                        color_continuous_scale=[[0, C["accent"]], [1, C["primary"]]],
                        hover_data=[c for c in ["avg_reviews"] if c in shop_scores.columns],
                    )
                    fig.update_layout(
                        coloraxis_showscale=False, xaxis_title="Average score", yaxis_title=None
                    )
                    fig = apply_theme(fig, "Shop score mix", 430)
                    st.plotly_chart(fig, width="stretch")

            with comp2:
                if "price_bucket" in fd.columns:
                    bucket_scores = (
                        fd.groupby("price_bucket")
                        .agg(avg_score=("score", "mean"), products=("score", "count"))
                        .reset_index()
                        .sort_values("avg_score", ascending=False)
                    )
                    fig = px.bar(
                        bucket_scores,
                        x="price_bucket",
                        y="avg_score",
                        color="products",
                        color_continuous_scale=[[0, C["secondary"]], [1, C["primary"]]],
                    )
                    fig.update_layout(
                        coloraxis_showscale=False,
                        xaxis_title="Price bucket",
                        yaxis_title="Average score",
                    )
                    fig = apply_theme(fig, "Score by price bucket", 430)
                    st.plotly_chart(fig, width="stretch")

        with export_tab:
            export_cols = [
                c
                for c in [
                    "title",
                    "shop_name",
                    "category",
                    "source_platform",
                    "price",
                    "rating",
                    "review_count",
                    "score",
                    "product_url",
                    "discount_pct",
                    "price_bucket",
                ]
                if c in fd.columns
            ]
            export_df = fd[export_cols].copy() if export_cols else fd.copy()
            if "price" in export_df.columns:
                export_df["price"] = export_df["price"].map(
                    lambda value: format_currency(value) if pd.notna(value) else ""
                )
            if "score" in export_df.columns:
                export_df["score"] = export_df["score"].map(lambda value: f"{float(value):.3f}")
            if "rating" in export_df.columns:
                export_df["rating"] = export_df["rating"].map(
                    lambda value: f"{float(value):.2f}" if pd.notna(value) else ""
                )
            if "discount_pct" in export_df.columns:
                export_df["discount_pct"] = export_df["discount_pct"].map(
                    lambda value: f"{float(value) * 100:.1f}%" if pd.notna(value) else ""
                )
            st.dataframe(export_df, height=500)
            st.download_button(
                "Download filtered leaderboard as CSV",
                data=fd.to_csv(index=False).encode("utf-8"),
                file_name="topk_filtered_view.csv",
                mime="text/csv",
            )

    # Per-category analysis
    topk_cat = load_csv("topk_per_category.csv")
    if not topk_cat.empty and "category" in topk_cat.columns and "score" in topk_cat.columns:
        with st.expander("Category Score Breakdown"):
            cat_avg = (
                topk_cat.groupby("category")["score"]
                .mean()
                .sort_values(ascending=False)
                .head(15)
                .reset_index()
            )
            cat_avg.columns = ["Category", "Avg Score"]
            fig = px.bar(
                cat_avg,
                x="Avg Score",
                y="Category",
                orientation="h",
                color="Avg Score",
                color_continuous_scale=[[0, C["primary"]], [1, C["success"]]],
            )
            fig.update_layout(coloraxis_showscale=False)
            fig = apply_theme(fig, "Average Score by Category", 380)
            st.plotly_chart(fig, width="stretch")


# ══════════════════════════════════════════════════════════════
# SHOP ANALYSIS
# ══════════════════════════════════════════════════════════════
elif page == "Shop Analysis":
    st.markdown('<div class="page-title">Shop Performance</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Compare shop scale, pricing, ratings, discount posture, and ranking quality inside a single performance board.</div>',
        unsafe_allow_html=True,
    )

    df = load_features()
    topk_shop = load_csv("topk_per_shop.csv")
    topk_products = load_csv("topk_products.csv")

    if df.empty:
        st.warning("No data available.")
        st.stop()

    if "shop_name" in df.columns:
        stats = (
            df.groupby("shop_name")
            .agg(
                products=("product_id", "count"),
                avg_price=("price", "mean"),
                avg_rating=("rating", "mean"),
                avg_discount=("discount_pct", "mean"),
            )
            .reset_index()
            .sort_values("products", ascending=False)
        )

        # Prefer shop-level artifact; fallback to topk_products when legacy topk_per_shop lacks keys.
        if (
            not topk_shop.empty
            and "shop_name" in topk_shop.columns
            and "score" in topk_shop.columns
        ):
            shop_scores = (
                topk_shop.groupby("shop_name")["score"]
                .mean()
                .reset_index()
                .rename(columns={"score": "avg_score"})
            )
            stats = stats.merge(shop_scores, on="shop_name", how="left")
        elif (
            not topk_products.empty
            and "shop_name" in topk_products.columns
            and "score" in topk_products.columns
        ):
            shop_scores = (
                topk_products.groupby("shop_name")["score"]
                .mean()
                .reset_index()
                .rename(columns={"score": "avg_score"})
            )
            stats = stats.merge(shop_scores, on="shop_name", how="left")
        else:
            stats["avg_score"] = np.nan

        stats["avg_rating"] = stats["avg_rating"].fillna(0)
        stats["avg_discount"] = stats["avg_discount"].fillna(0)
        largest_shop = stats.iloc[0]["shop_name"] if not stats.empty else "Unknown"
        max_products = int(stats["products"].max()) if not stats.empty else 0
        best_shop = (
            stats.sort_values("avg_score", ascending=False).iloc[0]["shop_name"]
            if "avg_score" in stats.columns and stats["avg_score"].notna().any()
            else largest_shop
        )
        rating_leader = (
            stats.loc[stats["avg_rating"] > 0]
            .sort_values("avg_rating", ascending=False)
            .iloc[0]["shop_name"]
            if (stats["avg_rating"] > 0).any()
            else "No rated shop"
        )
        avg_shop_rating = (
            float(stats.loc[stats["avg_rating"] > 0, "avg_rating"].mean())
            if (stats["avg_rating"] > 0).any()
            else np.nan
        )
        concentration = (
            float(max_products / stats["products"].sum()) if stats["products"].sum() else 0.0
        )

        dossier_html = []
        for _, row in stats.head(4).iterrows():
            avg_price_text = (
                format_currency(row.get("avg_price"))
                if pd.notna(row.get("avg_price"))
                else "Price unavailable"
            )
            avg_rating_text = (
                f"{float(row.get('avg_rating', 0.0)):.2f}"
                if pd.notna(row.get("avg_rating")) and row.get("avg_rating", 0.0) > 0
                else "n/a"
            )
            avg_score_text = (
                f"{float(row.get('avg_score')):.3f}" if pd.notna(row.get("avg_score")) else "n/a"
            )
            catalog_share = (
                (float(row.get("products", 0)) / stats["products"].sum()) * 100
                if stats["products"].sum()
                else 0.0
            )
            dossier_html.append(
                f'<div class="shop-dossier">'
                f'<div class="shop-dossier-top">'
                f'<div class="shop-dossier-name">{html.escape(safe_text(row.get("shop_name")))}</div>'
                f'<div class="shop-dossier-score">avg score {avg_score_text}</div>'
                f"</div>"
                f'<div class="shop-dossier-copy">'
                f"{int(row.get('products', 0))} tracked products · "
                f"avg price {avg_price_text} · "
                f"avg rating {avg_rating_text}"
                f"</div>"
                f'<div class="pill-row">'
                f'<span class="signal-pill">discount {float(row.get("avg_discount", 0.0)) * 100:.1f}%</span>'
                f'<span class="signal-pill">catalog share {catalog_share:.1f}%</span>'
                f"</div>"
                f"</div>"
            )

        st.markdown(
            f"""
            <div class="shop-shell">
                <div class="shop-card">
                    <div class="shop-kicker">Shop intelligence / comparative view</div>
                    <div class="shop-title">A merchant board for reading who dominates, who converts attention, and who lags.</div>
                    <div class="shop-copy">
                        The current catalog leader by scale is <strong>{html.escape(safe_text(largest_shop))}</strong>, while the best visible average score belongs to <strong>{html.escape(safe_text(best_shop))}</strong>.
                        This page is designed to separate raw catalog size from quality signals such as average rating, discount posture, and score concentration.
                    </div>
                    <div class="shop-mini-grid">
                        <div class="shop-mini">
                            <div class="shop-mini-label">Catalog concentration</div>
                            <div class="shop-mini-value">{concentration * 100:.1f}%</div>
                            <div class="shop-mini-copy">Share of all tracked products owned by the largest single shop.</div>
                        </div>
                        <div class="shop-mini">
                            <div class="shop-mini-label">Highest score shop</div>
                            <div class="shop-mini-value">{html.escape(safe_text(best_shop))}</div>
                            <div class="shop-mini-copy">Average score leader among shops represented in the Top-K per shop output.</div>
                        </div>
                        <div class="shop-mini">
                            <div class="shop-mini-label">Largest catalog</div>
                            <div class="shop-mini-value">{max_products}</div>
                            <div class="shop-mini-copy">Tracked product count for the biggest visible merchant catalog.</div>
                        </div>
                        <div class="shop-mini">
                            <div class="shop-mini-label">Rating leader</div>
                            <div class="shop-mini-value">{html.escape(safe_text(rating_leader))}</div>
                            <div class="shop-mini-copy">Shop with the strongest average rating among rows with rating coverage.</div>
                        </div>
                    </div>
                </div>
                <div class="shop-card">
                    <div class="shop-kicker">Top shop dossiers</div>
                    <div class="shop-dossier-list">{"".join(dossier_html)}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Tracked Shops", len(stats))
        s2.metric("Largest Catalog", largest_shop)
        s3.metric("Max Products", max_products)
        s4.metric("Avg Shop Rating", f"{avg_shop_rating:.2f}" if pd.notna(avg_shop_rating) else "—")

        st.markdown(
            f"""
            <div class="shop-note-grid">
                <div class="shop-note">
                    <div class="shop-note-title">Scale vs quality</div>
                    <div class="shop-note-copy">The biggest catalog belongs to <strong>{html.escape(safe_text(largest_shop))}</strong>, but the strongest average score sits with <strong>{html.escape(safe_text(best_shop))}</strong>. Those are not always the same merchant.</div>
                </div>
                <div class="shop-note">
                    <div class="shop-note-title">Rating coverage</div>
                    <div class="shop-note-copy">The rating leader is <strong>{html.escape(safe_text(rating_leader))}</strong>. This is useful because many shops still have incomplete review/rating coverage.</div>
                </div>
                <div class="shop-note">
                    <div class="shop-note-title">Market concentration</div>
                    <div class="shop-note-copy">The largest shop controls <strong>{concentration * 100:.1f}%</strong> of all tracked products, which is the quickest indicator of assortment imbalance.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-header">Shop comparison table</div>', unsafe_allow_html=True
        )
        display_stats = stats.copy()
        if "avg_score" in display_stats.columns:
            display_stats["avg_score"] = display_stats["avg_score"].map(
                lambda value: f"{float(value):.3f}" if pd.notna(value) else ""
            )
        if "avg_price" in display_stats.columns:
            display_stats["avg_price"] = display_stats["avg_price"].map(
                lambda value: format_currency(value) if pd.notna(value) else ""
            )
        if "avg_rating" in display_stats.columns:
            display_stats["avg_rating"] = display_stats["avg_rating"].map(
                lambda value: f"{float(value):.2f}" if pd.notna(value) and value > 0 else ""
            )
        if "avg_discount" in display_stats.columns:
            display_stats["avg_discount"] = display_stats["avg_discount"].map(
                lambda value: f"{float(value) * 100:.1f}%" if pd.notna(value) else ""
            )
        st.dataframe(display_stats, height=320)

        tab1, tab2, tab3, tab4 = st.tabs(
            ["Radar Comparison", "Price Distribution", "Score Ranking", "Shop Matrix"]
        )

        with tab1:
            metrics = ["avg_price", "avg_rating", "avg_discount", "products"]
            rd = stats.copy()
            for m in metrics:
                mn, mx = rd[m].min(), rd[m].max()
                rd[f"{m}_n"] = (rd[m] - mn) / (mx - mn) if mx > mn else 0.5

            fig = go.Figure()
            for i, (_, row) in enumerate(rd.iterrows()):
                vals = [row[f"{m}_n"] for m in metrics] + [row[f"{metrics[0]}_n"]]
                theta = ["Price", "Rating", "Discount", "Products", "Price"]
                fig.add_trace(
                    go.Scatterpolar(
                        r=vals,
                        theta=theta,
                        fill="toself",
                        name=row["shop_name"],
                        opacity=0.55,
                        line=dict(color=C["palette"][i % len(C["palette"])]),
                    )
                )
            fig = apply_theme(fig, "Shop Comparison", 450)
            fig.update_layout(
                polar=dict(
                    bgcolor=C["card"],
                    radialaxis=dict(gridcolor="rgba(214,168,95,0.10)", showticklabels=False),
                    angularaxis=dict(gridcolor="rgba(214,168,95,0.10)"),
                )
            )
            st.plotly_chart(fig, width="stretch")

        with tab2:
            fig = px.box(
                df.dropna(subset=["price"]),
                x="shop_name",
                y="price",
                color="shop_name",
                color_discrete_sequence=C["palette"],
            )
            fig = apply_theme(fig, "Price Distribution by Shop", 400)
            fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Price ($)")
            st.plotly_chart(fig, width="stretch")

        with tab3:
            if "avg_score" in stats.columns and stats["avg_score"].notna().any():
                ss = (
                    stats[["shop_name", "avg_score", "products"]]
                    .dropna(subset=["avg_score"])
                    .sort_values("avg_score", ascending=True)
                    .rename(
                        columns={
                            "shop_name": "Shop",
                            "avg_score": "Avg Score",
                            "products": "Products",
                        }
                    )
                    .copy()
                )
                fig = px.bar(
                    ss,
                    x="Avg Score",
                    y="Shop",
                    orientation="h",
                    color="Avg Score",
                    color_continuous_scale=[
                        [0, C["secondary"]],
                        [0.5, C["warning"]],
                        [1, C["success"]],
                    ],
                )
                fig.update_layout(coloraxis_showscale=False)
                fig = apply_theme(fig, "Shop Score Ranking", 380)
                st.plotly_chart(fig, width="stretch")
            else:
                st.info(
                    "Shop score ranking unavailable: no usable `shop_name`+`score` artifact in current analytics."
                )

        with tab4:
            matrix = stats.copy()
            matrix["price_fill"] = matrix["avg_price"].fillna(matrix["avg_price"].median())
            y_col = (
                "avg_score"
                if ("avg_score" in matrix.columns and matrix["avg_score"].notna().any())
                else "avg_discount"
            )
            fig = px.scatter(
                matrix,
                x="products",
                y=y_col,
                size="price_fill",
                color="avg_discount",
                hover_name="shop_name",
                color_continuous_scale=[[0, C["accent"]], [1, C["primary"]], [1, C["secondary"]]],
            )
            fig.update_layout(
                xaxis_title="Tracked products",
                yaxis_title="Average score" if y_col == "avg_score" else "Average discount",
                coloraxis_colorbar_title="Avg discount",
            )
            fig = apply_theme(fig, "Shop matrix: scale vs score", 430)
            st.plotly_chart(fig, width="stretch")


# ══════════════════════════════════════════════════════════════
# ML MODELS
# ══════════════════════════════════════════════════════════════
elif page == "ML Models":
    st.markdown('<div class="page-title">Machine Learning Models</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Model evaluation, feature influence, and error surfaces for high-potential product prediction.</div>',
        unsafe_allow_html=True,
    )

    rf = load_json("model_metrics.json")
    xgb = load_json("model_metrics_xgboost.json")

    if not rf and not xgb:
        st.warning("No model metrics. Run training steps.")
        st.stop()

    rf_acc = rf.get("accuracy") if rf else np.nan
    xgb_acc = xgb.get("accuracy") if xgb else np.nan
    rf_f1 = rf.get("f1") if rf else np.nan
    xgb_f1 = xgb.get("f1") if xgb else np.nan
    rf_gate = rf.get("honesty_gate", {}) if rf else {}
    xgb_gate = xgb.get("honesty_gate", {}) if xgb else {}
    rf_status = safe_text(rf_gate.get("status"), "unknown").upper()
    xgb_status = safe_text(xgb_gate.get("status"), "unknown").upper()
    target_origin = safe_text((rf or xgb or {}).get("target_origin"), "unknown")
    grouped_rf_f1 = (rf.get("grouped_cv") or {}).get("f1") if rf else None
    grouped_xgb_f1 = (xgb.get("grouped_cv") or {}).get("f1") if xgb else None
    grouped_rf_cal = (rf.get("grouped_cv") or {}).get("calibration") if rf else None
    grouped_xgb_cal = (xgb.get("grouped_cv") or {}).get("calibration") if xgb else None
    grouped_feas = (
        (rf.get("grouped_cv_feasibility") or xgb.get("grouped_cv_feasibility") or {})
        if (rf or xgb)
        else {}
    )
    best_model = (
        "XGBoost"
        if pd.notna(xgb_f1) and xgb_f1 >= (rf_f1 if pd.notna(rf_f1) else -1)
        else "RandomForest"
    )
    f1_gap = abs((xgb_f1 if pd.notna(xgb_f1) else 0.0) - (rf_f1 if pd.notna(rf_f1) else 0.0))
    sample_count = max([m.get("n_samples", 0) for m in [rf, xgb] if m], default=0)
    feature_count = max([m.get("n_features", 0) for m in [rf, xgb] if m], default=0)
    top_driver = (
        rf.get("feature_importance", [{}])[0].get("feature", "n/a")
        if rf and rf.get("feature_importance")
        else "n/a"
    )

    if rf_status == "RED" or xgb_status == "RED":
        st.error(
            "Reliability warning: at least one model is honesty-gate RED. "
            "Treat results as heuristic consistency, not real-world predictiveness."
        )
    elif rf_status == "YELLOW" or xgb_status == "YELLOW":
        st.warning(
            "Reliability caution: honesty-gate indicates elevated risk of over-optimistic performance."
        )
    else:
        st.success("Reliability check: honesty-gate is green for available models.")

    if grouped_feas.get("high_risk"):
        st.warning(
            "Grouped-CV feasibility warning: positive labels are concentrated in very few shops. "
            "Cross-shop recall may collapse even with threshold tuning."
        )

    driver_html = []
    if rf and rf.get("feature_importance"):
        for feature in rf["feature_importance"][:5]:
            driver_html.append(
                f'<div class="model-driver">'
                f'<div class="model-driver-name">{html.escape(safe_text(feature.get("feature"), "unknown feature"))}</div>'
                f'<div class="model-driver-value">{float(feature.get("importance", 0.0)):.4f}</div>'
                f"</div>"
            )

    st.markdown(
        f"""
        <div class="model-shell">
            <div class="model-card">
                <div class="model-kicker">Model review / validated training outputs</div>
                <div class="model-title">A model room for reading confidence, separation, and business-facing reliability.</div>
                <div class="model-copy">
                    The current comparison is led by <strong>{best_model}</strong>, with the best F1 currently at <strong>{max([m.get("f1", 0) for m in [rf, xgb] if m]):.3f}</strong>.
                    This page is designed to answer whether the models are merely accurate, or whether they are meaningfully stable, interpretable, and worth presenting as part of the pipeline story.
                    Current target origin is <strong>{html.escape(target_origin)}</strong>; grouped-by-shop CV is shown below to reduce memorization risk.
                </div>
                <div class="model-mini-grid">
                    <div class="model-mini">
                        <div class="model-mini-label">Best model</div>
                        <div class="model-mini-value">{best_model}</div>
                        <div class="model-mini-copy">Current winner by F1 score on the validated analytics snapshot.</div>
                    </div>
                    <div class="model-mini">
                        <div class="model-mini-label">F1 gap</div>
                        <div class="model-mini-value">{f1_gap:.3f}</div>
                        <div class="model-mini-copy">How far apart the two model families currently are in balanced performance.</div>
                    </div>
                    <div class="model-mini">
                        <div class="model-mini-label">Training rows</div>
                        <div class="model-mini-value">{sample_count}</div>
                        <div class="model-mini-copy">Largest sample count reported in the current model metrics payload.</div>
                    </div>
                    <div class="model-mini">
                        <div class="model-mini-label">Feature set</div>
                        <div class="model-mini-value">{feature_count}</div>
                        <div class="model-mini-copy">Number of engineered features available to the current models.</div>
                    </div>
                    <div class="model-mini">
                        <div class="model-mini-label">RF grouped F1</div>
                        <div class="model-mini-value">{(f"{float(grouped_rf_f1):.3f}" if grouped_rf_f1 is not None else "n/a")}</div>
                        <div class="model-mini-copy">GroupKFold score using <strong>shop_name</strong> as fold boundary.</div>
                    </div>
                    <div class="model-mini">
                        <div class="model-mini-label">XGB grouped F1</div>
                        <div class="model-mini-value">{(f"{float(grouped_xgb_f1):.3f}" if grouped_xgb_f1 is not None else "n/a")}</div>
                        <div class="model-mini-copy">Grouped validation score for cross-shop generalization realism.</div>
                    </div>
                </div>
            </div>
            <div class="model-card">
                <div class="model-kicker">Top driver snapshot</div>
                <div class="model-copy" style="margin-bottom: 14px;">The strongest current RandomForest driver is <strong>{html.escape(safe_text(top_driver))}</strong>. These are the variables currently steering the classification boundary the most.</div>
                <div class="model-driver-list">{"".join(driver_html) if driver_html else '<div class="model-note-copy">Feature importance is not available in the current metrics payload.</div>'}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model_f1s = [m.get("f1", 0) for m in [rf, xgb] if m]
    m1, m2, m3 = st.columns(3)
    m1.metric("Models Loaded", sum(bool(m) for m in [rf, xgb]))
    m2.metric("Best F1", f"{max(model_f1s):.3f}" if model_f1s else "—")
    m3.metric("Best Accuracy", f"{max([m.get('accuracy', 0) for m in [rf, xgb] if m]):.3f}")

    st.markdown(
        f"""
        <div class="model-note-grid">
            <div class="model-note">
                <div class="model-note-title">Accuracy posture</div>
                <div class="model-note-copy">RandomForest accuracy is <strong>{rf_acc:.3f}</strong> and XGBoost accuracy is <strong>{xgb_acc:.3f}</strong>. The gap is small, which suggests the feature space is already doing much of the heavy lifting.</div>
            </div>
            <div class="model-note">
                <div class="model-note-title">Balanced performance</div>
                <div class="model-note-copy">RandomForest F1 is <strong>{rf_f1:.3f}</strong> while XGBoost reaches <strong>{xgb_f1:.3f}</strong>. This is the more useful headline when positive-class quality matters.</div>
            </div>
            <div class="model-note">
                <div class="model-note-title">Reliability status</div>
                <div class="model-note-copy">Honesty gate status: <strong>RF {rf_status}</strong> and <strong>XGB {xgb_status}</strong>. If either is RED, model performance should be communicated as exploratory, not predictive certainty.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Performance", "Feature Importance", "Confusion Matrix", "Model Deltas"]
    )

    with tab1:
        models = []
        if rf:
            models.append(
                {
                    "Model": "RandomForest",
                    **{k: rf.get(k) for k in ["accuracy", "precision", "recall", "f1"]},
                }
            )
        if xgb:
            models.append(
                {
                    "Model": "XGBoost",
                    **{k: xgb.get(k) for k in ["accuracy", "precision", "recall", "f1"]},
                }
            )

        if models:
            model_table = pd.DataFrame(models)
            st.dataframe(
                model_table.style.format(
                    {k: "{:.3f}" for k in ["accuracy", "precision", "recall", "f1"]}
                ),
                height=140,
            )

            met = ["accuracy", "precision", "recall", "f1"]
            fig = go.Figure()
            colors = [C["primary"], C["accent"]]
            for i, m in enumerate(models):
                fig.add_trace(
                    go.Bar(
                        name=m["Model"],
                        x=met,
                        y=[m.get(k, 0) for k in met],
                        marker_color=colors[i % 2],
                        text=[f"{m.get(k, 0):.3f}" for k in met],
                        textposition="outside",
                        textfont_size=11,
                    )
                )
            fig.update_layout(barmode="group", yaxis_range=[0, 1.12])
            fig = apply_theme(fig, "Model Comparison", 400)
            st.plotly_chart(fig, width="stretch")

    with tab2:
        if rf.get("feature_importance"):
            fi = pd.DataFrame(rf["feature_importance"]).sort_values("importance")
            fig = px.bar(
                fi,
                x="importance",
                y="feature",
                orientation="h",
                color="importance",
                color_continuous_scale=[[0, C["primary"]], [1, C["success"]]],
            )
            fig.update_layout(coloraxis_showscale=False, yaxis_title=None)
            fig = apply_theme(fig, "RandomForest — Feature Importance", 380)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Feature importance data not available. Re-run training.")

    with tab3:
        for name, metrics in [("RandomForest", rf), ("XGBoost", xgb)]:
            if metrics.get("confusion_matrix"):
                cm = np.array(metrics["confusion_matrix"])
                labels = ["Negative", "Positive"]
                fig = px.imshow(
                    cm,
                    text_auto=True,
                    x=labels,
                    y=labels,
                    color_continuous_scale=[[0, C["card"]], [1, C["primary"]]],
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                )
                fig = apply_theme(fig, f"{name} — Confusion Matrix", 320)
                st.plotly_chart(fig, width="stretch")

    with tab4:
        delta_rows = []
        if rf and xgb:
            for metric in ["accuracy", "precision", "recall", "f1"]:
                delta_rows.append(
                    {
                        "Metric": metric,
                        "RandomForest": rf.get(metric, 0),
                        "XGBoost": xgb.get(metric, 0),
                        "Delta": xgb.get(metric, 0) - rf.get(metric, 0),
                    }
                )
        if delta_rows:
            delta_df = pd.DataFrame(delta_rows)
            st.dataframe(
                delta_df.style.format(
                    {"RandomForest": "{:.3f}", "XGBoost": "{:.3f}", "Delta": "{:+.3f}"}
                ),
                height=220,
            )
            fig = px.bar(
                delta_df,
                x="Metric",
                y="Delta",
                color="Delta",
                color_continuous_scale=[
                    [0, C["secondary"]],
                    [0.5, C["warning"]],
                    [1, C["success"]],
                ],
            )
            fig.update_layout(coloraxis_showscale=False, yaxis_title="XGBoost - RandomForest")
            fig = apply_theme(fig, "Metric delta view", 360)
            st.plotly_chart(fig, width="stretch")

    with st.expander("Grouped-CV calibration diagnostics", expanded=False):
        diag_rows = []
        if grouped_rf_cal and isinstance(grouped_rf_cal, dict):
            rf_best = grouped_rf_cal.get("best", {})
            rf_default = grouped_rf_cal.get("default", {})
            diag_rows.extend(
                [
                    {
                        "Model": "RandomForest",
                        "Profile": "Best threshold",
                        "Threshold": rf_best.get("threshold"),
                        "Precision": rf_best.get("precision"),
                        "Recall": rf_best.get("recall"),
                        "F1": rf_best.get("f1"),
                    },
                    {
                        "Model": "RandomForest",
                        "Profile": "Default threshold (0.5)",
                        "Threshold": rf_default.get("threshold"),
                        "Precision": rf_default.get("precision"),
                        "Recall": rf_default.get("recall"),
                        "F1": rf_default.get("f1"),
                    },
                ]
            )
        if grouped_xgb_cal and isinstance(grouped_xgb_cal, dict):
            xgb_best = grouped_xgb_cal.get("best", {})
            xgb_default = grouped_xgb_cal.get("default", {})
            diag_rows.extend(
                [
                    {
                        "Model": "XGBoost",
                        "Profile": "Best threshold",
                        "Threshold": xgb_best.get("threshold"),
                        "Precision": xgb_best.get("precision"),
                        "Recall": xgb_best.get("recall"),
                        "F1": xgb_best.get("f1"),
                    },
                    {
                        "Model": "XGBoost",
                        "Profile": "Default threshold (0.5)",
                        "Threshold": xgb_default.get("threshold"),
                        "Precision": xgb_default.get("precision"),
                        "Recall": xgb_default.get("recall"),
                        "F1": xgb_default.get("f1"),
                    },
                ]
            )
        if diag_rows:
            diag_df = pd.DataFrame(diag_rows)
            st.dataframe(
                diag_df.style.format(
                    {
                        "Threshold": "{:.2f}",
                        "Precision": "{:.3f}",
                        "Recall": "{:.3f}",
                        "F1": "{:.3f}",
                    }
                ),
                height=220,
            )
        else:
            st.info("Grouped-CV calibration diagnostics are not available in current metrics.")

    with st.expander("Positive-label concentration by shop", expanded=False):
        by_shop = grouped_feas.get("by_shop") if isinstance(grouped_feas, dict) else None
        if by_shop:
            by_shop_df = pd.DataFrame(by_shop)
            if not by_shop_df.empty:
                by_shop_df["positive_rate"] = (
                    by_shop_df["positives"] / by_shop_df["rows"].replace(0, np.nan)
                ).fillna(0.0)
                st.dataframe(
                    by_shop_df[["shop_name", "positives", "rows", "positive_rate"]].style.format(
                        {"positive_rate": "{:.3f}"}
                    ),
                    height=240,
                )
                st.caption(
                    f"Shops with positives: {grouped_feas.get('shops_with_positive_labels', 'n/a')} / {grouped_feas.get('total_shops', 'n/a')}; "
                    f"max single-shop positive share: {float(grouped_feas.get('max_positive_share_single_shop', 0.0)) * 100:.1f}%"
                )
        else:
            st.info(
                "Positive-label concentration diagnostics are not available in current metrics."
            )


# ══════════════════════════════════════════════════════════════
# SEGMENTATION
# ══════════════════════════════════════════════════════════════
elif page == "Segmentation":
    st.markdown('<div class="page-title">Product Segmentation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Cluster structure and anomaly detection for understanding how the catalog organizes itself beyond simple ranking.</div>',
        unsafe_allow_html=True,
    )

    pca = load_csv("pca_viz.csv")
    clusters = load_csv("clusters.csv")
    dbscan = load_csv("dbscan_clusters.csv")

    kmeans_cluster_count = (
        int(clusters["cluster"].nunique())
        if not clusters.empty and "cluster" in clusters.columns
        else 0
    )
    largest_cluster = (
        int(clusters["cluster"].value_counts().max())
        if not clusters.empty and "cluster" in clusters.columns
        else 0
    )
    avg_cluster_score = (
        float(clusters["score"].mean())
        if not clusters.empty and "score" in clusters.columns and clusters["score"].notna().any()
        else np.nan
    )
    dbscan_outliers = (
        int((dbscan["dbscan_cluster"] == -1).sum())
        if not dbscan.empty and "dbscan_cluster" in dbscan.columns
        else 0
    )
    dbscan_cluster_count = (
        int(
            dbscan["dbscan_cluster"].nunique() - (1 if -1 in dbscan["dbscan_cluster"].values else 0)
        )
        if not dbscan.empty and "dbscan_cluster" in dbscan.columns
        else 0
    )

    segment_html = []
    if not clusters.empty and "cluster" in clusters.columns:
        cluster_summary = (
            clusters.groupby("cluster")
            .agg(
                products=("cluster", "size"),
                avg_score=("score", "mean") if "score" in clusters.columns else ("cluster", "size"),
                top_shop=(
                    "shop_name",
                    lambda s: s.mode().iloc[0] if not s.mode().empty else "Unknown",
                )
                if "shop_name" in clusters.columns
                else ("cluster", lambda _: "Unknown"),
                top_category=(
                    "category",
                    lambda s: s.mode().iloc[0] if not s.mode().empty else "uncategorized",
                )
                if "category" in clusters.columns
                else ("cluster", lambda _: "uncategorized"),
            )
            .reset_index()
            .sort_values("products", ascending=False)
        )
        for _, row in cluster_summary.head(4).iterrows():
            segment_html.append(
                f'<div class="segment-driver">'
                f'<div class="segment-driver-name">Cluster {int(row.get("cluster", 0))} · {html.escape(safe_text(row.get("top_category"), "uncategorized"))}</div>'
                f'<div class="segment-driver-value">{int(row.get("products", 0))} items</div>'
                f"</div>"
            )
    else:
        cluster_summary = pd.DataFrame()

    st.markdown(
        f"""
        <div class="segment-shell">
            <div class="segment-card">
                <div class="segment-kicker">Segmentation / catalog geometry</div>
                <div class="segment-title">A structure map for understanding how the catalog groups itself and where the anomalies live.</div>
                <div class="segment-copy">
                    KMeans currently partitions the catalog into <strong>{kmeans_cluster_count}</strong> groups, while DBSCAN finds <strong>{dbscan_cluster_count}</strong> dense regions and <strong>{dbscan_outliers}</strong> anomalous products.
                    This page is designed to explain whether the catalog has coherent structure, where high-score pockets exist, and which products sit outside the normal shape.
                </div>
                <div class="segment-mini-grid">
                    <div class="segment-mini">
                        <div class="segment-mini-label">KMeans clusters</div>
                        <div class="segment-mini-value">{kmeans_cluster_count}</div>
                        <div class="segment-mini-copy">Distinct clusters found in the current PCA-backed segmentation view.</div>
                    </div>
                    <div class="segment-mini">
                        <div class="segment-mini-label">Largest cluster</div>
                        <div class="segment-mini-value">{largest_cluster}</div>
                        <div class="segment-mini-copy">Product count in the densest KMeans segment.</div>
                    </div>
                    <div class="segment-mini">
                        <div class="segment-mini-label">Avg cluster score</div>
                        <div class="segment-mini-value">{avg_cluster_score:.3f}</div>
                        <div class="segment-mini-copy">Average score across products carrying cluster assignments.</div>
                    </div>
                    <div class="segment-mini">
                        <div class="segment-mini-label">DBSCAN outliers</div>
                        <div class="segment-mini-value">{dbscan_outliers}</div>
                        <div class="segment-mini-copy">Products flagged as out-of-pattern by density-based clustering.</div>
                    </div>
                </div>
            </div>
            <div class="segment-card">
                <div class="segment-kicker">Segment snapshot</div>
                <div class="segment-copy" style="margin-bottom: 14px;">These are the biggest cluster groupings currently visible in the catalog, summarized by dominant category and size.</div>
                <div class="segment-driver-list">{"".join(segment_html) if segment_html else '<div class="segment-note-copy">Cluster summary becomes available once clustering outputs are present.</div>'}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="segment-note-grid">
            <div class="segment-note">
                <div class="segment-note-title">Cluster density</div>
                <div class="segment-note-copy">The largest KMeans cluster currently holds <strong>{largest_cluster}</strong> products. That helps indicate whether the feature space is balanced or dominated by one commercial profile.</div>
            </div>
            <div class="segment-note">
                <div class="segment-note-title">Outlier read</div>
                <div class="segment-note-copy">DBSCAN flags <strong>{dbscan_outliers}</strong> products as anomalies. These are useful for identifying unusual assortments, niche items, or noisy catalog rows.</div>
            </div>
            <div class="segment-note">
                <div class="segment-note-title">Score behavior</div>
                <div class="segment-note-copy">Average score across clustered products currently sits at <strong>{avg_cluster_score:.3f}</strong>, which helps connect segmentation back to downstream ranking quality.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["KMeans Clustering", "DBSCAN Anomalies", "Segment Tables"])

    with tab1:
        if not pca.empty and "cluster" in pca.columns:
            pca["cluster"] = pca["cluster"].astype(str)
            fig = px.scatter(
                pca,
                x="pc1",
                y="pc2",
                color="cluster",
                color_discrete_sequence=C["palette"],
                opacity=0.7,
            )
            fig.update_traces(marker=dict(size=7, line=dict(width=0.5, color="rgba(0,0,0,0.3)")))
            fig = apply_theme(fig, "PCA Projection — KMeans Clusters", 480)
            fig.update_layout(xaxis_title="PC 1", yaxis_title="PC 2")
            st.plotly_chart(fig, width="stretch")

            if not clusters.empty:
                ca, cb = st.columns(2)
                with ca:
                    sizes = clusters["cluster"].value_counts().reset_index()
                    sizes.columns = ["Cluster", "Size"]
                    sizes["Cluster"] = sizes["Cluster"].astype(str)
                    fig = px.pie(
                        sizes,
                        values="Size",
                        names="Cluster",
                        hole=0.55,
                        color_discrete_sequence=C["palette"],
                    )
                    fig.update_traces(textposition="inside", textinfo="percent+label")
                    fig = apply_theme(fig, "Cluster Distribution", 320)
                    st.plotly_chart(fig, width="stretch")
                with cb:
                    if "score" in clusters.columns:
                        cs = (
                            clusters.groupby("cluster")
                            .agg(count=("cluster", "size"), avg_score=("score", "mean"))
                            .reset_index()
                        )
                        cs.columns = ["Cluster", "Products", "Avg Score"]
                        st.dataframe(cs.style.format({"Avg Score": "{:.3f}"}), height=260)
        else:
            st.info("Run clustering step for PCA visualization.")

    with tab2:
        if not dbscan.empty and "dbscan_cluster" in dbscan.columns:
            n_out = int((dbscan["dbscan_cluster"] == -1).sum())
            n_cl = dbscan["dbscan_cluster"].nunique() - (
                1 if -1 in dbscan["dbscan_cluster"].values else 0
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("Clusters", n_cl)
            c2.metric("Outliers", n_out)
            c3.metric("Total", len(dbscan))

            ds = dbscan["dbscan_cluster"].value_counts().reset_index()
            ds.columns = ["Cluster", "Count"]
            ds["Cluster"] = ds["Cluster"].astype(str)
            ds["Type"] = ds["Cluster"].apply(lambda x: "Outlier" if x == "-1" else "Cluster")
            fig = px.bar(
                ds,
                x="Cluster",
                y="Count",
                color="Type",
                color_discrete_map={"Outlier": C["secondary"], "Cluster": C["primary"]},
            )
            fig = apply_theme(fig, "DBSCAN Cluster Distribution", 340)
            st.plotly_chart(fig, width="stretch")

            with st.expander(f"View {n_out} Outlier Products"):
                outs = dbscan[dbscan["dbscan_cluster"] == -1]
                cols = [c for c in ["title", "shop_name", "category"] if c in outs.columns]
                st.dataframe(outs[cols] if cols else outs)
        else:
            st.info("Run DBSCAN step for anomaly detection.")

    with tab3:
        t1, t2 = st.columns(2)
        with t1:
            if not clusters.empty:
                cluster_table = clusters.copy()
                if "score" in cluster_table.columns:
                    cluster_table["score"] = cluster_table["score"].map(
                        lambda value: f"{float(value):.3f}" if pd.notna(value) else ""
                    )
                st.dataframe(cluster_table.head(80), height=420)
        with t2:
            if not dbscan.empty:
                st.dataframe(dbscan.head(80), height=420)


# ══════════════════════════════════════════════════════════════
# ASSOCIATION RULES
# ══════════════════════════════════════════════════════════════
elif page == "Association Rules":
    st.markdown('<div class="page-title">Association Rules</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Pattern mining across categories, platforms, and price buckets to surface co-occurring commercial signals.</div>',
        unsafe_allow_html=True,
    )

    rules = load_csv("association_rules.csv")
    if rules.empty:
        st.warning("No rules generated. Run the association rules step.")
        st.stop()

    avg_conf = float(rules["confidence"].mean()) if "confidence" in rules.columns else np.nan
    avg_lift = float(rules["lift"].mean()) if "lift" in rules.columns else np.nan
    avg_support = float(rules["support"].mean()) if "support" in rules.columns else np.nan
    strong_rules = (
        int(((rules["lift"] >= 2.0) & (rules["confidence"] >= 0.6)).sum())
        if all(c in rules.columns for c in ["lift", "confidence"])
        else 0
    )
    strongest_rule = (
        rules.sort_values("lift", ascending=False).iloc[0]
        if "lift" in rules.columns and not rules.empty
        else None
    )
    strongest_lift = float(strongest_rule.get("lift", 0.0)) if strongest_rule is not None else 0.0

    highlights_html = []
    if strongest_rule is not None:
        best_rules = rules.sort_values("lift", ascending=False).head(4)
        for _, row in best_rules.iterrows():
            highlights_html.append(
                f'<div class="rules-highlight">'
                f'<div class="rules-highlight-main"><strong>{html.escape(safe_text(row.get("antecedents"), "antecedent"))}</strong> → <strong>{html.escape(safe_text(row.get("consequents"), "consequent"))}</strong></div>'
                f'<div class="rules-highlight-meta">lift {float(row.get("lift", 0.0)):.2f} · confidence {float(row.get("confidence", 0.0)):.2f} · support {float(row.get("support", 0.0)):.2f}</div>'
                f"</div>"
            )

    st.markdown(
        f"""
        <div class="rules-shell">
            <div class="rules-card">
                <div class="rules-kicker">Association mining / co-occurrence structure</div>
                <div class="rules-title">A rule intelligence board for decoding which signals reliably travel together.</div>
                <div class="rules-copy">
                    The mining process currently surfaced <strong>{len(rules)}</strong> rules, with average confidence at <strong>{avg_conf:.2f}</strong> and average lift at <strong>{avg_lift:.2f}</strong>.
                    This page helps separate trivial co-occurrence from rules that are both frequent enough and strong enough to influence strategy.
                </div>
                <div class="rules-mini-grid">
                    <div class="rules-mini">
                        <div class="rules-mini-label">Total rules</div>
                        <div class="rules-mini-value">{len(rules)}</div>
                        <div class="rules-mini-copy">All generated rules after support/confidence thresholds.</div>
                    </div>
                    <div class="rules-mini">
                        <div class="rules-mini-label">Strong rules</div>
                        <div class="rules-mini-value">{strong_rules}</div>
                        <div class="rules-mini-copy">Rules with lift ≥ 2.0 and confidence ≥ 0.6 in the current output.</div>
                    </div>
                    <div class="rules-mini">
                        <div class="rules-mini-label">Average support</div>
                        <div class="rules-mini-value">{avg_support:.2f}</div>
                        <div class="rules-mini-copy">Average prevalence of antecedent→consequent rule pairs.</div>
                    </div>
                    <div class="rules-mini">
                        <div class="rules-mini-label">Best lift</div>
                        <div class="rules-mini-value">{strongest_lift:.2f}</div>
                        <div class="rules-mini-copy">Highest observed association strength above random expectation.</div>
                    </div>
                </div>
            </div>
            <div class="rules-card">
                <div class="rules-kicker">Top high-lift rules</div>
                <div class="rules-highlight-list">{"".join(highlights_html) if highlights_html else '<div class="rules-note-copy">Rule highlights become available after mining outputs are generated.</div>'}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Rules", f"{len(rules):,}")
    c2.metric(
        "Avg Confidence",
        f"{rules['confidence'].mean():.2f}" if "confidence" in rules.columns else "—",
    )
    c3.metric("Avg Lift", f"{rules['lift'].mean():.2f}" if "lift" in rules.columns else "—")

    st.markdown(
        f"""
        <div class="rules-note-grid">
            <div class="rules-note">
                <div class="rules-note-title">Signal strength</div>
                <div class="rules-note-copy">The strongest rule currently has lift <strong>{strongest_lift:.2f}</strong>, which is a clear indicator of non-random co-occurrence.</div>
            </div>
            <div class="rules-note">
                <div class="rules-note-title">Reliability</div>
                <div class="rules-note-copy">Average confidence is <strong>{avg_conf:.2f}</strong>, meaning the consequent is frequently observed when the antecedent appears.</div>
            </div>
            <div class="rules-note">
                <div class="rules-note-title">Practicality</div>
                <div class="rules-note-copy">With average support at <strong>{avg_support:.2f}</strong>, many rules are not just strong but also common enough to be operationally useful.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["Rule Map", "High-Lift View", "Rules Table"])

    with tab1:
        if all(c in rules.columns for c in ["support", "confidence", "lift"]):
            fig = px.scatter(
                rules.head(300),
                x="support",
                y="confidence",
                size="lift",
                color="lift",
                color_continuous_scale=[[0, C["primary"]], [0.5, C["accent"]], [1, C["success"]]],
                hover_data=[c for c in ["antecedents", "consequents"] if c in rules.columns],
                opacity=0.7,
            )
            fig.update_traces(marker=dict(line=dict(width=0.3, color="rgba(0,0,0,0.3)")))
            fig = apply_theme(fig, "Support vs Confidence (size = lift)", 480)
            st.plotly_chart(fig, width="stretch")

    with tab2:
        if all(c in rules.columns for c in ["support", "confidence", "lift"]):
            hi = rules[(rules["lift"] >= rules["lift"].quantile(0.75))].copy()
            hi = hi.sort_values(["lift", "confidence"], ascending=[False, False]).head(60)
            fig = px.bar(
                hi,
                x="lift",
                y="antecedents",
                color="confidence",
                orientation="h",
                color_continuous_scale=[
                    [0, C["secondary"]],
                    [0.5, C["warning"]],
                    [1, C["success"]],
                ],
                hover_data=[c for c in ["consequents", "support"] if c in hi.columns],
            )
            fig.update_layout(
                coloraxis_showscale=False, xaxis_title="Lift", yaxis_title="Antecedent"
            )
            fig = apply_theme(fig, "High-lift rule concentration", 520)
            st.plotly_chart(fig, width="stretch")

    with tab3:
        cols = [
            c
            for c in [
                "antecedents",
                "consequents",
                "support",
                "confidence",
                "lift",
                "leverage",
                "conviction",
            ]
            if c in rules.columns
        ]
        sorted_r = rules.sort_values("lift", ascending=False) if "lift" in rules.columns else rules
        table_df = sorted_r[cols].head(80) if cols else sorted_r.head(80)
        st.dataframe(
            table_df.style.format(
                {
                    "support": "{:.3f}",
                    "confidence": "{:.3f}",
                    "lift": "{:.2f}",
                    "leverage": "{:.3f}",
                    "conviction": "{:.2f}",
                }
            )
            if cols
            else table_df,
            height=520,
        )
        st.download_button(
            "Download rules table as CSV",
            data=sorted_r.to_csv(index=False).encode("utf-8"),
            file_name="association_rules_ranked.csv",
            mime="text/csv",
        )


# ══════════════════════════════════════════════════════════════
# LLM INSIGHTS
# ══════════════════════════════════════════════════════════════
elif page == "LLM Insights":
    st.markdown('<div class="page-title">LLM Intelligence Hub</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Narrative BI, guided strategy generation, and governed chat assistance layered on top of curated analytics artifacts.</div>',
        unsafe_allow_html=True,
    )

    topk = load_csv("topk_products.csv")
    analytics_assets = set(mcp.list_analytics())
    strategic_assets = [
        "topk_products.csv",
        "topk_per_shop.csv",
        "topk_per_category.csv",
        "model_metrics.json",
        "model_metrics_xgboost.json",
        "association_rules.csv",
    ]
    asset_coverage = sum(name in analytics_assets for name in strategic_assets)
    topk_rows = len(topk)
    category_coverage = int(topk["category"].nunique()) if "category" in topk.columns else 0
    shop_coverage = int(topk["shop_name"].nunique()) if "shop_name" in topk.columns else 0
    score_coverage = (
        float(topk["score"].notna().mean()) * 100
        if "score" in topk.columns and topk_rows > 0
        else 0.0
    )
    llm_actions = [
        (
            "Executive Summary",
            "Fast narrative briefing over validated outputs",
        ),
        (
            "Strategic Recommendations",
            "Category and shop-level strategic direction",
        ),
        (
            "Competitive Profiling",
            "Top-product profile generation for positioning",
        ),
    ]
    action_html = []
    for action_name, action_meta in llm_actions:
        action_html.append(
            f'<div class="llm-action">'
            f'<div class="llm-action-main">{html.escape(action_name)}</div>'
            f'<div class="llm-action-meta">{html.escape(action_meta)}</div>'
            f"</div>"
        )

    st.markdown(
        f"""
        <div class="llm-shell">
            <div class="llm-card">
                <div class="llm-kicker">LLM layer / governed analytics narrative</div>
                <div class="llm-title">A decision-assistant surface that prioritizes insight quality and data boundary control.</div>
                <div class="llm-copy">
                    The assistant currently reads curated analytics assets rather than raw catalog payloads, with <strong>{asset_coverage}/{len(strategic_assets)}</strong> strategic files available for narrative generation.
                    This page is designed to deliver executive synthesis and exploratory Q&amp;A while preserving strict separation between reporting features and sensitive raw data.
                </div>
                <div class="llm-mini-grid">
                    <div class="llm-mini">
                        <div class="llm-mini-label">Strategic assets</div>
                        <div class="llm-mini-value">{asset_coverage}/{len(strategic_assets)}</div>
                        <div class="llm-mini-copy">Coverage of core ranking, model, and rule outputs used for LLM narratives.</div>
                    </div>
                    <div class="llm-mini">
                        <div class="llm-mini-label">Top-K rows</div>
                        <div class="llm-mini-value">{topk_rows}</div>
                        <div class="llm-mini-copy">Ranked products available to power executive and tactical prompts.</div>
                    </div>
                    <div class="llm-mini">
                        <div class="llm-mini-label">Category span</div>
                        <div class="llm-mini-value">{category_coverage}</div>
                        <div class="llm-mini-copy">Distinct categories represented in the current Top-K artifact.</div>
                    </div>
                    <div class="llm-mini">
                        <div class="llm-mini-label">Score coverage</div>
                        <div class="llm-mini-value">{score_coverage:.1f}%</div>
                        <div class="llm-mini-copy">Share of Top-K rows carrying usable score values for reasoning.</div>
                    </div>
                </div>
            </div>
            <div class="llm-card">
                <div class="llm-kicker">Available LLM actions</div>
                <div class="llm-copy" style="margin-bottom: 14px;">The assistant currently ships with three guided report-generation actions for fast decision support.</div>
                <div class="llm-action-list">{"".join(action_html)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="llm-note-grid">
            <div class="llm-note">
                <div class="llm-note-title">Data boundary</div>
                <div class="llm-note-copy">The LLM layer operates on <strong>curated analytics outputs</strong>, not full raw catalog dumps, which reduces leakage risk while preserving analytical usefulness.</div>
            </div>
            <div class="llm-note">
                <div class="llm-note-title">Analytical breadth</div>
                <div class="llm-note-copy">Current Top-K coverage spans <strong>{shop_coverage}</strong> shops and <strong>{category_coverage}</strong> categories, giving the assistant meaningful commercial breadth.</div>
            </div>
            <div class="llm-note">
                <div class="llm-note-title">Operational posture</div>
                <div class="llm-note-copy">The MCP architecture keeps capabilities read-only for analytics and logging, with no score mutation or arbitrary code execution path.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_report, tab_chat, tab_mcp = st.tabs(
        ["Strategic Reports", "BI Assistant (Chat)", "MCP Architecture"]
    )

    with tab_report:
        st.markdown(
            """<div class="insight-banner">
            Report generation is constrained to <strong>Top-K products, aggregate metrics, and model/rule outputs</strong>.
            Raw source pages and unrestricted data exports are intentionally outside the assistant boundary.
            </div>""",
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)

        if c1.button("Executive Summary", use_container_width=True, key="llm_exec_summary"):
            with st.spinner("Calling Gemini API..."):
                st.markdown("### Executive Summary")
                st.info(get_llm_summary())

        if c2.button("Strategic Recommendations", use_container_width=True, key="llm_strategy"):
            with st.spinner("Generating Chain-of-Thought Strategy..."):
                from src.llm.summarizer import generate_strategy_report

                data = {}
                if not topk.empty:
                    ranked_shops = pd.Series(dtype=float)
                    if all(c in topk.columns for c in ["shop_name", "score"]):
                        ranked_shops = (
                            topk.groupby("shop_name")["score"].mean().sort_values(ascending=False)
                        )
                    data = {
                        "top_categories": topk["category"].value_counts().head(5).index.tolist()
                        if "category" in topk.columns
                        else [],
                        "best_shop": ranked_shops.index[0] if not ranked_shops.empty else "",
                    }

                st.markdown("### Marketing Strategy & Trends")
                st.success(generate_strategy_report(data))

        if c3.button("Competitive Profiling", use_container_width=True, key="llm_profile"):
            with st.spinner("Profiling Top Products..."):
                from src.llm.summarizer import generate_product_profile

                st.markdown("### Customer & Product Profile")
                st.warning(generate_product_profile(mcp.get_top_products(5)))

    with tab_chat:
        st.markdown(
            """<div class="insight-banner">
            Ask focused questions tied to ranking dynamics, category trends, shop performance, or model behavior.
            Responses are grounded in currently available analytics artifacts and active chat context.
            </div>""",
            unsafe_allow_html=True,
        )

        if "messages" not in st.session_state:
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Hello! I am your eCommerce AI Assistant. How can I help you analyze the data today?",
                }
            ]

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ex: What are the 5 emerging products this week?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    from src.llm.summarizer import chat_with_data

                    ctx = {"top_products": mcp.get_top_products(5) if mcp else []}
                    if not topk.empty and "category" in topk.columns:
                        ctx["top_categories"] = topk["category"].value_counts().head(5).to_dict()

                    response = chat_with_data(prompt, ctx, st.session_state.messages)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

    with tab_mcp:
        st.markdown(
            """<div class="insight-banner">
            MCP defines explicit roles and permissions: dashboard host, controlled client routing,
            read-only analytics access, structured LLM calls, and append-only usage/audit logging.
            </div>""",
            unsafe_allow_html=True,
        )

        st.markdown(
            """
| Concept | Implementation |
|---------|---------------|
| **Host** | This Streamlit dashboard |
| **Client** | `MCPClient` routes requests to servers |
| **AnalyticsReader** | Read-only access to whitelisted CSV/JSON files, plus secure `get_top_products()` |
| **SummaryGenerator** | LLM calls with structured input only |
| **Permissions** | No raw data, no code execution, no score modification |
| **Audit Logs** | `llm_usage_log.jsonl`, `mcp_access_log.jsonl` |
            """
        )

        with st.expander("LLM Usage Log", expanded=False):
            from src.config import analytics_dir as _adir

            lp = _adir() / "llm_usage_log.jsonl"
            if lp.exists():
                raw_log = lp.read_text(encoding="utf-8")
                records = parse_jsonl_loose(raw_log)
                if records:
                    log_df = pd.DataFrame(records)
                    preferred_cols = [
                        c
                        for c in [
                            "timestamp",
                            "source",
                            "action",
                            "resource",
                            "prompt_preview",
                            "response_preview",
                            "detail",
                        ]
                        if c in log_df.columns
                    ]
                    if preferred_cols:
                        log_df = log_df[preferred_cols]
                    st.dataframe(log_df.tail(25), height=320)
                else:
                    st.warning("LLM usage log exists but could not be parsed as JSON records.")
            else:
                st.info("No LLM usage log yet.")
