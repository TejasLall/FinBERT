"""
Automated Daily Newspaper Financial Sentiment & Market Impact Dashboard
=======================================================================

A single-file Streamlit app that:
  1. Ingests a whole day's financial newspaper / news roundup (PDF, TXT or CSV)
     or a single headline typed by the user.
  2. Extracts and cleans the text, then splits it into individual articles.
  3. Runs FinBERT (ProsusAI/finbert) in *batches* to classify each article as
     Positive / Negative / Neutral with a confidence score.
  4. Extracts stock tickers (e.g. $TSLA, AAPL, RELIANCE, NIFTY) with regex.
  5. Aggregates everything into a daily "market mood" dashboard using Plotly.

End-to-end pipeline (useful slide for the viva):

    Upload ─► Text extraction ─► Cleaning ─► Article splitting ─► Tokenization
          ─► Batched FinBERT inference ─► Softmax probabilities ─► Aggregation
          ─► Ticker mapping ─► Interactive dashboard

Run with:  streamlit run app.py
"""

from __future__ import annotations

import bisect
import hashlib
import html
import io
import os
os.environ["RAPIDOCR_CACHE_DIR"] = "/tmp/.rapidocr"
os.environ["XDG_CACHE_HOME"] = "/tmp/.cache"
import math
import re
import threading
import time
from collections import Counter
from functools import lru_cache
from importlib.util import find_spec
from typing import NamedTuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# PDF engines are optional at import time so the app still starts (and can
# analyse TXT/CSV files) even if one of them is missing.
try:
    import pdfplumber
except ImportError:  # pragma: no cover
    pdfplumber = None

try:  # `pypdf` is the maintained successor of PyPDF2; use whichever exists.
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        PdfReader = None

try:  # PyMuPDF: text blocks with positions, and page rendering for OCR.
    import pymupdf
except ImportError:  # pragma: no cover
    try:
        import fitz as pymupdf  # older PyMuPDF releases
    except ImportError:
        pymupdf = None


# =============================================================================
# 1. CONFIGURATION & CONSTANTS
# =============================================================================

MODEL_NAME = "ProsusAI/finbert"

# BERT-family models have a fixed maximum input length of 512 *tokens*
# (sub-word pieces, not words). Anything longer must be truncated.
MAX_TOKENS = 512

# ~300 English words ≈ 400-450 WordPiece tokens, which safely fits inside the
# 512-token window. Longer articles are split into chunks of this size so that
# truncation never silently throws away half an article.
MAX_WORDS_PER_CHUNK = 300

SENTIMENTS = ["Positive", "Negative", "Neutral"]

# Dual-tone palette: the single accent green marks positive news; negative and
# neutral sit on a light → dark grey scale. The three also differ clearly in
# lightness, so they stay distinguishable for colour-blind viewers. One fixed
# colour per sentiment, reused by every chart and card.
ACCENT = "#4ade80"
SENTIMENT_COLORS = {"Positive": ACCENT, "Negative": "#e4e4e7", "Neutral": "#52525b"}

# Shape cues so sentiment is never communicated by colour alone (accessibility).
SENTIMENT_ICONS = {"Positive": "▲", "Negative": "▼", "Neutral": "●"}
SENTIMENT_LABELS = {s: f"{SENTIMENT_ICONS[s]} {s}" for s in SENTIMENTS}

MODE_UPLOAD = "Analyze a newspaper"
MODE_QUICK = "Test a single headline"

SPLIT_AUTO = "Auto-detect"
SPLIT_BLOCKS = "Headline / paragraph blocks"
SPLIT_LINES = "One item per line"

PDF_MODE_LAYOUT = "Layout-aware (OCR when needed)"
PDF_MODE_OCR = "Always OCR"
PDF_MODE_PLAIN = "Plain text flow (old)"

# Column names we look for when a CSV is uploaded (first match wins).
TEXT_COLUMN_HINTS = ["headline", "title", "text", "content", "article", "summary", "news"]

# --- Ticker dictionaries -----------------------------------------------------
# Bare upper-case words are only accepted as tickers if they appear in this
# whitelist. Without it, words like CEO, GDP, RBI, IPO or USA would all be
# mistaken for stock symbols (high false-positive rate).
KNOWN_TICKERS = {
    # US large caps
    "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META", "NVDA", "TSLA", "NFLX",
    "AMD", "INTC", "IBM", "ORCL", "CRM", "ADBE", "JPM", "BAC", "WFC", "C",
    "WMT", "DIS", "XOM", "CVX", "PFE", "JNJ", "PEP", "UBER", "PYPL", "COIN",
    "PLTR", "BRK.B",
    # Indian large caps (NSE symbols)
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "ITC",
    "BHARTIARTL", "WIPRO", "HCLTECH", "KOTAKBANK", "AXISBANK", "MARUTI",
    "TATAMOTORS", "TATASTEEL", "ADANIENT", "ADANIPORTS", "SUNPHARMA",
    "BAJFINANCE", "ASIANPAINT", "HINDUNILVR", "ONGC", "NTPC", "POWERGRID",
    "COALINDIA", "ZOMATO", "PAYTM", "LICI", "M&M",
    # Indices
    "NIFTY", "BANKNIFTY", "SENSEX", "NASDAQ", "DOW", "SPX", "FTSE", "NIKKEI", "DAX",
}
KNOWN_TICKERS.discard("C")  # single letters are far too ambiguous in prose

# Company / index names as they appear in running text → canonical ticker.
COMPANY_ALIASES = {
    "Apple": "AAPL", "Microsoft": "MSFT", "Amazon": "AMZN", "Alphabet": "GOOGL",
    "Google": "GOOGL", "Meta Platforms": "META", "Meta": "META", "Facebook": "META",
    "Nvidia": "NVDA", "Tesla": "TSLA", "Netflix": "NFLX", "Intel": "INTC",
    "Oracle": "ORCL", "Salesforce": "CRM", "Adobe": "ADBE", "JPMorgan": "JPM",
    "JP Morgan": "JPM", "Bank of America": "BAC", "Wells Fargo": "WFC",
    "Walmart": "WMT", "Disney": "DIS", "Exxon": "XOM", "ExxonMobil": "XOM",
    "Chevron": "CVX", "Pfizer": "PFE", "Johnson & Johnson": "JNJ", "PepsiCo": "PEP",
    "PayPal": "PYPL", "Coinbase": "COIN", "Palantir": "PLTR",
    "Berkshire Hathaway": "BRK.B",
    "Reliance Industries": "RELIANCE", "Reliance": "RELIANCE",
    "Tata Consultancy Services": "TCS", "Infosys": "INFY",
    "HDFC Bank": "HDFCBANK", "ICICI Bank": "ICICIBANK",
    "State Bank of India": "SBIN", "SBI": "SBIN", "Bharti Airtel": "BHARTIARTL",
    "Airtel": "BHARTIARTL", "Wipro": "WIPRO", "HCL Technologies": "HCLTECH",
    "HCLTech": "HCLTECH", "Kotak Mahindra Bank": "KOTAKBANK", "Axis Bank": "AXISBANK",
    "Maruti Suzuki": "MARUTI", "Maruti": "MARUTI", "Tata Motors": "TATAMOTORS",
    "Tata Steel": "TATASTEEL", "Adani Enterprises": "ADANIENT",
    "Adani Ports": "ADANIPORTS", "Sun Pharma": "SUNPHARMA",
    "Bajaj Finance": "BAJFINANCE", "Asian Paints": "ASIANPAINT",
    "Hindustan Unilever": "HINDUNILVR", "Coal India": "COALINDIA",
    "Zomato": "ZOMATO", "Paytm": "PAYTM", "Mahindra & Mahindra": "M&M",
    "Bank Nifty": "BANKNIFTY", "Nifty Bank": "BANKNIFTY", "Nifty 50": "NIFTY",
    "Nifty": "NIFTY", "Sensex": "SENSEX", "Nasdaq": "NASDAQ",
    "Dow Jones": "DOW", "Dow": "DOW", "S&P 500": "SPX", "S&P": "SPX",
    "FTSE 100": "FTSE", "Nikkei": "NIKKEI",
}
KNOWN_TICKERS |= set(COMPANY_ALIASES.values())


# --- Design system -----------------------------------------------------------
# A minimalist dark theme: near-black canvas, white type, one accent colour.
# The same values drive the CSS, the HTML cards and the Plotly figures.
CARD_BG = "#111111"  # bento card surface (also used for the gaps between chart segments)
TEXT = "#fafafa"
MUTED = "#a1a1aa"
FAINT = "#71717a"
FONT_STACK = "Inter, 'Helvetica Neue', Helvetica, Arial, sans-serif"

# Shared Plotly layout: transparent canvas so the card surface shows through,
# Inter type, dark hover labels and a 500 ms eased transition between states.
CHART_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family=FONT_STACK, color=MUTED, size=12),
    transition=dict(duration=500, easing="cubic-in-out"),
    hoverlabel=dict(
        bgcolor="#1c1c1e",
        bordercolor="rgba(255,255,255,0.12)",
        font=dict(family=FONT_STACK, color=TEXT, size=12),
    ),
)
# No grid lines, zero lines, spines or tick marks: just the data and its labels.
CLEAN_AXIS = dict(showgrid=False, zeroline=False, showline=False, ticks="")

# Injected once per run with st.html (a <style>-only block takes no layout
# space). Native widgets are themed by .streamlit/config.toml; this CSS adds
# the fintech layer on top: Inter, the bento cards and the micro-interactions.
BENTO_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg: #0a0a0a;
    --card: #111111;
    --border: rgba(255, 255, 255, 0.1);
    --border-strong: rgba(255, 255, 255, 0.2);
    --text: #fafafa;
    --muted: #a1a1aa;
    --faint: #71717a;
    --accent: #4ade80;
    --accent-soft: rgba(74, 222, 128, 0.08);
    --radius: 24px;
}

/* ---- Typography & canvas ------------------------------------------------ */
html, body, .stApp {
    font-family: 'Inter', 'Helvetica Neue', Helvetica, Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}
.stApp {
    background:
        radial-gradient(1100px 460px at 50% -180px, rgba(255, 255, 255, 0.07), transparent 70%),
        var(--bg);
    color: var(--text);
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"], .block-container { max-width: 1240px; padding-top: 2.5rem; }
[data-testid="stSidebar"] { background: #070707; border-right: 1px solid var(--border); }

/* ---- Hero ---------------------------------------------------------------- */
.hero { padding: 0 0 0.5rem; }
.hero-eyebrow { color: var(--muted); font-size: 0.82rem; font-weight: 500; margin: 1.4rem 0 0.8rem; }
.hero-title {
    margin: 0; padding: 0;
    font-size: clamp(2.4rem, 5.6vw, 4.2rem);
    font-weight: 600; line-height: 1.02; letter-spacing: -0.045em;
    background: linear-gradient(180deg, #ffffff 35%, #8b8b94 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero-sub { color: var(--muted); font-size: 1.05rem; line-height: 1.6; max-width: 620px; margin: 1.1rem 0 0; }

/* ---- Live model badge (pulsing dot) -------------------------------------- */
.live-pill {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 6px 12px 6px 10px; border-radius: 999px;
    border: 1px solid var(--border); background: rgba(255, 255, 255, 0.03);
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase;
    color: var(--text); white-space: nowrap;
}
.live-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--accent); animation: pulse-glow 2s ease-in-out infinite; }
.live-pill.off { color: var(--muted); }
.live-pill.off .live-dot { background: var(--faint); animation: none; }
@keyframes pulse-glow {
    0%   { box-shadow: 0 0 0 0 rgba(74, 222, 128, 0.55); }
    70%  { box-shadow: 0 0 0 8px rgba(74, 222, 128, 0); }
    100% { box-shadow: 0 0 0 0 rgba(74, 222, 128, 0); }
}

/* ---- Section labels ------------------------------------------------------ */
.section-label {
    display: flex; align-items: baseline; justify-content: space-between; flex-wrap: wrap; gap: 4px 16px;
    margin: 2rem 0 0.1rem;
}
.section-label .sl-title { font-size: 0.74rem; font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted); }
.section-label .sl-meta { font-size: 0.8rem; color: var(--faint); font-variant-numeric: tabular-nums; }

/* ---- Bento containers (native Streamlit elements inside) ------------------
   st.container(key="bento-…") renders with the class .st-key-bento-….
   The lift uses `top`, not `transform`: a transformed ancestor would trap
   Streamlit's fullscreen chart/table view and offset Plotly's hover. */
[class*="st-key-bento-"] {
    position: relative; top: 0;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
    transition: all 0.3s ease;
}
[class*="st-key-bento-"]:hover {
    top: -4px;
    border-color: var(--border-strong);
    box-shadow: 0 30px 60px -30px rgba(0, 0, 0, 0.9), 0 0 0 1px rgba(255, 255, 255, 0.02);
}
.card-eyebrow { font-size: 0.7rem; font-weight: 600; letter-spacing: 0.13em; text-transform: uppercase; color: var(--faint); }
.card-title { font-size: 1.15rem; font-weight: 600; letter-spacing: -0.02em; color: var(--text); margin-top: 4px; }
.card-note { font-size: 0.86rem; color: var(--muted); margin-top: 4px; line-height: 1.5; }
.empty-state { color: var(--muted); font-size: 0.9rem; line-height: 1.6; padding: 12px 0; }
.method-list { list-style: none; padding: 0 !important; margin: 0 !important; display: grid; gap: 14px; }
.method-list li { font-size: 0.86rem; color: var(--muted); line-height: 1.5; margin: 0 !important; padding: 0 !important; }
.method-list b { display: block; color: var(--text); font-weight: 600; margin-bottom: 2px; }

/* ---- Bento KPI grid (pure HTML cards) ------------------------------------ */
.bento-grid {
    display: grid; gap: 16px;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    grid-template-areas: "hero hero a b" "hero hero c d";
}
.bento-grid.single-row { grid-template-columns: 2fr 1fr 1fr 1fr; grid-template-areas: "hero a b c"; }
.area-hero { grid-area: hero; } .area-a { grid-area: a; } .area-b { grid-area: b; }
.area-c { grid-area: c; } .area-d { grid-area: d; }
.bento-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
    min-height: 150px;
    display: flex; flex-direction: column; justify-content: space-between; gap: 18px;
    transition: all 0.3s ease;
}
.bento-card:hover {
    transform: scale(1.02);
    border-color: var(--border-strong);
    box-shadow: 0 30px 60px -30px rgba(0, 0, 0, 0.9);
}
.bento-card.hero {
    padding: 32px;
    background: radial-gradient(520px 260px at 0% 0%, rgba(74, 222, 128, 0.1), transparent 65%), var(--card);
}
.kpi-label { font-size: 0.8rem; font-weight: 500; color: var(--muted); }
.kpi-value {
    margin-top: 10px;
    font-size: 2.1rem; font-weight: 600; letter-spacing: -0.035em; line-height: 1.05;
    color: var(--text); font-variant-numeric: tabular-nums; overflow-wrap: anywhere;
}
.bento-card.hero .kpi-value { font-size: clamp(2.6rem, 5vw, 4.2rem); letter-spacing: -0.05em; }
.kpi-sub { margin-top: 8px; font-size: 0.82rem; color: var(--faint); font-variant-numeric: tabular-nums; }
.kpi-sub.up { color: var(--accent); }
.kpi-sub.down { color: #e4e4e7; }
.mood-bar { display: flex; gap: 3px; height: 8px; border-radius: 999px; overflow: hidden; }
.mood-seg { height: 100%; min-width: 4px; }
.mood-legend { display: flex; flex-wrap: wrap; gap: 6px 18px; margin-top: 12px; font-size: 0.8rem; color: var(--muted); }
.mood-legend i { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
.mood-legend b { color: var(--text); font-weight: 600; margin-left: 5px; font-variant-numeric: tabular-nums; }

/* Welcome "how it works" cards */
.steps-grid { display: grid; gap: 16px; grid-template-columns: repeat(3, minmax(0, 1fr)); }
.step-num { font-size: 0.8rem; font-weight: 600; color: var(--accent); font-variant-numeric: tabular-nums; }
.step-title { font-size: 1.15rem; font-weight: 600; letter-spacing: -0.02em; color: var(--text); }
.step-body { margin-top: 6px; font-size: 0.88rem; line-height: 1.55; color: var(--muted); }

@media (max-width: 1000px) {
    .bento-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); grid-template-areas: "hero hero" "a b" "c d"; }
    .bento-grid.single-row { grid-template-columns: repeat(3, minmax(0, 1fr)); grid-template-areas: "hero hero hero" "a b c"; }
}
@media (max-width: 760px) { .steps-grid { grid-template-columns: 1fr; } }
@media (max-width: 560px) {
    .bento-grid { grid-template-columns: 1fr; grid-template-areas: "hero" "a" "b" "c" "d"; }
    .bento-grid.single-row { grid-template-columns: 1fr; grid-template-areas: "hero" "a" "b" "c"; }
}

/* ---- Buttons: white pill primary, outline secondary ---------------------- */
.stButton button[kind="primary"],
[data-testid="stBaseButton-primary"] {
    background: #fafafa !important;
    border: 1px solid #fafafa !important;
    color: #0a0a0a !important;
    border-radius: 999px;
    font-weight: 600;
    padding: 0.55rem 1.3rem;
    transition: all 0.3s ease;
}
.stButton button[kind="primary"] *,
[data-testid="stBaseButton-primary"] * { color: #0a0a0a !important; }
.stButton button[kind="primary"]:hover,
[data-testid="stBaseButton-primary"]:hover {
    background: #ffffff !important;
    transform: scale(1.02);
    box-shadow: 0 0 0 6px rgba(255, 255, 255, 0.08);
}
[data-testid="stBaseButton-secondary"],
.stDownloadButton button {
    border-radius: 999px;
    background: transparent;
    border: 1px solid var(--border-strong);
    transition: all 0.3s ease;
}
[data-testid="stBaseButton-secondary"]:hover,
.stDownloadButton button:hover {
    border-color: var(--text);
    color: var(--text);
    transform: scale(1.02);
}

/* ---- File uploader ------------------------------------------------------- */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(255, 255, 255, 0.02);
    border: 1px dashed rgba(255, 255, 255, 0.16);
    border-radius: 18px;
    padding: 1.6rem;
    transition: all 0.3s ease;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--accent);
    background: var(--accent-soft);
    transform: scale(1.01);
}

/* ---- Multiselect chips: neutral pills (white on the green accent is hard to read) */
[data-baseweb="tag"] {
    background: #262626 !important;
    border: 1px solid var(--border-strong);
    border-radius: 999px !important;
}
[data-baseweb="tag"] span, [data-baseweb="tag"] svg { color: var(--text) !important; fill: var(--text) !important; }

/* ---- Expanders & tabs ---------------------------------------------------- */
[data-testid="stExpander"] details { border: 1px solid var(--border); border-radius: 16px; background: transparent; }
[data-baseweb="tab-list"] { gap: 4px; }

/* Respect the OS "reduce motion" setting. */
@media (prefers-reduced-motion: reduce) {
    .live-dot { animation: none; }
    .bento-card, [class*="st-key-bento-"], [data-testid="stFileUploaderDropzone"],
    .stButton button, .stDownloadButton button { transition: none; }
    .bento-card:hover, [data-testid="stFileUploaderDropzone"]:hover,
    .stButton button:hover, .stDownloadButton button:hover { transform: none; }
    [class*="st-key-bento-"]:hover { top: 0; }
}
</style>
"""


def inject_css() -> None:
    st.html(BENTO_CSS)


def esc(text) -> str:
    """HTML-escape any value before it goes into a custom HTML snippet."""
    return html.escape(str(text))


def render_header(model_ok: bool, device: str) -> None:
    """Hero header with a pulsing "LIVE MODEL ACTIVE" badge (grey when offline)."""
    if model_ok:
        pill = f'<span class="live-pill"><span class="live-dot"></span>Live model active · {esc(device)}</span>'
    else:
        pill = '<span class="live-pill off"><span class="live-dot"></span>Model offline</span>'
    st.html(
        '<header class="hero">'
        f"{pill}"
        '<p class="hero-eyebrow">Automated Daily Newspaper · Financial Sentiment &amp; Market Impact</p>'
        '<h1 class="hero-title">Market sentiment,<br>decoded.</h1>'
        '<p class="hero-sub">Drop in a day\'s financial newspaper. FinBERT reads every article, '
        "scores its tone and maps it to the tickers it mentions.</p>"
        "</header>"
    )


def section_label(title: str, meta: str = "") -> None:
    st.html(f'<div class="section-label"><span class="sl-title">{esc(title)}</span><span class="sl-meta">{esc(meta)}</span></div>')


def bento(key: str, stretch: bool = True):
    """
    A rounded "bento" card that can hold native Streamlit elements.

    The key becomes the CSS class .st-key-bento-<key>, which BENTO_CSS styles.
    height="stretch" makes cards placed side by side in st.columns equal height.
    """
    return st.container(key=f"bento-{key}", height="stretch" if stretch else "content")


def card_header(eyebrow: str, title: str, note: str = "") -> None:
    note_html = f'<div class="card-note">{esc(note)}</div>' if note else ""
    st.html(f'<div class="card-eyebrow">{esc(eyebrow)}</div><div class="card-title">{esc(title)}</div>{note_html}')


def empty_state(text: str) -> None:
    st.html(f'<div class="empty-state">{esc(text)}</div>')


def kpi_card(label: str, value: str, sub: str = "", *, area: str, sub_class: str = "",
             extra: str = "", tooltip: str = "") -> str:
    """HTML for one bento KPI card. `extra` is trusted HTML built by this app."""
    hero = " hero" if area == "hero" else ""
    sub_html = f'<div class="kpi-sub {sub_class}">{esc(sub)}</div>' if sub else ""
    return (
        f'<div class="bento-card{hero} area-{area}" title="{esc(tooltip)}">'
        f'<div><div class="kpi-label">{esc(label)}</div><div class="kpi-value">{esc(value)}</div>{sub_html}</div>'
        f"{extra}</div>"
    )


# =============================================================================
# 2. MODEL LOADING (cached once per server process)
# =============================================================================

@st.cache_resource(show_spinner="Loading FinBERT… (first run downloads ~440 MB)")
def load_finbert():
    """
    Load the FinBERT tokenizer + classification model exactly once.

    VIVA NOTE - why @st.cache_resource?
      Streamlit re-executes the whole script on every widget interaction.
      Without caching, the 110M-parameter model would be re-loaded from disk
      on every click. `cache_resource` keeps ONE shared model object in memory
      for all users/sessions (it is meant for unpicklable resources such as
      ML models and DB connections).

    FinBERT = BERT-base further pre-trained on financial text and fine-tuned on
    the Financial PhraseBank dataset for 3-class sentiment classification.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    # Use a GPU if one is available; otherwise fall back to the CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # eval() switches off training-only behaviour such as Dropout, so the same
    # input always produces the same output (deterministic inference).
    model.eval()
    return tokenizer, model, device


# =============================================================================
# 3. BATCHED SENTIMENT INFERENCE (the core ML step)
# =============================================================================

def classify_texts(texts: list[str], batch_size: int = 16, progress_callback=None) -> list[dict]:
    """
    Classify a list of texts with FinBERT using mini-batches.

    Returns one dict per input text (same order as `texts`) containing:
      sentiment, confidence, p_positive, p_negative, p_neutral, sentiment_score

    VIVA NOTE - why batch processing?
      * A transformer does its work as large matrix multiplications. Feeding 16
        texts at once turns 16 small matrix ops into ONE big one, which CPUs
        (SIMD / multi-core) and especially GPUs execute far more efficiently
        than 16 separate forward passes.
      * But we can't push all 500 articles of a newspaper in one go either:
        memory grows with batch_size × sequence_length. Mini-batches are the
        compromise between speed and memory, and they also let us update a
        progress bar after every batch.
    """
    if not texts:
        return []

    tokenizer, model, device = load_finbert()

    # Read the label names from the model config instead of hard-coding
    # "0 = positive, 1 = negative, ...". Different checkpoints order their
    # classes differently; trusting id2label avoids silently swapped labels.
    id2label = {int(i): str(label).capitalize() for i, label in model.config.id2label.items()}

    n = len(texts)
    results: list[dict | None] = [None] * n

    # VIVA NOTE - length-sorted batching (a classic inference optimisation).
    # Within a batch, every sequence is padded to the length of the LONGEST one.
    # If a 10-word headline shares a batch with a 400-word article, the
    # headline gets ~500 useless [PAD] tokens. Sorting by length first groups
    # similar-length texts together, so far less compute is wasted on padding.
    # We remember the original positions and put results back in order.
    order = sorted(range(n), key=lambda i: len(texts[i]))
    n_batches = math.ceil(n / batch_size)

    for b in range(n_batches):
        batch_ids = order[b * batch_size:(b + 1) * batch_size]
        batch_texts = [texts[i] for i in batch_ids]

        # VIVA NOTE - tokenization.
        #   * WordPiece splits text into sub-word tokens, e.g.
        #     "downgraded" → ["down", "##grade", "##d"], and maps each to an id.
        #   * [CLS] is prepended (its final hidden state feeds the classifier)
        #     and [SEP] is appended.
        #   * padding=True       → pad every sequence to the longest in THIS batch
        #                           (dynamic padding, cheaper than always 512).
        #   * truncation=True    → cut anything beyond max_length tokens.
        #   * attention_mask     → 1 for real tokens, 0 for [PAD], so the
        #                           self-attention layers ignore the padding.
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_TOKENS,
            return_tensors="pt",
        ).to(device)

        # VIVA NOTE - torch.no_grad().
        # During inference we never call .backward(), so there is no need to
        # record the computation graph for gradients. Disabling it saves
        # memory and time. (eval() and no_grad() are complementary: eval()
        # changes layer behaviour, no_grad() turns off gradient bookkeeping.)
        with torch.no_grad():
            logits = model(**encoded).logits  # shape: [batch_size, 3]

        # VIVA NOTE - logits → probabilities.
        # Logits are raw, unbounded scores. Softmax exponentiates and
        # normalises them so each row sums to 1: p_i = e^{z_i} / Σ_j e^{z_j}.
        # The arg-max class is the prediction; its probability is the
        # model's confidence.
        probs = torch.softmax(logits, dim=-1).cpu()
        confidences, label_ids = probs.max(dim=-1)

        for row, original_idx in enumerate(batch_ids):
            result = {
                "sentiment": id2label[int(label_ids[row])],
                "confidence": float(confidences[row]),
            }
            for class_id, label in id2label.items():
                result[f"p_{label.lower()}"] = float(probs[row, class_id])

            # A single signed number per article in [-1, +1]:
            #   +1 = certainly positive, -1 = certainly negative, 0 = neutral/mixed.
            # Useful for ranking "most bullish" and "most bearish" news.
            result["sentiment_score"] = result.get("p_positive", 0.0) - result.get("p_negative", 0.0)
            results[original_idx] = result

        if progress_callback is not None:
            progress_callback(b + 1, n_batches, min((b + 1) * batch_size, n), n)

    return results  # type: ignore[return-value]


# =============================================================================
# 4. TEXT EXTRACTION, CLEANING & ARTICLE SPLITTING
# =============================================================================

@st.cache_data(show_spinner="Extracting text from PDF…")
def extract_pdf_pages(data: bytes) -> tuple[list[str], str]:
    """
    Return (list_of_page_texts, engine_name).

    pdfplumber is tried first; PyPDF2/pypdf is the fallback if pdfplumber is
    missing, crashes on a malformed file, or returns no text.

    `use_text_flow=True` makes pdfplumber follow the order in which text was
    written into the PDF. For multi-column newspaper layouts this usually
    keeps each column (and therefore each article) together, instead of
    reading straight across all columns line by line.
    """
    pages: list[str] = []

    if pdfplumber is not None:
        try:
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                for page in pdf.pages:
                    try:
                        pages.append(page.extract_text(use_text_flow=True) or "")
                    except TypeError:  # very old pdfplumber without that option
                        pages.append(page.extract_text() or "")
            if sum(len(p.strip()) for p in pages) > 20:
                return pages, "pdfplumber"
        except Exception:
            pages = []

    if PdfReader is not None:
        reader = PdfReader(io.BytesIO(data))
        pages = [(page.extract_text() or "") for page in reader.pages]
        return pages, "PyPDF2"

    if pdfplumber is None:
        raise RuntimeError("No PDF library installed. Run: pip install pdfplumber PyPDF2")
    return pages, "pdfplumber"


def decode_text_file(data: bytes) -> str:
    """Decode an uploaded .txt, tolerating Windows / Latin-1 encoded files."""
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1", errors="replace")


@st.cache_data(show_spinner=False)
def read_csv_bytes(data: bytes) -> pd.DataFrame:
    """Read a CSV, auto-detecting the delimiter (comma, semicolon, tab…)."""
    try:
        return pd.read_csv(io.BytesIO(data), sep=None, engine="python", encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(io.BytesIO(data), sep=None, engine="python", encoding="latin-1")


def detect_text_column(df: pd.DataFrame):
    """Guess which CSV column holds the news text."""
    candidates = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    for hint in TEXT_COLUMN_HINTS:
        for col in candidates:
            if hint in str(col).lower():
                return col
    if candidates:  # otherwise: the column with the longest average text
        return max(candidates, key=lambda c: df[c].astype(str).str.len().mean())
    return None


# --- Newspaper noise patterns ------------------------------------------------
# A short line matching any of these is layout furniture, not news.
_WEEKDAY = r"(?:mon|tues|wednes|thurs|fri|satur|sun)day"
_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
_NOISE_LINE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"^(?:page\s*)?\d{1,3}(?:\s*(?:of|/)\s*\d{1,3})?$",               # "12", "Page 3 of 20"
        r"^(?:continued|contd\.?)\s+(?:on|from)\s+(?:page|p\.?)\s*\d+",  # jump lines
        r"(?:©|\(c\)\s*\d{4}|copyright|all rights reserved)",             # copyright footers
        r"^(?:subscribe|follow us|download (?:the|our) app|advertisement|advt\.?|sponsored|e-?paper)\b",
        rf"^{_WEEKDAY},?\s+(?:{_MONTH}\s+\d{{1,2}}|\d{{1,2}}\s+{_MONTH}),?\s+\d{{4}}",  # datelines
        rf"^\d{{1,2}}\s+{_MONTH},?\s+\d{{4}}$",                          # "13 October 2026"
        r"^vol(?:ume)?\.?\s*[\divxlc]+\b.*\bno\.?\s*\d+",                 # "Vol. XII No. 245"
        r"^(?:price|pages?)\s*:?\s*(?:rs\.?|₹|\$)?\s*\d",                 # masthead price / page count
    ]
]
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_BULLET_RE = re.compile(r"^[\s•▪■◆►\-–—*]+")
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")  # "infla-\ntion" → "inflation"
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


def clean_inline(text: str) -> str:
    """Remove URLs, e-mails, leading bullets and collapse whitespace."""
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _BULLET_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_line(line: str) -> str:
    """Clean one line; return "" if the whole line is newspaper noise."""
    line = clean_inline(line)
    if not re.search(r"[^\W_]", line):  # nothing but punctuation / separators left
        return ""
    # Only short lines are tested against the noise rules, so a real sentence
    # that happens to contain e.g. "copyright" is never thrown away.
    if line and len(line.split()) <= 12 and any(p.search(line) for p in _NOISE_LINE_PATTERNS):
        return ""
    return line


def prepare_lines(raw_text: str) -> list[str]:
    """Raw page text → cleaned lines ("" marks a paragraph break / removed noise)."""
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)  # re-join words hyphenated across lines
    return [clean_line(line) for line in text.split("\n")]


def remove_running_headers(pages_lines: list[list[str]]) -> list[list[str]]:
    """
    Drop running headers/footers: lines that repeat on more than half the pages
    (e.g. "THE FINANCIAL DAILY | Page 4", "Markets", the paper's masthead).
    Digits are masked so "Page 3" and "Page 4" count as the same line.
    """
    repeated = _repeated_line_keys(pages_lines)
    if not repeated:
        return pages_lines
    return [[("" if l and _repeat_key(l) in repeated else l) for l in lines] for lines in pages_lines]


def _repeat_key(line: str) -> str:
    return re.sub(r"\d+", "#", line.lower()).strip()


def _repeated_line_keys(pages_lines: list[list[str]]) -> set[str]:
    """Normalised lines that appear on more than half of the pages (at least 2)."""
    n_pages = len(pages_lines)
    if n_pages < 2:
        return set()
    page_counts: Counter = Counter()
    for lines in pages_lines:
        page_counts.update({_repeat_key(l) for l in lines if l})
    return {k for k, c in page_counts.items() if c >= 2 and c > n_pages / 2}


def _is_headline_like(line: str) -> bool:
    """Short, no final full stop, and mostly Capitalised / ALL CAPS words."""
    words = line.split()
    if not 2 <= len(words) <= 16 or line.rstrip().endswith((".", ",", ";")):
        return False
    if line.isupper():
        return True
    capitalised = sum(1 for w in words if w[0].isupper() or w[0].isdigit() or w[0] in "$₹")
    return capitalised / len(words) >= 0.6


def detect_split_mode(lines: list[str]) -> str:
    """
    Auto-detect how a document is laid out.

    In flowing prose that wraps across lines (typical PDF/newspaper text), many
    lines start mid-sentence with a lowercase word. In a list of headlines
    (one item per line) almost every line starts with a capital letter.
    """
    non_empty = [l for l in lines if l]
    if not non_empty:
        return SPLIT_LINES
    lowercase_ratio = sum(l[0].islower() for l in non_empty) / len(non_empty)
    return SPLIT_LINES if lowercase_ratio < 0.15 else SPLIT_BLOCKS


def _group_into_blocks(lines: list[str]) -> list[list[str]]:
    """
    Group lines into article blocks. A new article starts when:
      * a blank line (paragraph break or removed noise) is met, or
      * a headline-like line appears after body text has already been seen.
    Consecutive headline lines (a headline wrapped over 2-3 lines) stay together.
    """
    blocks: list[list[str]] = []
    current: list[str] = []
    has_body = False

    for line in lines:
        if not line:
            if current:
                blocks.append(current)
            current, has_body = [], False
            continue
        headline = _is_headline_like(line)
        if headline and (has_body or len(current) >= 3):
            blocks.append(current)
            current, has_body = [], False
        current.append(line)
        has_body = has_body or not headline

    if current:
        blocks.append(current)
    return blocks


def _shorten(text: str, limit: int = 140) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


def _block_to_article(block: list[str]) -> tuple[str, str]:
    """Return (title, full_text). Leading headline lines become the title."""
    head: list[str] = []
    i = 0
    while i < len(block) and i < 3 and _is_headline_like(block[i]):
        head.append(block[i])
        i += 1
    body = " ".join(block[i:])
    if head and body:
        title = " ".join(head)
        return title, f"{title.rstrip(':')}. {body}"
    if head:
        title = " ".join(head)
        return title, title
    return _shorten(body), body


def chunk_long_text(text: str, max_words: int = MAX_WORDS_PER_CHUNK) -> list[str]:
    """
    Split long articles at sentence boundaries into <= max_words chunks so that
    every part fits in FinBERT's 512-token window instead of being truncated.
    """
    if len(text.split()) <= max_words:
        return [text]
    chunks, current, count = [], [], 0
    for sentence in _SENTENCE_END_RE.split(text):
        n_words = len(sentence.split())
        if current and count + n_words > max_words:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(sentence)
        count += n_words
    if current:
        chunks.append(" ".join(current))
    return chunks


def _finalise(title: str, text: str, min_words: int, page) -> list[dict]:
    """Filter tiny fragments and chunk long articles into model-sized pieces."""
    if len(text.split()) < min_words:
        return []
    chunks = chunk_long_text(text)
    if len(chunks) == 1:
        return [{"page": page, "title": title, "text": text}]
    return [
        {"page": page, "title": f"{title} (part {k}/{len(chunks)})", "text": chunk}
        for k, chunk in enumerate(chunks, 1)
    ]


def _deduplicate(articles: list[dict]) -> list[dict]:
    """Remove repeated articles (same text ignoring case/punctuation), keep order."""
    seen, unique = set(), []
    for art in articles:
        key = re.sub(r"\W+", " ", art["text"].lower()).strip()
        if key not in seen:
            seen.add(key)
            unique.append(art)
    return unique


def articles_from_pages(pages: list[str], split_mode: str, min_words: int) -> tuple[list[dict], str]:
    """Full text pipeline for PDF pages (or a TXT file treated as one page)."""
    pages_lines = remove_running_headers([prepare_lines(p) for p in pages])

    if split_mode == SPLIT_AUTO:
        split_mode = detect_split_mode([l for lines in pages_lines for l in lines])

    articles: list[dict] = []
    for page_no, lines in enumerate(pages_lines, 1):
        if split_mode == SPLIT_LINES:
            blocks = [[l] for l in lines if l]
        else:
            blocks = _group_into_blocks(lines)
        for block in blocks:
            title, text = _block_to_article(block)
            articles.extend(_finalise(title, text, min_words, page_no))
    return _deduplicate(articles), split_mode


def articles_from_csv(df: pd.DataFrame, column, min_words: int) -> list[dict]:
    """One CSV row = one article/headline."""
    articles: list[dict] = []
    for raw in df[column].dropna().astype(str):
        text = clean_inline(raw)
        articles.extend(_finalise(_shorten(text), text, min_words, None))
    return _deduplicate(articles)


# =============================================================================
# 4b. LAYOUT-AWARE PDF READING (PyMuPDF text blocks, with OCR fallback)
# =============================================================================
#
# VIVA NOTE - why not just page.extract_text()?
# A newspaper page is a grid of columns. Plain text extraction reads each
# horizontal line straight across the page, so line 1 of column 1 is followed
# by line 1 of column 2, and every story gets interleaved with its neighbours.
#
# Path 1 - readable PDFs: PyMuPDF's page.get_text("blocks") already groups
# the text into paragraph blocks, each with its bounding box. We sort the
# blocks column by column, drop tiny ones (page numbers, captions, bylines)
# and send each larger block to FinBERT as a chunk.
#
# Path 2 - scrambled or scanned pages: some e-papers (e.g. Financial Express)
# deliberately scramble their text layer, so every extractor (PyMuPDF
# included) returns junk such as "\x01\x02\x03". Those pages are rendered to an
# image and read with OCR (RapidOCR); headline detection then splits the OCR
# lines into articles (see _segment_page).

OCR_DPI = 110             # 13.6 pt newspaper body text ≈ 21 px tall: comfortable for OCR
OCR_MIN_SCORE = 0.5       # drop low-confidence OCR lines
OCR_SECONDS_PER_PAGE = 30  # rough CPU estimate shown in the UI
COLUMN_TOLERANCE = 15     # blocks whose left edges are within 15 pt share a column


class Box(NamedTuple):
    """A line of OCR text and its position on the page, in PDF points."""
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    size: float         # text-box height, used as the font size
    bold: bool = False


@st.cache_resource
def _pdf_lock() -> threading.Lock:
    # PyMuPDF is not thread-safe, and every Streamlit session runs in its own thread.
    return threading.Lock()


def _require_pymupdf() -> None:
    if pymupdf is None:
        raise RuntimeError("PyMuPDF is not installed. Run: pip install pymupdf")


def _is_garbled(text: str) -> bool:
    """True if a page's text layer is missing or mostly unreadable symbols."""
    visible = [ch for ch in text if not ch.isspace()]
    if len(visible) < 50:
        return True  # scanned image or empty page
    bad = sum(1 for ch in visible if ch == "�" or "" <= ch <= "" or ord(ch) < 32)
    return bad / len(visible) > 0.2


@st.cache_data(show_spinner="Scanning PDF…", max_entries=8)
def scan_pdf(digest: str, _data: bytes) -> list[dict]:
    """Fast per-page check (no OCR): page size and whether the text layer is usable."""
    _require_pymupdf()
    with _pdf_lock(), pymupdf.open(stream=_data, filetype="pdf") as doc:
        return [
            {"width": page.rect.width, "height": page.rect.height, "garbled": _is_garbled(page.get_text())}
            for page in doc
        ]


@st.cache_data(show_spinner=False, max_entries=512)
def _native_page_blocks(digest: str, page_index: int, _data: bytes) -> list[tuple]:
    """PyMuPDF text blocks: (x0, y0, x1, y1, text), one per paragraph-like block."""
    with _pdf_lock(), pymupdf.open(stream=_data, filetype="pdf") as doc:
        blocks = doc[page_index].get_text("blocks")
    # Each block is (x0, y0, x1, y1, text, block_no, block_type); type 1 = image.
    return [(b[0], b[1], b[2], b[3], b[4]) for b in blocks if b[6] == 0 and b[4].strip()]


def _render_page(data: bytes, page_index: int, dpi: int):
    """Render one PDF page to a PIL image."""
    from PIL import Image

    _require_pymupdf()
    with _pdf_lock(), pymupdf.open(stream=data, filetype="pdf") as doc:
        pix = doc[page_index].get_pixmap(dpi=dpi)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def _sort_blocks_by_column(blocks: list[tuple]) -> list[tuple]:
    """
    Reading order for blocks: column by column, top to bottom within a column.
    Blocks whose left edges line up (within COLUMN_TOLERANCE) are one column.
    """
    column_of, column, previous = {}, -1, None
    for x0 in sorted({b[0] for b in blocks}):
        if previous is None or x0 - previous > COLUMN_TOLERANCE:
            column += 1
        column_of[x0], previous = column, x0
    return sorted(blocks, key=lambda b: (column_of[b[0]], b[1]))


def _block_chunks(blocks: list[tuple], page_no: int, min_words: int, repeated: set[str]) -> list[dict]:
    """Turn sorted text blocks into FinBERT chunks, dropping tiny and noisy blocks."""
    chunks = []
    for x0, y0, x1, y1, raw in _sort_blocks_by_column(blocks):
        lines = [clean_line(line) for line in raw.split("\n")]
        lines = [line for line in lines if line and not _is_noise_line(line)]
        text = _join_lines(lines)
        # Tiny blocks (page numbers, captions, bylines) and running headers are skipped.
        if len(text.split()) < min_words or _repeat_key(text) in repeated:
            continue
        title = lines[0] if len(lines) > 1 and len(lines[0].split()) <= 14 else _shorten(text)
        for chunk in _finalise(title, text, min_words, page_no):
            chunk["regions"] = [[x0, y0, x1, y1]]
            chunks.append(chunk)
    return chunks


@st.cache_resource(show_spinner="Loading OCR engine…")
def get_ocr_engine():
    """RapidOCR (ONNX, CPU) with the English PP-OCRv5 recogniser, which keeps word spacing."""
    from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR

    return RapidOCR(params={
        "Global.max_side_len": 6000,   # default 2000 px would shrink big newspaper pages too much
        "Global.log_level": "error",
        "Rec.lang_type": LangRec.EN,
        "Rec.ocr_version": OCRVersion.PPOCRV5,
        "Rec.model_type": ModelType.MOBILE,
        "Rec.rec_batch_num": 32,
    })


@st.cache_data(show_spinner=False, max_entries=512)
def _ocr_page_boxes(digest: str, page_index: int, dpi: int, _data: bytes) -> list[tuple]:
    """Render one page and OCR it. Returns line boxes converted back to PDF points."""
    result = get_ocr_engine()(_render_page(_data, page_index, dpi))
    if result.boxes is None:
        return []
    k = 72 / dpi
    boxes = []
    for poly, text, score in zip(result.boxes, result.txts, result.scores):
        text = text.strip()
        if score < OCR_MIN_SCORE or not text:
            continue
        xs, ys = [p[0] * k for p in poly], [p[1] * k for p in poly]
        boxes.append((text, min(xs), min(ys), max(xs), max(ys), max(ys) - min(ys), False))
    return boxes


def _body_size(boxes: list[Box]) -> float:
    """The page's body-text size: the most common size, weighted by characters."""
    counts: Counter = Counter()
    for b in boxes:
        counts[round(b.size * 2) / 2] += len(b.text)
    return counts.most_common(1)[0][0] if counts else 10.0


def _gaps(intervals: list[tuple[float, float]], min_gap: float) -> list[tuple[float, float]]:
    """Empty stretches (at least min_gap wide) between 1-D intervals, e.g. column gutters."""
    intervals = sorted(intervals)
    gaps, reach = [], intervals[0][1]
    for start, end in intervals[1:]:
        if start > reach and start - reach >= min_gap:
            gaps.append((reach, start))
        reach = max(reach, end)
    return gaps


def _split_at(boxes: list[Box], cuts: list[float], axis: str) -> list[list[Box]]:
    """Split boxes at the given x or y positions; groups come back in order."""
    groups: list[list[Box]] = [[] for _ in range(len(cuts) + 1)]
    for b in boxes:
        centre = (b.x0 + b.x1) / 2 if axis == "x" else (b.y0 + b.y1) / 2
        groups[bisect.bisect(cuts, centre)].append(b)
    return [g for g in groups if g]


def _xy_cut(boxes: list[Box], body: float, depth: int = 0) -> list[list[Box]]:
    """
    Recursive XY-cut: split the page into regions (leaves) in reading order.

    VIVA NOTE - the algorithm.
    Project all text boxes onto the y-axis and the x-axis. An empty stretch in
    a projection is whitespace that runs across the whole region:
      * a wide horizontal band (≥ 1.8 × body size) separates stacked stories
        → cut horizontally, read top → bottom;
      * a vertical gap that no box crosses is a column gutter
        → cut vertically, read left → right;
      * if the body text has gutters but a headline spanning several columns
        crosses them, the spanning headline blocks the vertical cut. So we cut
        horizontally just above and below the spanning headline first; the
        columns underneath can then be cut apart;
      * otherwise a smaller horizontal gap (≥ 1 × body size) is used.
    Recurse until nothing can be cut: each leaf is a single column of text.
    """
    if len(boxes) <= 1 or depth > 80:
        return [boxes]

    def recurse(groups: list[list[Box]]) -> list[list[Box]]:
        return [leaf for g in groups for leaf in _xy_cut(g, body, depth + 1)]

    y_gaps = _gaps([(b.y0, b.y1) for b in boxes], 0.3 * body)

    # 1. Tall empty band → separate stacked stories first.
    strong = [(a + b) / 2 for a, b in y_gaps if b - a >= 1.8 * body]
    if strong and len(groups := _split_at(boxes, strong, "y")) > 1:
        return recurse(groups)

    # 2. Column gutters that no box crosses → left-to-right columns.
    x_cuts = [(a + b) / 2 for a, b in _gaps([(b.x0, b.x1) for b in boxes], 0.5 * body)]
    if x_cuts and len(groups := _split_at(boxes, x_cuts, "x")) > 1:
        return recurse(groups)

    # 3. Gutters exist in the body text, but a spanning headline crosses them.
    body_boxes = [b for b in boxes if b.size < 1.15 * body]
    if len(body_boxes) >= 2:
        gutters = _gaps([(b.x0, b.x1) for b in body_boxes], 0.5 * body)
        spanning = [b for b in boxes if any(b.x0 < g0 and b.x1 > g1 for g0, g1 in gutters)]
        if gutters and spanning:
            fine_gaps = _gaps([(b.y0, b.y1) for b in boxes], 0.0)
            cuts = set()
            for s in spanning:
                above = [g for g in fine_gaps if g[1] <= s.y0 + 0.01]
                below = [g for g in fine_gaps if g[0] >= s.y1 - 0.01]
                if above:
                    cuts.add(sum(max(above)) / 2)
                if below:
                    cuts.add(sum(min(below)) / 2)
            if cuts and len(groups := _split_at(boxes, sorted(cuts), "y")) > 1:
                return recurse(groups)

    # 4. Ordinary paragraph-sized horizontal gaps.
    weak = [(a + b) / 2 for a, b in y_gaps if b - a >= 1.0 * body]
    if weak and len(groups := _split_at(boxes, weak, "y")) > 1:
        return recurse(groups)

    return [boxes]


def _attach_drop_caps(lines: list[Box], body: float) -> list[Box]:
    """Join a drop cap (one giant letter, e.g. "D" + "ELHI'S BARRICADES…") onto its first line."""
    is_cap = [len(b.text) == 1 and b.text.isalpha() and b.y1 - b.y0 >= 1.8 * body for b in lines]
    rest = [b for b, cap in zip(lines, is_cap) if not cap]
    for cap in (b for b, c in zip(lines, is_cap) if c):
        candidates = [
            i for i, ln in enumerate(rest)
            if cap.y0 - 0.5 * body <= (ln.y0 + ln.y1) / 2 <= cap.y1 and 0 <= ln.x0 - cap.x1 <= 2 * body
        ]
        if candidates:
            i = min(candidates, key=lambda j: rest[j].y0)
            rest[i] = rest[i]._replace(text=cap.text + rest[i].text, x0=cap.x0)
    return rest


def _is_noise_line(text: str) -> bool:
    """Stock tables, bylines and page furniture that should not reach FinBERT."""
    letters = sum(ch.isalpha() for ch in text)
    if letters < 0.4 * max(len(text.replace(" ", "")), 1):
        return True  # mostly numbers: stock tables, prices, scores
    if _CAPTION_RE.match(text):
        return True  # photo captions and credits
    return bool(re.match(r"^by\s+[A-Z]", text)) and len(text.split()) <= 6


_CAPTION_RE = re.compile(r"^(?:file\s+)?(?:photo|pic|picture|image|graphic|illustration)s?\s*[:|/-]", re.IGNORECASE)


def _prepare_page_lines(boxes: list[Box]) -> tuple[list[Box], float]:
    """Size filtering, drop caps and text cleaning for one page. Returns (lines, body size)."""
    if not boxes:
        return [], 10.0
    body = _body_size(boxes)
    # Tiny print (captions, credits) and giant display type (mastheads, section
    # titles such as a 300 pt "Opinion") are not article text.
    boxes = [b for b in boxes if 0.75 * body <= b.size <= 7 * body]
    lines = []
    for ln in _attach_drop_caps(boxes, body):
        text = clean_line(ln.text)
        if text and not _is_noise_line(text):
            lines.append(ln._replace(text=text))
    return lines, body


def _is_layout_headline(line: Box, body: float, ratio: float) -> bool:
    """Headline = clearly larger type, or a bold, slightly larger short line."""
    if len(line.text) < 4:
        return False
    if line.size >= ratio * body:
        return True
    return (line.bold and line.size >= 1.05 * body and len(line.text.split()) <= 16
            and not line.text.endswith("."))


def _x_overlap(a, b) -> float:
    """Horizontal overlap of two boxes (Box or [x0, y0, x1, y1])."""
    ax0, ax1 = (a.x0, a.x1) if isinstance(a, Box) else (a[0], a[2])
    bx0, bx1 = (b.x0, b.x1) if isinstance(b, Box) else (b[0], b[2])
    return min(ax1, bx1) - max(ax0, bx0)


def _headline_blocks(lines: list[Box], body: float, ratio: float) -> tuple[list[dict], list[Box]]:
    """
    Group headline lines into headline blocks (kicker + multi-line headline).

    Returns (blocks, remaining body lines). A block is a dict with its lines,
    bounding box [x0, y0, x1, y1] and the size of its main headline line.
    """
    heads = sorted((ln for ln in lines if _is_layout_headline(ln, body, ratio)), key=lambda b: b.y0)
    rest = [ln for ln in lines if not _is_layout_headline(ln, body, ratio)]

    def text_crosses(x: float, y: float) -> bool:
        """Does body text just below y run across x (i.e. x is inside a column, not a gutter)?"""
        return any(ln.x0 < x < ln.x1 and y <= ln.y0 <= y + 8 * body for ln in rest)

    blocks: list[dict] = []
    for h in heads:
        for blk in reversed(blocks):
            last = blk["lines"][-1]
            gap = h.y0 - last.y1
            next_line = -0.3 * h.size <= gap <= 0.8 * min(h.size, last.size) and \
                _x_overlap(h, last) >= 0.3 * min(h.x1 - h.x0, last.x1 - last.x0)
            # Same row, close together: one headline that OCR split into pieces
            # ("Modi@25" + "Steady agri-growth"), unless the gap between them
            # lines up with a column gutter below: then they are two headlines
            # side by side ("Rice exports…" | "NTPC seeking role…").
            left, right = (last, h) if last.x0 <= h.x0 else (h, last)
            same_row = abs((h.y0 + h.y1) - (last.y0 + last.y1)) / 2 <= 0.5 * min(h.size, last.size) and \
                -0.5 * h.size <= right.x0 - left.x1 <= 1.5 * max(h.size, last.size) and \
                text_crosses((left.x1 + right.x0) / 2, max(h.y1, last.y1))
            if next_line or same_row:
                blk["lines"].append(h)
                break
        else:
            blocks.append({"lines": [h]})

    for blk in blocks:
        # A short line straight below the headline, in similar-size type, is its
        # next line (OCR box heights vary with the letters in a line, so such a
        # line can fall just under the headline threshold).
        while True:
            last = blk["lines"][-1]
            nxt = [ln for ln in rest
                   if -0.3 * last.size <= ln.y0 - last.y1 <= 0.6 * last.size and ln.size >= 0.8 * last.size
                   and len(ln.text.split()) <= 12 and _x_overlap(ln, last) >= 0.5 * (ln.x1 - ln.x0)]
            if not nxt:
                break
            rest.remove(nxt[0])
            blk["lines"].append(nxt[0])

    # Blocks that don't start a story are treated as text:
    #  * short ALL-CAPS blocks: bylines ("SHYAMAL MAJUMDAR"), section kickers ("INDIA MARKET");
    #  * blocks where every line starts lowercase: large-type pull quotes
    #    ("sales have grown"). Headlines start with a capital, though OCR may
    #    split one into pieces such as "Registrations" + "of EVs jump".
    kept = []
    for blk in blocks:
        text = " ".join(b.text for b in blk["lines"])
        starts = [re.search(r"[A-Za-z]", b.text) for b in blk["lines"]]
        all_lower = all(m is None or m.group().islower() for m in starts)
        if (text.isupper() and len(text.split()) <= 3) or all_lower:
            rest.extend(blk["lines"])
        else:
            kept.append(blk)
    blocks = kept

    def bbox(blk: dict) -> list[float]:
        return [min(b.x0 for b in blk["lines"]), min(b.y0 for b in blk["lines"]),
                max(b.x1 for b in blk["lines"]), max(b.y1 for b in blk["lines"])]

    for blk in blocks:
        blk["bbox"], blk["size"] = bbox(blk), max(b.size for b in blk["lines"])

    # Decks and kickers: a smaller headline block just below or above a bigger
    # one, within its width, belongs to it (e.g. "As Jio prepares for its IPO…"
    # under "Jio changed telecom, Starlink faces the test").
    merged = True
    while merged:
        merged = False
        for big in sorted(blocks, key=lambda b: -b["size"]):
            for small in blocks:
                if small is big or small["size"] >= big["size"]:
                    continue
                sb, bb = small["bbox"], big["bbox"]
                gap = max(sb[1] - bb[3], bb[1] - sb[3])
                if gap <= 2.5 * body and _x_overlap(sb, bb) >= 0.9 * (sb[2] - sb[0]):
                    big["lines"] = sorted(big["lines"] + small["lines"], key=lambda b: (b.y0, b.x0))
                    big["bbox"] = bbox(big)
                    blocks.remove(small)
                    merged = True
                    break
            if merged:
                break
    return blocks, rest


def _owner(line: Box, blocks: list[dict]):
    """
    Index of the headline block that owns a body line, or None.

    VIVA NOTE - how newspapers are laid out: a headline sits on top of its own
    story, and the story's text runs in the columns directly beneath it. So a
    body line belongs to the *nearest headline above it that horizontally
    covers it*. A headline spanning three columns owns all three; a one-column
    headline owns only its column; a lower headline takes over from there.
    """
    best, best_bottom = None, float("-inf")
    width = line.x1 - line.x0
    for i, blk in enumerate(blocks):
        x0, _, x1, y1 = blk["bbox"]
        if y1 > line.y0 + 0.3 * line.size:
            continue  # the headline is not above this line
        overlap = _x_overlap(line, blk["bbox"])
        if overlap > 0 and (overlap >= 0.5 * width or overlap >= 0.8 * (x1 - x0)) and y1 > best_bottom:
            best, best_bottom = i, y1
    return best


def _join_lines(texts: list[str]) -> str:
    """Join text lines, re-joining words hyphenated across a line break."""
    return re.sub(r"\s+", " ", _HYPHEN_BREAK_RE.sub(r"\1\2", "\n".join(texts))).strip()


def _reading_order(lines: list[Box], body: float) -> tuple[list[Box], list[list[float]]]:
    """Order one article's lines column by column (XY-cut). Returns (lines, region boxes)."""
    if not lines:
        return [], []
    ordered, regions = [], []
    for leaf in _xy_cut(lines, body):
        leaf = sorted(leaf, key=lambda b: (round(b.y0, 1), b.x0))
        ordered.extend(leaf)
        regions.append([min(b.x0 for b in leaf), min(b.y0 for b in leaf),
                        max(b.x1 for b in leaf), max(b.y1 for b in leaf)])
    return ordered, regions


def _segment_page(lines: list[Box], body: float, page_no: int, min_words: int,
                  headline_ratio: float) -> tuple[list[dict], dict]:
    """Split one page's lines into articles: headline blocks + the text they own."""
    blocks, rest = _headline_blocks(lines, body, headline_ratio)
    owned: dict = {i: [] for i in range(len(blocks))}
    orphans = []
    for ln in rest:
        i = _owner(ln, blocks)
        (owned[i] if i is not None else orphans).append(ln)

    articles, n_regions = [], 0
    for i, blk in enumerate(blocks):
        head = _join_lines([b.text for b in blk["lines"]])
        body_lines, regions = _reading_order(owned[i], body)
        n_regions += len(regions)
        text = _join_lines([b.text for b in body_lines])
        full = f"{head.rstrip(':')}. {text}" if text else head
        for art in _finalise(head, full, min_words, page_no):
            art["regions"] = [blk["bbox"]] + regions
            art["_y"] = blk["bbox"][1]
            articles.append(art)

    # Text with no headline above it (continuations, briefs): one article per region.
    for leaf in _xy_cut(orphans, body) if orphans else []:
        leaf_lines, regions = _reading_order(leaf, body)
        n_regions += len(regions)
        text = _join_lines([b.text for b in leaf_lines])
        for art in _finalise(_shorten(text), text, min_words, page_no):
            art["regions"] = regions
            art["_y"] = regions[0][1]
            articles.append(art)

    articles.sort(key=lambda a: (a["_y"], a["regions"][0][0]))
    for art in articles:
        del art["_y"]
    return articles, {"headlines": len(blocks), "regions": n_regions}


def articles_from_pdf_layout(data: bytes, digest: str, page_indices: list[int], min_words: int,
                             force_ocr: bool, progress_cb=None) -> tuple[list[dict], dict]:
    """
    Full PDF pipeline for the selected pages.

    Readable pages: PyMuPDF text blocks, sorted by column → chunks.
    Scrambled/scanned pages: OCR lines → articles split at headlines.
    Running headers/footers (repeated in the page margins) are removed from both.
    """
    info = scan_pdf(digest, data)
    pages = []
    for done, idx in enumerate(page_indices):
        use_ocr = force_ocr or info[idx]["garbled"]
        if progress_cb is not None:
            progress_cb(done, len(page_indices), idx + 1, use_ocr)
        page = {"index": idx, "height": info[idx]["height"], "ocr": use_ocr}
        if use_ocr:
            page["lines"], page["body"] = _prepare_page_lines(
                [Box(*t) for t in _ocr_page_boxes(digest, idx, OCR_DPI, data)]
            )
        else:
            page["blocks"] = _native_page_blocks(digest, idx, data)
        pages.append(page)
    if progress_cb is not None:
        progress_cb(len(page_indices), len(page_indices), None, False)

    # Running headers/footers: text in the top/bottom 7% that repeats across pages.
    def margin_texts(p: dict) -> list[str]:
        h = p["height"]
        if p["ocr"]:
            return [ln.text for ln in p["lines"] if ln.y1 < 0.07 * h or ln.y0 > 0.93 * h]
        return [_join_lines([clean_line(t) for t in b[4].split("\n")])
                for b in p["blocks"] if b[3] < 0.07 * h or b[1] > 0.93 * h]

    repeated = _repeated_line_keys([margin_texts(p) for p in pages])

    articles: list[dict] = []
    stats = {"pages": len(pages), "ocr_pages": sum(p["ocr"] for p in pages), "regions": 0, "headlines": 0}
    for p in pages:
        if not p["ocr"]:
            chunks = _block_chunks(p["blocks"], p["index"] + 1, min_words, repeated)
            articles.extend(chunks)
            stats["regions"] += len(chunks)
            continue
        lines = [ln for ln in p["lines"] if _repeat_key(ln.text) not in repeated]
        # OCR box heights include padding and vary with the letters in a line.
        # Measured on a real e-paper: pull quotes ≤ 1.23 × body, the smallest
        # headlines ≥ 1.31 × body, so 1.28 is the headline cut-off.
        page_articles, page_stats = _segment_page(lines, p["body"], p["index"] + 1, min_words, 1.28)
        articles.extend(page_articles)
        stats["regions"] += page_stats["regions"]
        stats["headlines"] += page_stats["headlines"]
    return _deduplicate(articles), stats


@st.cache_data(show_spinner=False, max_entries=32)
def _page_png(digest: str, page_index: int, dpi: int, _data: bytes) -> bytes:
    buffer = io.BytesIO()
    _render_page(_data, page_index, dpi).save(buffer, format="PNG")
    return buffer.getvalue()


# Distinct outline colours for the debug overlay (neighbouring articles must differ).
_OVERLAY_COLORS = ["#4ade80", "#60a5fa", "#f472b6", "#facc15", "#a78bfa", "#fb923c", "#2dd4bf"]


def layout_overlay_image(data: bytes, digest: str, page_no: int, page_articles: list[dict], dpi: int = 60):
    """The page image with a numbered, coloured box around every detected article."""
    from PIL import Image, ImageDraw

    image = Image.open(io.BytesIO(_page_png(digest, page_no - 1, dpi, data))).convert("RGBA")
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    k = dpi / 72
    for n, art in enumerate(page_articles, 1):
        hex_color = _OVERLAY_COLORS[(n - 1) % len(_OVERLAY_COLORS)]
        rgb = tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
        for x0, y0, x1, y1 in art["regions"]:
            draw.rectangle([x0 * k, y0 * k, x1 * k, y1 * k], outline=rgb + (255,), fill=rgb + (40,), width=3)
        x0, y0 = min(r[0] for r in art["regions"]) * k, min(r[1] for r in art["regions"]) * k
        draw.rectangle([x0, y0, x0 + 26, y0 + 18], fill=rgb + (255,))
        draw.text((x0 + 5, y0 + 3), str(n), fill=(10, 10, 10, 255))
    return Image.alpha_composite(image, layer).convert("RGB")


# =============================================================================
# 5. TICKER EXTRACTION (regex pattern matching)
# =============================================================================

# $TSLA, $AAPL, $BRK.B  (the "cashtag" convention used on financial Twitter/X)
_CASHTAG_RE = re.compile(r"\$([A-Z]{1,6}(?:\.[A-Z])?)\b")
# NSE:RELIANCE, NASDAQ: AAPL, BSE:TCS
_EXCHANGE_RE = re.compile(r"\b(?:NSE|BSE|NYSE|NASDAQ|LSE|TSX)\s*:\s*([A-Z][A-Z0-9&\-]{0,14})\b")
# Any ALL-CAPS token (2-12 chars); only kept if it is in KNOWN_TICKERS.
_UPPER_TOKEN_RE = re.compile(r"\b[A-Z][A-Z&]{1,11}(?:\.[A-Z])?\b")
# Company names. Longest names first, so "Reliance Industries" wins over
# "Reliance" and "Bank Nifty" wins over "Nifty" at the same position.
_ALIAS_LOOKUP = {name.lower(): ticker for name, ticker in COMPANY_ALIASES.items()}
_ALIAS_RE = re.compile(
    r"(?<![\w&])(?:"
    + "|".join(re.escape(n) for n in sorted(COMPANY_ALIASES, key=len, reverse=True))
    + r")(?![\w&])",
    re.IGNORECASE,
)


@lru_cache(maxsize=32)
def _custom_ticker_regex(extras: frozenset):
    if not extras:
        return None
    alternatives = "|".join(re.escape(t) for t in sorted(extras, key=len, reverse=True))
    return re.compile(rf"(?<![\w&])(?:{alternatives})(?![\w&])", re.IGNORECASE)


def extract_tickers(text: str, extras: frozenset = frozenset()) -> list[str]:
    """
    Find every stock/index symbol mentioned in `text`, using four rules:
      1. Cashtags          $TSLA            → TSLA
      2. Exchange prefixes NSE:RELIANCE     → RELIANCE
      3. Whitelisted CAPS  "INFY rallies"   → INFY   (whitelist avoids CEO/GDP)
      4. Company names     "Infosys shares" → INFY
    plus any user-supplied extra tickers from the sidebar.
    """
    found: set[str] = set()
    found.update(_CASHTAG_RE.findall(text))
    found.update(_EXCHANGE_RE.findall(text))

    # Remove "NASDAQ: AAPL"-style spans first, so the exchange name itself
    # isn't also counted as the Nasdaq index.
    remaining = _EXCHANGE_RE.sub(" ", text)
    found.update(t for t in _UPPER_TOKEN_RE.findall(remaining) if t in KNOWN_TICKERS or t in extras)
    found.update(_ALIAS_LOOKUP[m.lower()] for m in _ALIAS_RE.findall(remaining))

    custom_re = _custom_ticker_regex(extras)
    if custom_re is not None:
        found.update(m.upper() for m in custom_re.findall(text))
    return sorted(found)


# =============================================================================
# 6. DASHBOARD COMPONENTS (bento cards + minimalist Plotly)
# =============================================================================

def show_chart(fig: go.Figure, key: str, alt: str) -> None:
    # theme=None keeps our minimalist layout instead of Streamlit's chart theme;
    # hiding the mode bar keeps the cards clean (zoom/pan still work).
    st.plotly_chart(fig, key=key, theme=None, alt=alt, config={"displayModeBar": False})


def render_kpis(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Bento KPI grid: a large market-mood hero card plus four smaller cards."""
    counts = df["sentiment"].value_counts().reindex(SENTIMENTS, fill_value=0)
    pos, neg, neu = (int(counts[s]) for s in SENTIMENTS)
    total = len(df)
    net_score = float(df["sentiment_score"].mean())

    # "Bullish %" = share of positive articles among the *opinionated* ones
    # (positive + negative). Neutral reporting ("RBI meets on Friday") says
    # nothing about direction, so it is excluded from the ratio.
    opinionated = pos + neg
    if opinionated == 0:
        mood = "Neutral"
    elif pos == neg:
        mood = "50/50 Balanced"
    elif pos > neg:
        mood = f"{pos / opinionated:.0%} Bullish"
    else:
        mood = f"{neg / opinionated:.0%} Bearish"

    if net_score > 0:
        net_sub, net_class = f"▲ {net_score:+.2f} net score", "up"
    elif net_score < 0:
        net_sub, net_class = f"▼ {net_score:+.2f} net score", "down"
    else:
        net_sub, net_class = "● +0.00 net score", ""

    # Proportional sentiment bar + legend at the bottom of the hero card.
    segments = "".join(
        f'<span class="mood-seg" style="flex:{n} 1 0;background:{SENTIMENT_COLORS[s]}"></span>'
        for s, n in zip(SENTIMENTS, (pos, neg, neu)) if n
    )
    legend = "".join(
        f'<span><i style="background:{SENTIMENT_COLORS[s]}"></i>{SENTIMENT_LABELS[s]}<b>{n}</b></span>'
        for s, n in zip(SENTIMENTS, (pos, neg, neu))
    )
    hero_extra = (
        f'<div><div class="mood-bar" role="img" aria-label="{pos} positive, {neg} negative, {neu} neutral">'
        f'{segments}</div><div class="mood-legend">{legend}</div></div>'
    )

    pages = int(df["page"].dropna().nunique())
    articles_sub = f"across {pages} page{'s' if pages != 1 else ''}" if pages else "headlines and articles"
    tickers_sub = " · ".join(summary.index[:3]) if not summary.empty else "none detected"

    cards = [
        kpi_card(
            "Market mood", mood, net_sub, area="hero", sub_class=net_class, extra=hero_extra,
            tooltip="Bullish % = Positive ÷ (Positive + Negative); neutral articles are excluded. "
                    "Net score = average of P(positive) − P(negative) over all articles (−1 … +1).",
        ),
        kpi_card("Articles analysed", f"{total:,}", articles_sub, area="a",
                 tooltip="Articles / chunks classified by FinBERT"),
        kpi_card("Model confidence", f"{df['confidence'].mean():.0%}", "mean softmax probability", area="b",
                 tooltip="Mean probability of the predicted class"),
        kpi_card("Bullish vs bearish", f"{pos} / {neg}", f"▲ positive / ▼ negative · {neu} neutral", area="c",
                 tooltip="Article counts per predicted class"),
        kpi_card("Tickers tracked", f"{len(summary)}", tickers_sub, area="d",
                 tooltip="Distinct tickers mentioned; the three most-mentioned are listed"),
    ]
    st.html(f'<div class="bento-grid">{"".join(cards)}</div>')


def render_donut(df: pd.DataFrame) -> None:
    counts = df["sentiment"].value_counts().reindex(SENTIMENTS, fill_value=0)
    total = len(df)
    fig = go.Figure(
        go.Pie(
            labels=[f"{SENTIMENT_LABELS[s]}  {counts[s] / total:.0%}" for s in SENTIMENTS],
            values=counts.values,
            hole=0.76,
            sort=False,
            direction="clockwise",
            # Card-coloured outlines open a thin gap between the slices.
            marker=dict(colors=[SENTIMENT_COLORS[s] for s in SENTIMENTS], line=dict(color=CARD_BG, width=3)),
            textinfo="none",
            hovertemplate="<b>%{label}</b><br>%{value} articles<extra></extra>",
        )
    )
    fig.update_layout(
        **CHART_LAYOUT,
        annotations=[dict(
            text=f"<b style='font-size:40px;color:{TEXT}'>{total}</b><br><span style='font-size:12px'>articles</span>",
            x=0.5, y=0.5, showarrow=False,
        )],
        legend=dict(orientation="h", yanchor="top", y=-0.04, xanchor="center", x=0.5, font=dict(color="#d4d4d8")),
        margin=dict(t=8, b=8, l=8, r=8),
        height=340,
    )
    show_chart(
        fig, key="donut",
        alt=f"Donut chart of sentiment: {counts['Positive']} positive, {counts['Negative']} negative, "
            f"{counts['Neutral']} neutral articles",
    )


def ticker_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per ticker: article counts per sentiment, mentions, net score."""
    exploded = df[["tickers", "sentiment", "sentiment_score"]].explode("tickers").dropna(subset=["tickers"])
    if exploded.empty:
        return pd.DataFrame()
    summary = (
        exploded.groupby(["tickers", "sentiment"]).size()
        .unstack(fill_value=0)
        .reindex(columns=SENTIMENTS, fill_value=0)
    )
    summary["Mentions"] = summary.sum(axis=1)
    summary["Net score"] = exploded.groupby("tickers")["sentiment_score"].mean()
    summary["Signal"] = summary["Net score"].map(
        lambda s: "▲ Bullish" if s > 0.15 else ("▼ Bearish" if s < -0.15 else "● Neutral")
    )
    summary.index.name = "Ticker"
    return summary.sort_values(["Mentions", "Net score"], ascending=[False, False])


def render_ticker_breakdown(summary: pd.DataFrame) -> None:
    if summary.empty:
        empty_state("No stock tickers were detected. Add the symbols you care about under Tickers in the sidebar.")
        return

    top_n = len(summary)
    if len(summary) > 5:
        top_n = st.slider("Tickers shown", 5, min(40, len(summary)), min(12, len(summary)))
    top = summary.head(top_n)

    long = (
        top[SENTIMENTS].reset_index()
        .melt(id_vars="Ticker", var_name="Sentiment", value_name="Articles")
    )
    long["Sentiment"] = long["Sentiment"].map(SENTIMENT_LABELS)
    fig = px.bar(
        long,
        x="Articles",
        y="Ticker",
        color="Sentiment",
        orientation="h",
        color_discrete_map={SENTIMENT_LABELS[s]: SENTIMENT_COLORS[s] for s in SENTIMENTS},
        category_orders={"Sentiment": [SENTIMENT_LABELS[s] for s in SENTIMENTS]},
    )
    # Most-mentioned ticker on top (Plotly draws the first category at the bottom).
    fig.update_yaxes(
        **CLEAN_AXIS, categoryorder="array", categoryarray=list(top.index[::-1]), title=None,
        tickfont=dict(color=TEXT, size=12),
    )
    fig.update_xaxes(
        **CLEAN_AXIS, title=None, dtick=1 if top["Mentions"].max() <= 10 else None,
        tickfont=dict(color=FAINT, size=11),
    )
    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>%{fullData.name}: %{x} article(s)<extra></extra>",
        marker_line=dict(color=CARD_BG, width=1.5),  # thin gap between stacked segments
    )
    fig.update_layout(
        **CHART_LAYOUT,
        barmode="stack",
        bargap=0.45,
        barcornerradius=6,
        height=max(300, 30 * len(top) + 100),
        legend=dict(title=None, orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color="#d4d4d8")),
        margin=dict(t=36, b=8, l=8, r=8),
    )
    show_chart(fig, key="ticker_bar", alt=f"Stacked bar chart of sentiment for the top {len(top)} tickers by mentions")


def render_ticker_table(summary: pd.DataFrame) -> None:
    if summary.empty:
        empty_state("The leaderboard fills in once tickers are detected in the articles.")
        return
    st.dataframe(
        summary.reset_index()[["Ticker", "Mentions", "Positive", "Negative", "Neutral", "Net score", "Signal"]],
        hide_index=True,
        key="ticker_table",
        alt="Per-ticker sentiment leaderboard",
        column_config={
            "Net score": st.column_config.NumberColumn(
                "Net score", format="%+.2f", help="Average of P(positive) − P(negative) across articles mentioning the ticker"
            ),
        },
    )


def _table_view(df: pd.DataFrame, include_text: bool = False) -> tuple[pd.DataFrame, dict]:
    """Prepare a display copy of the results plus Streamlit column configs."""
    view = df.copy()
    view["sentiment"] = view["sentiment"].map(SENTIMENT_LABELS)
    view["tickers"] = view["tickers"].map(", ".join)
    columns = ["title", "sentiment", "confidence", "sentiment_score", "tickers"]
    if view["page"].notna().any():
        columns.append("page")
    if include_text:
        columns.append("text")
    config = {
        "title": st.column_config.TextColumn("Headline / article", width="large"),
        "sentiment": st.column_config.TextColumn("Sentiment"),
        "confidence": st.column_config.ProgressColumn("Confidence", min_value=0.0, max_value=1.0, format="%.2f"),
        "sentiment_score": st.column_config.NumberColumn(
            "Impact score", format="%+.2f", help="P(positive) − P(negative): +1 very bullish, −1 very bearish"
        ),
        "tickers": st.column_config.TextColumn("Tickers"),
        "page": st.column_config.NumberColumn("Page", format="%d"),
        "text": st.column_config.TextColumn("Full text", width="large"),
    }
    return view[columns], config


def render_top_impact(df: pd.DataFrame) -> None:
    """Filterable tables of the day's most bullish / bearish articles."""
    all_tickers = sorted({t for ts in df["tickers"] for t in ts})

    # All filters in one row above the tables.
    f1, f2, f3, f4, f5 = st.columns([1.3, 1.1, 1.3, 1.3, 0.7])
    chosen_sentiments = f1.multiselect("Sentiment", SENTIMENTS, default=SENTIMENTS)
    min_conf = f2.slider("Min confidence", 0.0, 1.0, 0.5, 0.05)
    chosen_tickers = f3.multiselect("Tickers", all_tickers, placeholder="All tickers")
    keyword = f4.text_input("Keyword search", placeholder="e.g. inflation, merger")
    top_n = int(f5.number_input("Rows", min_value=5, max_value=200, value=10, step=5))

    mask = df["sentiment"].isin(chosen_sentiments) & (df["confidence"] >= min_conf)
    if chosen_tickers:
        wanted = set(chosen_tickers)
        mask &= df["tickers"].map(lambda ts: bool(wanted.intersection(ts)))
    if keyword.strip():
        mask &= df["text"].str.contains(keyword.strip(), case=False, regex=False)
    filtered = df[mask]
    st.caption(f"{len(filtered)} of {len(df)} articles match the filters.")

    tab_pos, tab_neg, tab_all = st.tabs(["▲ Most positive", "▼ Most negative", "All articles"])

    with tab_pos:
        top_pos = filtered[filtered["sentiment"] == "Positive"].nlargest(top_n, "sentiment_score")
        if top_pos.empty:
            empty_state("No positive articles match the current filters.")
        else:
            view, config = _table_view(top_pos)
            st.dataframe(view, hide_index=True, column_config=config, key="top_pos", alt="Most positive articles")

    with tab_neg:
        top_neg = filtered[filtered["sentiment"] == "Negative"].nsmallest(top_n, "sentiment_score")
        if top_neg.empty:
            empty_state("No negative articles match the current filters.")
        else:
            view, config = _table_view(top_neg)
            st.dataframe(view, hide_index=True, column_config=config, key="top_neg", alt="Most negative articles")

    with tab_all:
        if filtered.empty:
            empty_state("No articles match the current filters.")
        else:
            ordered = filtered.sort_values("sentiment_score", ascending=False)
            view, config = _table_view(ordered, include_text=True)
            st.dataframe(view, hide_index=True, column_config=config, key="all_articles", alt="All classified articles")

            export = ordered.drop(columns=["regions"], errors="ignore")
            export["tickers"] = export["tickers"].map(", ".join)
            st.download_button(
                "Download results as CSV",
                export.to_csv(index=False).encode("utf-8"),
                file_name="finbert_daily_sentiment.csv",
                mime="text/csv",
                icon=":material/download:",
            )


METHOD_HTML = (
    '<ul class="method-list">'
    "<li><b>Confidence</b>The softmax probability of the predicted class (0–1).</li>"
    "<li><b>Impact score</b>P(positive) − P(negative). It ranks articles from most bullish (+1) "
    "to most bearish (−1); neutral articles sit near 0.</li>"
    "<li><b>Bullish %</b>Positive ÷ (Positive + Negative). Purely factual, neutral reporting "
    "carries no direction, so it is excluded.</li>"
    "<li><b>Ticker signal</b>Average impact score of every article that mentions the ticker "
    "(above +0.15 bullish, below −0.15 bearish).</li>"
    "</ul>"
)


def render_dashboard(df: pd.DataFrame, run_info: dict) -> None:
    summary = ticker_summary(df)
    rate = len(df) / max(run_info["elapsed"], 1e-9)
    section_label(
        "Overview",
        f"{len(df)} articles · {run_info['elapsed']:.1f}s · {rate:.1f} articles/s · "
        f"batch {run_info['batch_size']} · {run_info['device']}",
    )
    render_kpis(df, summary)

    # Asymmetric bento rows: narrow + wide, then wide + narrow.
    section_label("Breakdown")
    left, right = st.columns([1, 1.6], gap="small")
    with left, bento("donut"):
        card_header("Distribution", "Sentiment mix")
        render_donut(df)
    with right, bento("tickers"):
        card_header("Tickers", "Sentiment by ticker")
        render_ticker_breakdown(summary)

    left, right = st.columns([1.6, 1], gap="small")
    with left, bento("leaderboard"):
        card_header("Leaderboard", "Net sentiment by ticker")
        render_ticker_table(summary)
    with right, bento("method"):
        card_header("Methodology", "How to read these numbers")
        st.html(METHOD_HTML)

    section_label("Headlines")
    with bento("headlines", stretch=False):
        card_header("Top impact", "The day's most bullish and bearish coverage")
        render_top_impact(df)


# =============================================================================
# 7. PAGE FLOWS
# =============================================================================

def render_sidebar() -> dict:
    with st.sidebar:
        st.header("Settings")
        mode = st.radio("Mode", [MODE_UPLOAD, MODE_QUICK])

        st.subheader(":material/memory: Batch inference")
        batch_size = st.select_slider(
            "Batch size",
            options=[4, 8, 16, 32, 64],
            value=16,
            help="How many articles FinBERT processes per forward pass. Larger = faster on GPU but uses more memory.",
        )

        st.subheader(":material/picture_as_pdf: PDF reading")
        pdf_mode = st.selectbox(
            "PDF reading mode",
            [PDF_MODE_LAYOUT, PDF_MODE_OCR, PDF_MODE_PLAIN],
            help="Layout-aware uses PyMuPDF text blocks, sorted column by column, as FinBERT chunks. "
                 "Pages with a scrambled or missing text layer are read with OCR automatically and split "
                 "into articles at their headlines. "
                 "Plain text flow is the old left-to-right reader, kept for comparison.",
        )

        st.subheader(":material/content_cut: Article splitting")
        split_mode = st.selectbox(
            "Split strategy (TXT and plain-text PDFs)",
            [SPLIT_AUTO, SPLIT_BLOCKS, SPLIT_LINES],
            help="Auto-detect treats headline lists as one item per line, and flowing newspaper text "
                 "as headline + paragraph blocks. Layout-aware PDFs split at detected headlines instead.",
        )
        min_words = st.slider("Minimum words per article", 3, 20, 5,
                              help="Shorter fragments (captions, bylines, leftovers) are discarded.")

        st.subheader(":material/sell: Tickers")
        custom = st.text_input("Extra tickers to track (comma-separated)", placeholder="e.g. IRFC, SUZLON, SOFI")
        extras = frozenset(t.strip().upper().lstrip("$") for t in custom.split(",") if t.strip())

        st.divider()
        device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        st.caption(f"Model: `{MODEL_NAME}`  \nDevice: **{device}**")

    return {
        "mode": mode,
        "batch_size": int(batch_size),
        "split_mode": split_mode,
        "pdf_mode": pdf_mode,
        "min_words": int(min_words),
        "extras": extras,
        "device": device,
    }


def render_welcome() -> None:
    steps = [
        ("01", "Extract", "PDF, TXT or CSV in. Newspaper pages are read column by column (with OCR "
                          "for scrambled e-papers) and split into articles at their headlines."),
        ("02", "Classify", "FinBERT scores every article in length-sorted mini-batches: positive, "
                           "negative or neutral, each with a confidence."),
        ("03", "Visualize", "Daily market mood, sentiment per ticker and the day's highest-impact "
                            "headlines, all filterable."),
    ]
    cards = "".join(
        f'<div class="bento-card"><div class="step-num">{n}</div>'
        f'<div><div class="step-title">{t}</div><div class="step-body">{b}</div></div></div>'
        for n, t, b in steps
    )
    st.html(f'<div class="steps-grid">{cards}</div>')
    with st.expander(":material/schema: How the pipeline works"):
        st.markdown(
            "1. **Extraction**: PyMuPDF's `page.get_text(\"blocks\")` returns paragraph blocks with their "
            "positions. Pages whose text layer is scrambled or missing are rendered and read with **RapidOCR**.\n"
            "2. **Reading order**: blocks are sorted column by column (blocks whose left edges line up form a "
            "column). OCR pages are split into articles at their headlines (large type).\n"
            "3. **Cleaning & chunking**: tiny blocks (page numbers, captions, bylines), running headers and "
            "stock tables are removed; each remaining block is a FinBERT chunk. "
            "Long blocks are split to fit BERT's 512-token limit.\n"
            "4. **Tokenization**: WordPiece sub-words with padding, truncation and an attention mask.\n"
            "5. **Batched inference**: FinBERT classifies articles in mini-batches under `torch.no_grad()`.\n"
            "6. **Softmax**: logits become probabilities for Positive / Negative / Neutral.\n"
            "7. **Ticker extraction**: regex for cashtags, exchange prefixes, a symbol whitelist and company names.\n"
            "8. **Aggregation**: daily mood, sentiment distribution, per-ticker breakdown and top-impact news."
        )


def load_uploaded_articles(uploaded, settings: dict) -> tuple[list[dict], str] | None:
    """Turn the uploaded file into a list of article dicts plus a status message."""
    data = uploaded.getvalue()
    ext = uploaded.name.rsplit(".", 1)[-1].lower()

    if ext == "pdf" and settings["pdf_mode"] != PDF_MODE_PLAIN:
        return load_pdf_with_layout(data, settings)

    if ext == "pdf":
        pages, engine = extract_pdf_pages(data)
        if sum(len(p.strip()) for p in pages) < 50:
            st.warning(
                "Almost no text could be extracted. This PDF is probably a scanned image, "
                "which needs OCR (not supported). Try a text-based e-paper PDF instead."
            )
            return None
        articles, used_mode = articles_from_pages(pages, settings["split_mode"], settings["min_words"])
        return articles, f"Extracted **{len(articles)}** articles from **{len(pages)}** pages ({engine}, {used_mode.lower()})."

    if ext == "txt":
        articles, used_mode = articles_from_pages([decode_text_file(data)], settings["split_mode"], settings["min_words"])
        for art in articles:
            art["page"] = None
        return articles, f"Extracted **{len(articles)}** articles from the text file ({used_mode.lower()})."

    if ext == "csv":
        raw = read_csv_bytes(data)
        if raw.empty:
            st.warning("The CSV file has no rows.")
            return None
        guess = detect_text_column(raw)
        columns = list(raw.columns)
        column = st.selectbox("Column containing the news text", columns,
                              index=columns.index(guess) if guess in columns else 0)
        articles = articles_from_csv(raw, column, settings["min_words"])
        return articles, f"Loaded **{len(articles)}** articles from column **{column}** ({len(raw)} rows)."

    st.error(f"Unsupported file type: .{ext}")
    return None


def load_pdf_with_layout(data: bytes, settings: dict) -> tuple[list[dict], str] | None:
    """Layout-aware PDF flow: scan, pick pages, read them (OCR when needed) once."""
    digest = hashlib.sha1(data).hexdigest()
    info = scan_pdf(digest, data)
    n_pages = len(info)
    force_ocr = settings["pdf_mode"] == PDF_MODE_OCR

    scrambled = sum(p["garbled"] for p in info)
    if scrambled and not force_ocr:
        st.markdown(
            f":orange[:material/document_scanner:] **{scrambled} of {n_pages}** pages have a scrambled or "
            "missing text layer, so they will be read with OCR."
        )

    first, last = 1, 1
    if n_pages > 1:
        first, last = st.slider("Pages to read", 1, n_pages, (1, min(8, n_pages)),
                                help="OCR takes roughly half a minute per dense page on a CPU; results are cached.")
    page_indices = list(range(first - 1, last))
    needs_ocr = [i for i in page_indices if force_ocr or info[i]["garbled"]]
    if needs_ocr and find_spec("rapidocr") is None:
        st.error("OCR is needed for these pages but RapidOCR is not installed. Run: pip install rapidocr onnxruntime")
        return None

    # Reading (especially OCR) is slow, so it only starts on request. After
    # that, reruns reuse the per-page cache and skip the button.
    read_key = (digest, first, last, force_ocr)
    first_read = st.session_state.get("pdf_read_key") != read_key
    read_slot = st.empty()  # cleared after reading, so only "Run FinBERT" remains
    if first_read:
        with read_slot.container():
            if needs_ocr:
                minutes = len(needs_ocr) * OCR_SECONDS_PER_PAGE / 60
                st.caption(f"{len(needs_ocr)} page(s) need OCR: roughly {minutes:.0f}–{minutes * 1.5:.0f} min on a CPU.")
            clicked = st.button(f"Read pages {first}–{last}", type="primary", icon=":material/document_scanner:")
        if not clicked:
            return None
        read_slot.empty()

    progress = st.progress(0.0, text="Reading pages…") if first_read else None

    def on_progress(done: int, total: int, page_no, via_ocr: bool) -> None:
        if progress is None:
            return
        label = f"{'OCR' if via_ocr else 'Reading'} page {page_no} ({done + 1}/{total})" if page_no else "Analysing layout…"
        progress.progress(done / total, text=label)

    articles, stats = articles_from_pdf_layout(
        data, digest, page_indices, settings["min_words"], force_ocr, on_progress
    )
    if progress is not None:
        progress.empty()
    st.session_state["pdf_read_key"] = read_key

    ocr_note = f", {stats['ocr_pages']} via OCR" if stats["ocr_pages"] else ""
    headline_note = f" and {stats['headlines']} headlines" if stats["headlines"] else ""
    return articles, (
        f"Read **{stats['pages']}** pages{ocr_note}: **{len(articles)}** chunks from "
        f"{stats['regions']} text blocks{headline_note}."
    )


def render_layout_overlay(data: bytes, df: pd.DataFrame) -> None:
    """Show a page with a numbered box around every detected article."""
    digest = hashlib.sha1(data).hexdigest()
    pages = sorted(int(p) for p in df["page"].dropna().unique())
    if not pages:
        return
    page_no = st.selectbox("Page", pages, format_func=lambda p: f"Page {p}", key="overlay_page")
    page_df = df[df["page"] == page_no]
    image = layout_overlay_image(data, digest, page_no, page_df.to_dict("records"))
    left, right = st.columns([1.4, 1], gap="small")
    left.image(image, width="stretch", alt=f"Page {page_no} with detected article regions outlined")
    right.dataframe(
        pd.DataFrame({"#": range(1, len(page_df) + 1), "Detected article": page_df["title"].tolist()}),
        hide_index=True, key="overlay_titles", alt="Articles detected on this page",
    )


def run_batch_inference(texts: list[str], settings: dict, signature: str) -> None:
    """Run FinBERT with a live progress bar and store results in session state."""
    load_finbert()  # make sure the (cached) model is ready before timing starts
    progress = st.progress(0.0, text="Starting batch inference…")

    def on_progress(batch_no: int, n_batches: int, done: int, total: int) -> None:
        progress.progress(done / total, text=f"Batch {batch_no}/{n_batches} · {done}/{total} articles classified")

    start = time.perf_counter()
    results = classify_texts(texts, settings["batch_size"], on_progress)
    elapsed = time.perf_counter() - start
    progress.progress(1.0, text=f"Done: {len(texts)} articles in {elapsed:.1f}s")

    # VIVA NOTE - why session_state?
    # Every widget click re-runs this script from top to bottom. Storing the
    # predictions in st.session_state means moving a filter slider only
    # re-draws the charts and does NOT re-run the expensive model.
    st.session_state["analysis"] = {
        "signature": signature,
        "results": results,
        "elapsed": elapsed,
        "batch_size": settings["batch_size"],
        "device": settings["device"],
    }


def prepare_and_classify(uploaded, settings: dict) -> tuple[pd.DataFrame, dict] | None:
    """Extract articles from the upload and run (or reuse) FinBERT predictions."""
    try:
        loaded = load_uploaded_articles(uploaded, settings)
    except Exception as exc:  # corrupt / encrypted / malformed files
        st.error(f"Could not read **{uploaded.name}**: {exc}")
        return None
    if loaded is None:
        return None
    articles, message = loaded
    if not articles:
        st.warning("No articles were found. Try lowering *Minimum words per article* or changing the split strategy.")
        return None

    st.markdown(f":green[:material/check_circle:] {message}")
    df = pd.DataFrame(articles)
    with st.expander(":material/visibility: Preview extracted articles"):
        st.dataframe(df[["title", "text"]].head(100), hide_index=True, key="preview", alt="Extracted articles preview")
    if "regions" in df.columns and st.toggle(
        "Show detected layout", help="Draws a numbered box around every article found on a page."
    ):
        render_layout_overlay(uploaded.getvalue(), df)

    # A fingerprint of the exact texts. If the file or splitting settings
    # change, the fingerprint changes and the old predictions are invalid.
    texts = df["text"].tolist()
    signature = hashlib.sha1("\x1e".join(texts).encode("utf-8")).hexdigest()
    analysis = st.session_state.get("analysis")
    has_results = analysis is not None and analysis["signature"] == signature

    if has_results:
        clicked = st.button("Re-run analysis", type="primary", icon=":material/refresh:")
    else:
        clicked = st.button(f"Run FinBERT on {len(texts)} articles", type="primary", icon=":material/play_arrow:")
    if clicked:
        run_batch_inference(texts, settings, signature)
        analysis, has_results = st.session_state["analysis"], True

    if not has_results:
        return None

    results = pd.DataFrame(analysis["results"])
    df = pd.concat([df.reset_index(drop=True), results], axis=1)
    # Tickers are cheap to compute, so they are recomputed on every rerun.
    # That way new "extra tickers" from the sidebar apply instantly.
    df["tickers"] = df["text"].map(lambda t: extract_tickers(t, settings["extras"]))
    return df, analysis


def render_newspaper_flow(settings: dict) -> None:
    section_label("Source")
    with bento("source", stretch=False):
        card_header(
            "Upload", "Today's newspaper",
            "A full e-paper PDF, a .txt of headlines or articles, or a .csv with one article per row.",
        )
        uploaded = st.file_uploader("Newspaper file", type=["pdf", "txt", "csv"], label_visibility="collapsed")
        prepared = prepare_and_classify(uploaded, settings) if uploaded is not None else None

    if uploaded is None:
        section_label("How it works")
        render_welcome()
    elif prepared is not None:
        render_dashboard(*prepared)


def render_quick_test(settings: dict) -> None:
    section_label("Quick test")
    with bento("quick", stretch=False):
        card_header("Single headline", "Score one headline or short article")
        text = st.text_area(
            "Headline or short article",
            value="Tesla shares plunge 12% after weaker-than-expected quarterly deliveries",
            height=110,
            label_visibility="collapsed",
        )
        clicked = st.button("Analyze sentiment", type="primary", icon=":material/bolt:")
    if not clicked:
        return
    text = clean_inline(text)
    if not text:
        st.warning("Please enter some text first.")
        return

    result = classify_texts([text], batch_size=1)[0]
    sentiment = result["sentiment"]
    tickers = extract_tickers(text, settings["extras"])

    section_label("Result")
    cards = [
        kpi_card("Prediction", SENTIMENT_LABELS[sentiment], f"{result['confidence']:.1%} confidence", area="hero",
                 sub_class={"Positive": "up", "Negative": "down"}.get(sentiment, "")),
        kpi_card("Confidence", f"{result['confidence']:.0%}", "softmax probability", area="a"),
        kpi_card("Impact score", f"{result['sentiment_score']:+.2f}", "P(positive) − P(negative)", area="b"),
        kpi_card("Tickers", ", ".join(tickers) if tickers else "—", f"{len(tickers)} detected", area="c"),
    ]
    st.html(f'<div class="bento-grid single-row">{"".join(cards)}</div>')

    with bento("quick-probs", stretch=False):
        card_header("Model output", "Class probabilities")
        probs = [result[f"p_{s.lower()}"] for s in SENTIMENTS]
        # Dual-tone: only the predicted class is coloured; the others recede.
        colors = [SENTIMENT_COLORS[s] if s == sentiment else "#27272a" for s in SENTIMENTS]
        fig = go.Figure(
            go.Bar(
                x=probs,
                y=[SENTIMENT_LABELS[s] for s in SENTIMENTS],
                orientation="h",
                marker=dict(color=colors),
                text=[f"{p:.1%}" for p in probs],
                textposition="outside",
                textfont=dict(color=TEXT, size=13),
                cliponaxis=False,
                hovertemplate="<b>%{y}</b>: %{x:.1%}<extra></extra>",
            )
        )
        fig.update_xaxes(**CLEAN_AXIS, range=[0, 1.15], showticklabels=False)
        fig.update_yaxes(
            **CLEAN_AXIS, categoryorder="array", categoryarray=[SENTIMENT_LABELS[s] for s in SENTIMENTS[::-1]],
            tickfont=dict(color=TEXT, size=13),
        )
        fig.update_layout(
            **CHART_LAYOUT, height=220, bargap=0.45, barcornerradius=8,
            margin=dict(t=8, b=8, l=8, r=8), showlegend=False,
        )
        show_chart(fig, key="quick_probs", alt=f"Bar chart of class probabilities; predicted {sentiment}")


# =============================================================================
# 8. ENTRY POINT
# =============================================================================

def main() -> None:
    st.set_page_config(page_title="FinBERT Market Sentiment", page_icon="📰", layout="wide")
    inject_css()

    # Load the (cached) model up front so the LIVE badge reflects reality.
    # The first visit waits for the download; every later rerun is instant.
    model_error = None
    try:
        _, _, torch_device = load_finbert()
        device_label = "GPU" if torch_device.type == "cuda" else "CPU"
    except Exception as exc:  # no internet on first run, corrupt cache, etc.
        model_error, device_label = exc, ""
    render_header(model_error is None, device_label)
    if model_error is not None:
        st.error(f"FinBERT could not be loaded: {model_error}")
        return

    settings = render_sidebar()
    if settings["mode"] == MODE_QUICK:
        render_quick_test(settings)
    else:
        render_newspaper_flow(settings)


if __name__ == "__main__":
    main()
