"""
CreatorLens AI 2.0
A Multilingual Social Media Intelligence and Audience Understanding System

Run:   streamlit run app.py
Needs: streamlit, pandas, numpy, scikit-learn, altair (altair ships with Streamlit)

Pipeline
  Module 1  Preprocessing and normalisation  (clean text, emoji features, language tagger)
  Module 2  Sentiment and feedback-intent classifier (heuristic stand-in for XLM-R / MuRIL)
  Module 3  Semantic grouping and FAQ extraction (char n-gram TF-IDF + cosine similarity)
  Module 4  Content opportunity ranker (prioritisation engine)
"""

import html
import re

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="CreatorLens AI 2.0",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# 1. CUSTOM CSS / PROFESSIONAL STYLING LAYER
# =============================================================================

C = dict(
    bg="#DCE3E8", card="#141416", border="#232326", coral="#FF3E30", coral2="#FF8A5C",
    olive="#9BAA3C", text="#F4F4F5", muted="#8A8A93", amber="#F5A524", green="#4ADE80",
    deep="#7F1D1D",
)


APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root{
  --bg:#0B0B0C; --card:#141416; --border:#232326; --coral:#FF3E30; --coral2:#FF8A5C;
  --olive:#9BAA3C; --text:#F4F4F5; --muted:#8A8A93; --amber:#F5A524; --green:#4ADE80;
}
html, body, .stApp, [class*="css"]{ font-family:'Inter','Segoe UI',sans-serif; }
.stApp{ background:var(--bg); color:var(--text); }
header[data-testid="stHeader"]{ background:transparent; }
#MainMenu, footer{ visibility:hidden; }
.block-container{ padding-top:1.3rem; padding-bottom:3rem; max-width:1500px; }
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li{ color:var(--text); }
h1,h2,h3,h4{ color:var(--text); letter-spacing:-0.01em; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"]{ background:#0F0F11; border-right:1px solid var(--border); }
[data-testid="stSidebar"] .block-container{ padding-top:1rem; }
.brand{ display:flex; align-items:center; gap:12px; padding:6px 6px 18px 6px; }
.brand-logo{ width:38px; height:38px; border-radius:11px; display:flex; align-items:center; justify-content:center;
  background:linear-gradient(135deg,#FF3E30,#B3150B); box-shadow:0 0 22px rgba(255,62,48,.45); font-size:18px; }
.brand-name{ font-weight:800; font-size:17px; line-height:1.1; color:#fff; }
.brand-sub{ font-size:11px; color:var(--muted); letter-spacing:.04em; }
.side-label{ font-size:11px; font-weight:700; letter-spacing:.12em; text-transform:uppercase; color:#5F5F69; margin:22px 0 8px 6px; }
[data-testid="stSidebar"] div[role="radiogroup"]{ gap:4px; }
[data-testid="stSidebar"] div[role="radiogroup"] > label{
  background:transparent; border:1px solid transparent; border-radius:10px; padding:10px 14px; width:100%;
  transition:all .15s ease; cursor:pointer; }
[data-testid="stSidebar"] div[role="radiogroup"] > label:hover{ background:#18181B; }
[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child{ display:none; }
[data-testid="stSidebar"] div[role="radiogroup"] > label p{ color:#B4B4BC; font-weight:500; font-size:14px; }
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked){
  background:linear-gradient(90deg, rgba(255,62,48,.24), rgba(255,62,48,.04));
  border:1px solid rgba(255,62,48,.5); box-shadow:0 0 20px rgba(255,62,48,.18); }
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p{ color:#fff; font-weight:700; }
.side-status{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:14px; margin-top:26px;
  font-size:12px; color:var(--muted); line-height:1.7; }
.side-status b{ color:var(--text); }

/* ---------- Widgets ---------- */
[data-baseweb="select"] > div, [data-baseweb="input"] > div{ background:#141416 !important; border-color:#232326 !important; }
[data-baseweb="tag"]{ background:rgba(255,62,48,.18) !important; color:#FFB4AD !important; border-radius:8px !important; }
input{ color:var(--text) !important; }
.stDownloadButton button, .stButton button{
  background:linear-gradient(135deg,#FF3E30,#D92A1C); color:#fff; border:none; border-radius:10px; font-weight:700;
  padding:.6rem 1.2rem; box-shadow:0 6px 22px rgba(255,62,48,.35); transition:transform .1s ease, box-shadow .15s ease; }
.stDownloadButton button:hover, .stButton button:hover{ transform:translateY(-1px); box-shadow:0 10px 28px rgba(255,62,48,.5); color:#fff; border:none; }
.stDownloadButton button p, .stButton button p{ color:#fff !important; }

/* ---------- Tabs ---------- */
[data-baseweb="tab-list"]{ gap:6px; border-bottom:1px solid var(--border); }
button[role="tab"]{ background:transparent; border-radius:10px 10px 0 0; padding:12px 18px; }
button[role="tab"] p{ color:var(--muted); font-weight:600; font-size:14px; }
button[role="tab"][aria-selected="true"] p{ color:var(--coral); }
[data-baseweb="tab-highlight"]{ background:var(--coral) !important; height:3px !important; box-shadow:0 0 12px rgba(255,62,48,.8); }
[data-baseweb="tab-border"]{ background:transparent !important; }

/* ---------- Cards ---------- */
.lens-card, [class*="st-key-card_"]{
  background-color:#141416; border:1px solid #232326; border-radius:12px; padding:20px;
  box-shadow:0px 4px 20px rgba(0,0,0,0.4); }
.lens-card{ margin-bottom:14px; }
.card-head{ display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:10px; }
.card-title{ font-size:15px; font-weight:700; color:var(--text); }
.card-sub{ font-size:12px; color:var(--muted); margin-top:2px; }
.legend{ display:flex; gap:14px; font-size:12px; color:var(--muted); }
.legend i{ display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:6px; }

.page-head{ display:flex; justify-content:space-between; align-items:center; margin-bottom:16px; flex-wrap:wrap; gap:10px; }
.page-title{ font-size:26px; font-weight:800; color:#fff; letter-spacing:-0.02em; }
.page-sub{ font-size:13px; color:var(--muted); margin-top:2px; }
.head-chips{ display:flex; gap:8px; flex-wrap:wrap; }

/* ---------- Banner ---------- */
.banner{ display:flex; justify-content:space-between; align-items:center; gap:18px; flex-wrap:wrap;
  background:linear-gradient(100deg,#FF3E30 0%,#E8321F 55%,#B3150B 100%); border-radius:14px; padding:16px 22px; margin-bottom:18px;
  box-shadow:0 10px 36px rgba(255,62,48,.30), inset 0 1px 0 rgba(255,255,255,.18); }
.banner-left{ display:flex; align-items:center; gap:14px; }
.banner-icon{ width:40px; height:40px; border-radius:50%; background:rgba(255,255,255,.18); display:flex; align-items:center; justify-content:center; font-size:19px; }
.banner-title{ font-weight:800; font-size:16px; color:#fff; }
.banner-text{ font-size:13px; color:rgba(255,255,255,.88); margin-top:2px; }
.banner-btn{ background:#fff; color:#C4210F !important; font-weight:800; font-size:13px; padding:10px 20px; border-radius:999px;
  text-decoration:none !important; box-shadow:0 6px 18px rgba(0,0,0,.25); white-space:nowrap; }
.banner-btn:hover{ background:#FFF1EF; }

/* ---------- Metric cards ---------- */
.metric{ min-height:196px; position:relative; overflow:hidden; }
.metric-top{ display:flex; justify-content:space-between; align-items:center; }
.metric-label{ font-size:12px; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:var(--muted); }
.metric-icon{ width:30px; height:30px; border-radius:9px; background:rgba(255,62,48,.12); border:1px solid rgba(255,62,48,.3);
  display:flex; align-items:center; justify-content:center; font-size:14px; }
.metric-value{ font-size:36px; font-weight:800; color:#fff; margin:10px 0 2px 0; letter-spacing:-0.02em; }
.metric-value small{ font-size:15px; font-weight:600; color:var(--muted); margin-left:6px; letter-spacing:0; }
.delta-up{ color:var(--green); font-weight:700; font-size:13px; }
.delta-coral{ color:var(--coral2); font-weight:700; font-size:13px; }
.metric-note{ font-size:12px; color:var(--muted); margin-left:6px; }
.spark{ width:100%; height:56px; margin-top:10px; display:block; }
.progress{ height:8px; border-radius:99px; background:#232326; margin-top:16px; overflow:hidden; }
.progress > span{ display:block; height:100%; border-radius:99px; background:linear-gradient(90deg,#FF3E30,#FF8A5C); box-shadow:0 0 12px rgba(255,62,48,.6); }
.mini-row{ display:flex; gap:8px; margin-top:12px; flex-wrap:wrap; }

/* ---------- Chips ---------- */
.chip{ display:inline-block; padding:3px 10px; border-radius:999px; font-size:11px; font-weight:700; letter-spacing:.02em; margin:1px 4px 1px 0; white-space:nowrap; }
.chip-red{ background:rgba(255,62,48,.15); color:#FF8A80; border:1px solid rgba(255,62,48,.35); }
.chip-coral{ background:rgba(255,138,92,.14); color:#FFA27A; border:1px solid rgba(255,138,92,.35); }
.chip-amber{ background:rgba(245,165,36,.14); color:#F7BE5C; border:1px solid rgba(245,165,36,.35); }
.chip-green{ background:rgba(74,222,128,.12); color:#6EE7A0; border:1px solid rgba(74,222,128,.30); }
.chip-blue{ background:rgba(96,165,250,.12); color:#8FBFFB; border:1px solid rgba(96,165,250,.30); }
.chip-gray{ background:#1C1C1F; color:#A1A1AA; border:1px solid #2A2A2E; }

/* ---------- Tables ---------- */
.table-wrap{ width:100%; overflow-x:auto; }
.lens-table{ width:100%; border-collapse:collapse; font-size:13px; color:var(--text); }
.lens-table th{ text-align:left; color:var(--muted); font-weight:700; text-transform:uppercase; font-size:11px; letter-spacing:.08em;
  padding:10px 12px; border-bottom:1px solid var(--border); white-space:nowrap; }
.lens-table td{ padding:13px 12px; border-bottom:1px solid #1C1C1F; vertical-align:top; }
.lens-table tr:hover td{ background:#18181B; }
.lens-table tr.row-high td{ background:rgba(255,62,48,.07); }
.lens-table tr.row-high td:first-child{ box-shadow:inset 3px 0 0 var(--coral); }
.lens-table tr.row-med td{ background:rgba(245,165,36,.06); }
.lens-table tr.row-med td:first-child{ box-shadow:inset 3px 0 0 var(--amber); }
.idea-title{ font-weight:700; color:#fff; font-size:14px; }
.idea-sub{ font-size:11px; color:var(--muted); margin-top:3px; }
.score-num{ font-weight:800; font-size:18px; color:#fff; }
.bar{ height:6px; background:#232326; border-radius:99px; overflow:hidden; margin-top:6px; min-width:110px; }
.bar > span{ display:block; height:100%; background:linear-gradient(90deg,#FF3E30,#FF8A5C); border-radius:99px; }
.evidence{ font-size:12px; color:#C9C9CF; line-height:1.5; margin-bottom:6px; max-width:420px; }
.ev-meta{ color:var(--muted); font-size:11px; }
.rank{ width:28px; height:28px; border-radius:8px; background:rgba(255,62,48,.14); border:1px solid rgba(255,62,48,.35);
  color:#FF8A80; font-weight:800; display:flex; align-items:center; justify-content:center; }

.faq-grid{ display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); gap:12px; margin-top:8px; }
.faq{ background:#101012; border:1px solid var(--border); border-radius:12px; padding:14px; }
.faq-q{ font-size:13px; color:#fff; font-weight:600; line-height:1.5; margin:8px 0; }
.faq-meta{ font-size:11px; color:var(--muted); }
.section-title{ font-size:13px; font-weight:700; color:var(--muted); letter-spacing:.1em; text-transform:uppercase; margin:20px 0 6px 0; }
.stat-grid{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:14px; }
.stat{ background:#101012; border:1px solid var(--border); border-radius:12px; padding:14px; }
.stat b{ display:block; font-size:24px; color:#fff; }
.stat span{ font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.08em; }
.mod-grid{ display:grid; grid-template-columns:repeat(2,1fr); gap:14px; }
.mod h4{ margin:0 0 6px 0; font-size:15px; color:#fff; }
.mod p{ font-size:13px; color:#A1A1AA; line-height:1.6; margin:0 0 10px 0; }
"""
st.markdown(f"<style>{APP_CSS}</style>", unsafe_allow_html=True)


# =============================================================================
# SMALL UI HELPERS
# =============================================================================
def flat(s: str) -> str:
    """Flatten multi-line HTML so Streamlit's markdown never sees indented/blank lines."""
    return "".join(line.strip() for line in s.splitlines())


def esc(s) -> str:
    return html.escape(str(s))


def trunc(s: str, n: int = 120) -> str:
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def chip(text: str, kind: str = "gray") -> str:
    return f'<span class="chip chip-{kind}">{esc(text)}</span>'


SENT_CHIP = {"Positive": "green", "Neutral": "gray", "Negative": "red"}
INTENT_CHIP = {"Question": "blue", "Feature Request": "amber", "Complaint": "red", "Casual Engagement": "gray"}
LANG_CHIP = {"English": "blue", "Tamil": "amber", "Code-Mixed": "coral"}


def card(key: str):
    """A bordered container styled as a floating card (falls back on older Streamlit)."""
    try:
        return st.container(key=f"card_{key}")
    except TypeError:
        return st.container(border=True)


def show_chart(chart):
    try:
        st.altair_chart(chart, width="stretch", theme=None)
    except TypeError:
        st.altair_chart(chart, use_container_width=True, theme=None)


def show_download(label, data, file_name, mime, key):
    try:
        st.download_button(label, data=data, file_name=file_name, mime=mime, key=key, width="stretch")
    except TypeError:
        st.download_button(label, data=data, file_name=file_name, mime=mime, key=key, use_container_width=True)


def dark_chart(chart, height=290):
    return (
        chart.properties(height=height, background="transparent")
        .configure_view(stroke=None)
        .configure_axis(
            gridColor=C["border"], domainColor=C["border"], tickColor=C["border"],
            labelColor="#A1A1AA", titleColor="#A1A1AA", labelFont="Inter", titleFont="Inter",
            labelFontSize=11, titleFontSize=11, titleFontWeight=600,
        )
        .configure_legend(labelColor="#D4D4D8", titleColor="#A1A1AA", labelFont="Inter")
    )


def sparkline_svg(values, color, w=240, ht=56):
    v = np.asarray(values, dtype=float)
    lo, hi = v.min(), v.max()
    span = (hi - lo) or 1.0
    xs = np.linspace(0, w, len(v))
    ys = ht - 5 - (v - lo) / span * (ht - 14)
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    area = f"0,{ht} {pts} {w},{ht}"
    gid = "spk" + color.replace("#", "")
    return (
        f'<svg viewBox="0 0 {w} {ht}" preserveAspectRatio="none" class="spark">'
        f'<defs><linearGradient id="{gid}" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" stop-color="{color}" stop-opacity="0.38"/>'
        f'<stop offset="100%" stop-color="{color}" stop-opacity="0"/></linearGradient></defs>'
        f'<polygon points="{area}" fill="url(#{gid})"/>'
        f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.4" '
        f'stroke-linejoin="round" stroke-linecap="round"/></svg>'
    )


def html_table(headers, rows, row_classes=None):
    head = "".join(f"<th>{h_}</th>" for h_ in headers)
    body = []
    for i, cells in enumerate(rows):
        cls = f' class="{row_classes[i]}"' if row_classes and row_classes[i] else ""
        body.append(f"<tr{cls}>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    return f'<div class="table-wrap"><table class="lens-table"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


# =============================================================================
# 2. DATA STRATEGY: SYNTHETIC DATA
# =============================================================================
def get_mock_data() -> pd.DataFrame:
    """Synthetic YouTube / Instagram comments: English, Tanglish, Tamil script, code-mixed and complaints."""
    rows = [
        # (platform, hours_ago, likes, text)
        ("YouTube", 3, 142, "Bro, next video-la cutting length calculation pathi sollunga, romba confuse aavuthu 😂"),
        ("YouTube", 5, 64, "Great video, but can you explain the setup cost?"),
        ("YouTube", 7, 120, "சூப்பர் கான்செப்ட் bro! 🌟 Make a detailed video on structural engineering tools next time."),
        ("Instagram", 9, 8, "Worst explanation ever. Fraud company, waste of time. Unsubscribing right now!!!"),
        ("YouTube", 11, 96, "Anna cutting length formula for stirrups and bends epdi calculate panrathu? Please make a video 🙏"),
        ("Instagram", 14, 210, "Semma video bro 🔥🔥 keep going"),
        ("YouTube", 18, 52, "Setup cost evlo aagum bro? Budget-la panna mudiyuma?"),
        ("YouTube", 21, 83, "Which software do you use for beam design? ETABS or STAAD? Please make a tutorial."),
        ("YouTube", 26, 58, "Your cutting length calculation is wrong sir, the bend deduction is incorrect. This can be unsafe on site!!"),
        ("Instagram", 30, 175, "Love your content! Been following since day one ❤️"),
        ("YouTube", 33, 77, "Staad pro tutorial venum anna, beginners ku puriyura maari 🔥"),
        ("YouTube", 37, 29, "Audio quality is very low, can't hear properly. Please fix the mic"),
        ("YouTube", 41, 88, "கட்டிங் லெந்த் கால்குலேஷன் பத்தி ஒரு வீடியோ போடுங்க bro 🙏"),
        ("Instagram", 46, 5, "Total scam channel. You people are cheating students with fake content 😡"),
        ("YouTube", 50, 45, "Can you explain the difference between M20 and M25 concrete?"),
        ("YouTube", 55, 64, "நன்றி அண்ணா, ரொம்ப பயனுள்ள வீடியோ 🙏"),
        ("YouTube", 61, 19, "Sound romba kammiya irukku bro, headphone-la kooda kekkala"),
        ("Instagram", 66, 33, "Concrete mix ratio evlo podanum? M25 ku sollunga anna"),
        ("YouTube", 72, 37, "What is the total budget for this setup? Is there any cheaper alternative?"),
        ("YouTube", 79, 22, "Nice video"),
        ("Instagram", 85, 71, "Site-la romba use aagum, cutting length calculation video podunga sir 🙌"),
        ("YouTube", 92, 12, "First comment 😎"),
        ("YouTube", 101, 49, "ETABS vs STAAD edhu better? Detailed comparison video venum bro"),
        ("Instagram", 110, 60, "Super explanation 👏 Thanks for sharing such useful content"),
    ]
    anchor = pd.Timestamp.now().floor("h")
    return pd.DataFrame(
        {
            "comment_id": [f"CMT-{1001 + i}" for i in range(len(rows))],
            "platform": [r[0] for r in rows],
            "text": [r[3] for r in rows],
            "timestamp": [anchor - pd.Timedelta(hours=r[1]) for r in rows],
            "likes": [r[2] for r in rows],
        }
    )


@st.cache_data(show_spinner=False)
def build_trend_data() -> pd.DataFrame:
    """Synthetic 30-day activity series that feed the headline charts and sparklines."""
    rng = np.random.default_rng(7)
    t = np.arange(30)
    days = pd.date_range(end=pd.Timestamp.now().normalize(), periods=30)
    weekly = np.sin(t / 7 * 2 * np.pi)
    return pd.DataFrame(
        {
            "date": days,
            "analyzed": np.round(180 + t * 4.5 + 28 * weekly + rng.normal(0, 12, 30)),
            "code_mixed_pct": np.round(36 + t * 0.5 + rng.normal(0, 2.2, 30), 1),
            "engagement": np.round(1400 + t * 38 + 260 * np.sin(t / 7 * 2 * np.pi + 1) + rng.normal(0, 90, 30)),
            "recent": t >= 23,
        }
    )


# =============================================================================
# MODULE 1: PREPROCESSING AND NORMALISATION
# =============================================================================
TAMIL_RE = re.compile(r"[\u0B80-\u0BFF]")
LATIN_RE = re.compile(r"[A-Za-z]")
TOKEN_RE = re.compile(r"[a-z0-9']+|[\u0B80-\u0BFF]+")  # \w would split Tamil vowel signs, so use explicit ranges
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF\u2B50\u2B06\u2764\uFE0F\u200D]+",
    flags=re.UNICODE,
)

# Common Tanglish spelling variants -> canonical form
TANGLISH_NORMALISE = {
    "patri": "pathi", "pattri": "pathi", "solunga": "sollunga", "sollungo": "sollunga",
    "eppadi": "epdi", "epadi": "epdi", "evalo": "evlo", "evvalavu": "evlo", "yevlo": "evlo",
    "venam": "venum", "vendum": "venum", "podungo": "podunga", "plz": "please", "pls": "please",
    "rmba": "romba", "rombha": "romba", "sema": "semma", "irukku": "irukku", "iruku": "irukku",
}

# Words that signal Romanised Tamil (Tanglish)
TANGLISH_MARKERS = {
    "pathi", "sollunga", "romba", "aavuthu", "aaguthu", "aagum", "evlo", "epdi", "enna", "irukku", "venum",
    "podunga", "panni", "panra", "panrathu", "panna", "mudiyuma", "semma", "nalla", "illa", "illai", "theriyala",
    "puriyala", "puriyura", "anna", "thambi", "kekkala", "kammiya", "maari", "ku", "seiyunga", "pannunga",
    "vidunga", "kudunga", "kooda", "thappu", "machan", "macha", "konjam", "namma", "unga", "edhu", "oru",
    "mosam", "nanri", "podanum", "podanuma", "enakku", "ennaku",
}


def extract_emojis(text: str) -> str:
    chars = [c for seq in EMOJI_RE.findall(text) for c in seq if c not in "\ufe0f\u200d"]
    return " ".join(chars)


def clean_text(text: str) -> str:
    t = URL_RE.sub(" ", text)
    t = MENTION_RE.sub(" ", t)
    t = EMOJI_RE.sub(" ", t)
    t = t.lower()
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)        # soooo -> soo
    t = re.sub(r"([!?.,])\1+", r"\1", t)        # !!! -> !
    t = re.sub(r"[a-z']+", lambda m: TANGLISH_NORMALISE.get(m.group(0), m.group(0)), t)
    return re.sub(r"\s+", " ", t).strip()


def identify_language(text: str) -> str:
    """Heuristic tagger: English | Tamil | Code-Mixed (Tamil script + English, or Romanised Tamil / Tanglish)."""
    tamil = len(TAMIL_RE.findall(text))
    latin = len(LATIN_RE.findall(text))
    if tamil:
        ratio = tamil / max(tamil + latin, 1)
        return "Tamil" if ratio >= 0.85 else "Code-Mixed"
    tokens = TOKEN_RE.findall(text.lower())
    tokens = [TANGLISH_NORMALISE.get(t, t) for t in tokens]
    if sum(1 for t in tokens if t in TANGLISH_MARKERS) >= 1:
        return "Code-Mixed"
    return "English"


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["emojis"] = out["text"].apply(extract_emojis)
    out["emoji_count"] = out["emojis"].apply(lambda s: len(s.split()) if s else 0)
    out["clean_text"] = out["text"].apply(clean_text)
    out["language"] = out["text"].apply(identify_language)
    out["word_count"] = out["clean_text"].apply(lambda s: len(TOKEN_RE.findall(s)))
    return out


# =============================================================================
# MODULE 2: SENTIMENT AND FEEDBACK-INTENT CLASSIFIER
# (heuristic engine that mimics the output contract of XLM-RoBERTa / MuRIL;
#  swap `classify_comment` for a real transformer call in production)
# =============================================================================
POS_LEX = {
    "love": 2, "amazing": 2, "awesome": 2, "excellent": 2, "superb": 2, "brilliant": 2, "fantastic": 2,
    "semma": 2, "mass": 2, "அருமை": 2, "செம": 2,
    "great": 1, "super": 1, "nice": 1, "good": 1, "helpful": 1, "useful": 1, "perfect": 1, "best": 1,
    "thanks": 1, "thank": 1, "nalla": 1, "nanri": 1, "சூப்பர்": 1, "நன்றி": 1, "பயனுள்ள": 1, "நல்ல": 1,
}
NEG_LEX = {
    "worst": 2, "fraud": 2, "scam": 2, "waste": 2, "fake": 2, "cheating": 2, "terrible": 2, "useless": 2,
    "pathetic": 2, "trash": 2, "hate": 2, "stupid": 2, "unsafe": 2, "mosam": 2, "மோசம்": 2,
    "confuse": 1, "confusing": 1, "confused": 1, "low": 1, "bad": 1, "poor": 1, "wrong": 1, "incorrect": 1,
    "mistake": 1, "error": 1, "boring": 1, "kammiya": 1, "kekkala": 1, "puriyala": 1, "theriyala": 1, "thappu": 1,
}
POS_EMOJI = set("❤🔥🌟🙏😍👏💯🙌👍😊")
NEG_EMOJI = set("😡🤬👎😤😠💔")
TOXIC_TERMS = {"fraud", "scam", "fake", "cheating", "waste", "worst", "idiot", "stupid", "trash", "pathetic", "hate",
               "useless", "unsubscribing", "mosam", "மோசம்"}

COMPLAINT_RE = re.compile(
    r"very low|can'?t hear|cannot hear|not clear|\bwrong\b|\bincorrect\b|\bunsafe\b|\bmistake\b|\berror\b|"
    r"kekkala|kammiya|thappu|mosam|மோசம்"
)
STRUCTURAL_RE = re.compile(r"\b(wrong|incorrect|unsafe|mistake|error|deduction|hazard|collapse|crack|dangerous)\b|thappu")
FEATURE_RE = re.compile(
    r"\b(make|create|post|upload|add|do)\b.*\b(video|tutorial|series|episode|course)\b|\bnext (video|time)\b|"
    r"\b(podunga|sollunga|venum|seiyunga|pannunga|kudunga|vidunga)\b|\bplease\b.*\b(make|cover|share)\b|"
    r"\b(tutorial|comparison)\b.*\b(venum|please)\b|போடுங்க|சொல்லுங்க|வேணும்|செய்யுங்க"
)
QUESTION_RE = re.compile(
    r"\?|\b(how|what|which|why|when|where|can you|could you|is there|do you)\b|"
    r"\b(epdi|enna|evlo|yaar|mudiyuma|irukka|aagumaa|edhu)\b|எப்படி|என்ன|எவ்வளவு"
)


def classify_comment(clean: str, emojis: str) -> dict:
    tokens = TOKEN_RE.findall(clean)
    pos = sum(POS_LEX.get(t, 0) for t in tokens)
    neg = sum(NEG_LEX.get(t, 0) for t in tokens)
    em = emojis.replace(" ", "")
    score = pos - neg + 0.5 * sum(c in POS_EMOJI for c in em) - 1.0 * sum(c in NEG_EMOJI for c in em)
    toxic_hits = sum(1 for t in tokens if t in TOXIC_TERMS)
    is_complaint = toxic_hits >= 1 or neg >= 2 or bool(COMPLAINT_RE.search(clean))

    if is_complaint:
        intent = "Complaint"
    elif FEATURE_RE.search(clean):
        intent = "Feature Request"
    elif QUESTION_RE.search(clean):
        intent = "Question"
    else:
        intent = "Casual Engagement"

    if intent == "Complaint" or score <= -1:
        sentiment = "Negative"
    elif score >= 1:
        sentiment = "Positive"
    else:
        sentiment = "Neutral"

    confidence = min(0.97, 0.62 + 0.07 * min(abs(score), 3) + (0.08 if intent != "Casual Engagement" else 0.0))
    return {
        "sentiment": sentiment,
        "intent": intent,
        "sentiment_score": round(float(score), 2),
        "confidence": round(confidence, 2),
        "toxic_hits": int(toxic_hits),
        "structural": bool(is_complaint and STRUCTURAL_RE.search(clean)),
    }


def classify(df: pd.DataFrame) -> pd.DataFrame:
    recs = [classify_comment(c, e) for c, e in zip(df["clean_text"], df["emojis"])]
    out = pd.concat([df.reset_index(drop=True), pd.DataFrame(recs)], axis=1)
    # Safety / review flags
    flagged, severity, reason, action = [], [], [], []
    for r in out.itertuples():
        reasons = []
        if r.structural:
            reasons.append("Structural accuracy / safety")
        if r.toxic_hits >= 1:
            reasons.append("Abusive / reputational language")
        if r.intent == "Complaint" and not reasons:
            reasons.append("Quality complaint")
        is_flag = bool(reasons)
        sev = "High" if (r.structural or r.toxic_hits >= 2) else ("Medium" if is_flag else "")
        act = ""
        if r.structural:
            act = "Verify calculation, pin a correction"
        elif r.toxic_hits >= 1:
            act = "Review for policy violation / hide"
        elif is_flag:
            act = "Reply and log for production fix"
        flagged.append(is_flag); severity.append(sev); reason.append(" · ".join(reasons)); action.append(act)
    out["flagged"], out["flag_severity"], out["flag_reason"], out["recommended_action"] = flagged, severity, reason, action
    return out


# =============================================================================
# MODULE 3: SEMANTIC GROUPING AND FAQ EXTRACTION
# =============================================================================
TOPICS = [
    dict(name="Cutting Length Calculation",
         idea="Video: Bar-bending cutting length calculation, stirrups and bends, step by step",
         seed="cutting length calculation bar bending stirrup formula bend deduction கட்டிங் லெந்த் கால்குலேஷன்"),
    dict(name="Project Setup Cost & Budget",
         idea="Video / post: Complete setup cost breakdown with budget-friendly alternatives",
         seed="setup cost budget price total cost cheaper alternative evlo aagum rate expense investment"),
    dict(name="Structural Software Tutorials",
         idea="Series: Structural engineering tools (ETABS vs STAAD) beam design tutorial for beginners",
         seed="structural engineering tools software staad etabs beam design tutorial beginners detailed video comparison"),
    dict(name="Concrete Grades & Mix Design",
         idea="Explainer: M20 vs M25 concrete grades and mix ratios",
         seed="concrete mix ratio m20 m25 grade cement sand aggregate difference"),
    dict(name="Audio & Production Quality",
         idea="Fix: Audio quality and mic setup for upcoming uploads",
         seed="audio quality sound mic noise volume voice headphone kammiya kekkala hear"),
]
TOPIC_BY_NAME = {t["name"]: t for t in TOPICS}
SIM_THRESHOLD = 0.22
ACTIONABLE = ("Question", "Feature Request", "Complaint")
SEVERITY_WEIGHT = 10.0


def group_topics(df: pd.DataFrame) -> pd.DataFrame:
    """Assign actionable comments to the nearest request profile using char n-gram TF-IDF cosine similarity.
    Character n-grams make the match robust to Tanglish spelling variants and mixed scripts."""
    out = df.copy()
    out["topic"] = "—"
    out["topic_sim"] = 0.0
    mask = out["intent"].isin(ACTIONABLE).to_numpy()
    n = int(mask.sum())
    if n == 0:
        return out
    seeds = [t["seed"].lower() for t in TOPICS]
    texts = out.loc[mask, "clean_text"].tolist()
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    X = vec.fit_transform(texts + seeds)
    sims = cosine_similarity(X[:n], X[n:])
    seed_tokens = [set(TOKEN_RE.findall(s)) for s in seeds]

    topics, best_sims = [], []
    for i, text in enumerate(texts):
        toks = set(TOKEN_RE.findall(text))
        kw = np.array([len(toks & st_) for st_ in seed_tokens])
        score = sims[i] + 0.06 * np.minimum(kw, 3)   # keyword evidence lifts the similarity a little
        j = int(score.argmax())
        if score[j] >= SIM_THRESHOLD:
            topics.append(TOPICS[j]["name"])
        else:
            topics.append("General Feedback")
        best_sims.append(round(float(score[j]), 3))
    out.loc[mask, "topic"] = topics
    out.loc[mask, "topic_sim"] = best_sims
    return out


def extract_faqs(df: pd.DataFrame) -> pd.DataFrame:
    q = df[df["intent"].isin(["Question", "Feature Request"]) & ~df["topic"].isin(["—", "General Feedback"])]
    rows = []
    for topic, g in q.groupby("topic"):
        best = g.sort_values("likes", ascending=False).iloc[0]
        rows.append(dict(
            topic=topic, canonical=best["text"], asked=len(g), total_likes=int(g["likes"].sum()),
            languages=", ".join(sorted(g["language"].unique())),
        ))
    cols = ["topic", "canonical", "asked", "total_likes", "languages"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_values(["asked", "total_likes"], ascending=False).reset_index(drop=True)


# =============================================================================
# MODULE 4: CONTENT OPPORTUNITY RANKER
# Score = (Count of Similar Requests * 1.5) + (Average Likes * 0.8) + (Severity weight if structural complaint)
# =============================================================================
def rank_opportunities(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["rank", "topic", "idea", "requests", "avg_likes", "severity_weight", "score", "languages", "evidence", "tier"]
    act = df[df["intent"].isin(ACTIONABLE) & ~df["topic"].isin(["—", "General Feedback"])]
    rows = []
    for topic, g in act.groupby("topic"):
        n = len(g)
        avg = float(g["likes"].mean())
        sev = SEVERITY_WEIGHT if bool(g["structural"].any()) else 0.0
        score = n * 1.5 + avg * 0.8 + sev
        top = g.sort_values("likes", ascending=False).head(2)
        rows.append(dict(
            topic=topic, idea=TOPIC_BY_NAME[topic]["idea"], requests=n, avg_likes=round(avg, 1),
            severity_weight=sev, score=round(score, 1), languages=", ".join(sorted(g["language"].unique())),
            evidence=[(r.text, int(r.likes), r.language) for r in top.itertuples()],
        ))
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    mx = out["score"].max()
    out["tier"] = out["score"].apply(lambda s: "P1 · Create now" if s >= 0.75 * mx else ("P2 · Plan next" if s >= 0.4 * mx else "P3 · Backlog"))
    return out


@st.cache_data(show_spinner=False)
def load_pipeline() -> pd.DataFrame:
    df = get_mock_data()
    df = preprocess(df)
    df = classify(df)
    df = group_topics(df)
    return df


# =============================================================================
# RENDERERS
# =============================================================================
def render_header(title: str, subtitle: str):
    st.markdown(flat(f'''
    <div class="page-head">
      <div><div class="page-title">{esc(title)}</div><div class="page-sub">{esc(subtitle)}</div></div>
      <div class="head-chips">{chip("● Live sync", "green")}{chip("Channel: Civil Engineering Tamil", "gray")}{chip("Last 30 days", "gray")}</div>
    </div>'''), unsafe_allow_html=True)


def render_banner(df: pd.DataFrame):
    flagged = int(df["flagged"].sum())
    st.markdown(flat(f'''
    <div class="banner">
      <div class="banner-left">
        <div class="banner-icon">⚡</div>
        <div>
          <div class="banner-title">Finish setting up your engine</div>
          <div class="banner-text">All 4 pipeline modules are running in heuristic mode · {flagged} comments await review ·
          connect XLM-RoBERTa / MuRIL to go production-grade.</div>
        </div>
      </div>
      <a class="banner-btn" href="#">Complete setup →</a>
    </div>'''), unsafe_allow_html=True)


def activity_chart(trend: pd.DataFrame):
    base = alt.Chart(trend).encode(x=alt.X("date:T", axis=alt.Axis(title=None, format="%b %d", tickCount=6, labelAngle=0)))
    area = base.mark_area(
        line={"color": C["coral"], "strokeWidth": 3}, interpolate="monotone",
        color=alt.Gradient(gradient="linear", x1=1, x2=1, y1=1, y2=0,
                           stops=[alt.GradientStop(color="rgba(255,62,48,0.38)", offset=0),
                                  alt.GradientStop(color="rgba(255,62,48,0.0)", offset=1)]),
    ).encode(
        y=alt.Y("analyzed:Q", scale=alt.Scale(zero=False),
                axis=alt.Axis(title="Comments analysed", titleColor=C["coral"])),
        tooltip=[alt.Tooltip("date:T", format="%d %b"), alt.Tooltip("analyzed:Q", title="Analysed"),
                 alt.Tooltip("code_mixed_pct:Q", title="Code-mixed %")],
    )
    line = base.mark_line(color=C["olive"], strokeWidth=3, interpolate="monotone").encode(
        y=alt.Y("code_mixed_pct:Q", scale=alt.Scale(zero=False),
                axis=alt.Axis(title="Code-mixed share (%)", titleColor=C["olive"], orient="right", grid=False)),
    )
    return dark_chart(alt.layer(area, line).resolve_scale(y="independent"))


def engagement_chart(trend: pd.DataFrame):
    bars = alt.Chart(trend).mark_bar(size=9, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X("date:T", axis=alt.Axis(title=None, format="%b %d", tickCount=6, labelAngle=0)),
        y=alt.Y("engagement:Q", axis=alt.Axis(title="Likes + replies")),
        color=alt.condition("datum.recent", alt.value(C["coral"]), alt.value(C["deep"])),
        tooltip=[alt.Tooltip("date:T", format="%d %b"), alt.Tooltip("engagement:Q", title="Engagement")],
    )
    return dark_chart(bars)


def render_dashboard(df, opps, faqs, trend):
    render_header("Dashboard", "Audience understanding across English, Tamil and Tanglish comments")
    render_banner(df)

    # ---- Row 1: charts ----
    c1, c2 = st.columns([1.15, 1])
    with c1:
        with card("activity"):
            st.markdown(flat(f'''
            <div class="card-head"><div><div class="card-title">AI Activity &amp; Language Trend Overview</div>
            <div class="card-sub">Comments processed vs. share of code-mixed language</div></div>
            <div class="legend"><span><i style="background:{C["coral"]}"></i>Analysed</span>
            <span><i style="background:{C["olive"]}"></i>Code-mixed %</span></div></div>'''), unsafe_allow_html=True)
            show_chart(activity_chart(trend))
    with c2:
        with card("engage"):
            st.markdown(flat(f'''
            <div class="card-head"><div><div class="card-title">Engagement Trend</div>
            <div class="card-sub">Daily likes + replies · last 7 days highlighted</div></div>
            <div class="legend"><span><i style="background:{C["coral"]}"></i>Recent</span>
            <span><i style="background:{C["deep"]}"></i>Earlier</span></div></div>'''), unsafe_allow_html=True)
            show_chart(engagement_chart(trend))

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # ---- Row 2: metric cards ----
    reach_series = np.linspace(78_200, 84_600, 18) + np.sin(np.arange(18) * 0.9) * 450
    analyzed, queued = len(df), 38
    pct = int(round(analyzed / (analyzed + queued) * 100))
    insights_now = len(opps) + len(faqs) + int(df["flagged"].sum())
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(flat(f'''
        <div class="lens-card metric"><div class="metric-top"><span class="metric-label">Total followers / reach</span><span class="metric-icon">👥</span></div>
        <div class="metric-value">84.6K</div>
        <div><span class="delta-up">▲ +3.2%</span><span class="metric-note">vs. last month</span></div>
        {sparkline_svg(reach_series, C["coral"])}</div>'''), unsafe_allow_html=True)
    with m2:
        st.markdown(flat(f'''
        <div class="lens-card metric"><div class="metric-top"><span class="metric-label">Processing queue</span><span class="metric-icon">⏳</span></div>
        <div class="metric-value">{analyzed}<small>analysed · {queued} scheduled</small></div>
        <div><span class="delta-coral">{pct}% of batch complete</span></div>
        <div class="progress"><span style="width:{pct}%"></span></div>
        <div class="mini-row">{chip(f"{int(df['flagged'].sum())} flagged", "red")}{chip(f"{int((df['language'] == 'Code-Mixed').sum())} code-mixed", "coral")}{chip(f"{int((df['language'] == 'Tamil').sum())} Tamil", "amber")}</div></div>'''), unsafe_allow_html=True)
    with m3:
        st.markdown(flat(f'''
        <div class="lens-card metric"><div class="metric-top"><span class="metric-label">Content insights generated</span><span class="metric-icon">💡</span></div>
        <div class="metric-value">{1240 + insights_now:,}<small>all time</small></div>
        <div><span class="delta-up">▲ +100 Last Month</span><span class="metric-note">target 1,500</span></div>
        <div class="progress"><span style="width:{min(100, int((1240 + insights_now) / 1500 * 100))}%"></span></div>
        <div class="mini-row">{chip(f"{len(opps)} opportunities", "blue")}{chip(f"{len(faqs)} FAQ clusters", "amber")}{chip(f"{insights_now} this batch", "gray")}</div></div>'''), unsafe_allow_html=True)

    # ---- Row 3: tabs ----
    tab1, tab2, tab3 = st.tabs(["Audience Insights & Content Opportunities", "Flagged Review Queue", "Export Report"])

    with tab1:
        if opps.empty:
            st.info("No actionable request clusters for the current filters.")
        else:
            rows = []
            for r in opps.to_dict("records"):
                pct_w = r["score"] / opps["score"].max() * 100
                ev = "".join(
                    f'<div class="evidence">“{esc(trunc(t, 118))}” <span class="ev-meta">♥ {l} · {esc(lang)}</span></div>'
                    for t, l, lang in r["evidence"]
                )
                tier_kind = "red" if r["tier"].startswith("P1") else ("amber" if r["tier"].startswith("P2") else "gray")
                sev = chip("Structural +10", "red") if r["severity_weight"] > 0 else chip("None", "gray")
                langs = "".join(chip(x.strip(), LANG_CHIP.get(x.strip(), "gray")) for x in r["languages"].split(","))
                rows.append([
                    f'<div class="rank">{r["rank"]}</div>',
                    f'<div class="idea-title">{esc(r["idea"])}</div><div class="idea-sub">Cluster: {esc(r["topic"])}</div><div style="margin-top:8px">{chip(r["tier"], tier_kind)}</div>',
                    f'<div class="score-num">{r["score"]}</div><div class="bar"><span style="width:{pct_w:.0f}%"></span></div>',
                    f'{r["requests"]}', f'{r["avg_likes"]}', sev, langs, ev,
                ])
            st.markdown(flat(f'''<div class="lens-card"><div class="card-head"><div><div class="card-title">Ranked content opportunities</div>
            <div class="card-sub">Score = (similar requests × 1.5) + (avg likes × 0.8) + structural-complaint severity weight</div></div></div>''')
                        + html_table(["#", "Content idea", "Opportunity score", "Requests", "Avg likes", "Severity", "Languages", "Comment evidence"], rows)
                        + "</div>", unsafe_allow_html=True)

        if not faqs.empty:
            cards = "".join(
                flat(f'''<div class="faq"><div>{chip(f["topic"], "coral")}</div>
                <div class="faq-q">“{esc(trunc(f["canonical"], 150))}”</div>
                <div class="faq-meta">Asked by <b style="color:#fff">{f["asked"]}</b> viewers · ♥ {f["total_likes"]} · {esc(f["languages"])}</div></div>''')
                for f in faqs.to_dict("records")
            )
            st.markdown(f'<div class="lens-card"><div class="card-title">Extracted FAQ clusters</div>'
                        f'<div class="card-sub">Most-liked phrasing used as the canonical question for each cluster</div>'
                        f'<div class="faq-grid">{cards}</div></div>', unsafe_allow_html=True)

    with tab2:
        fl = df[df["flagged"]].copy()
        fl["_o"] = fl["flag_severity"].map({"High": 0, "Medium": 1}).fillna(2)
        fl = fl.sort_values(["_o", "timestamp"], ascending=[True, False])
        if fl.empty:
            st.success("Nothing to review. No complaints or harmful language for the current filters.")
        else:
            rows, classes = [], []
            for r in fl.itertuples():
                sev_chip = chip(r.flag_severity, "red" if r.flag_severity == "High" else "amber")
                reasons = "".join(chip(x, "red" if "Structural" in x or "Abusive" in x else "amber") for x in r.flag_reason.split(" · "))
                rows.append([
                    sev_chip,
                    f'<div style="max-width:420px;line-height:1.5">{esc(r.text)}</div><div class="ev-meta" style="margin-top:4px">{esc(r.comment_id)} · {esc(r.platform)} · {r.timestamp.strftime("%d %b, %H:%M")}</div>',
                    chip(r.language, LANG_CHIP.get(r.language, "gray")), reasons, f'♥ {r.likes}',
                    f'<span style="color:#FFB4AD;font-weight:600">{esc(r.recommended_action)}</span>',
                ])
                classes.append("row-high" if r.flag_severity == "High" else "row-med")
            n_high = int((fl["flag_severity"] == "High").sum())
            st.markdown(flat(f'''<div class="lens-card"><div class="card-head"><div><div class="card-title">Flagged review queue</div>
            <div class="card-sub">{len(fl)} comments need attention · {n_high} high severity</div></div>
            <div class="legend"><span><i style="background:{C["coral"]}"></i>High</span><span><i style="background:{C["amber"]}"></i>Medium</span></div></div>''')
                        + html_table(["Severity", "Comment", "Language", "Trigger", "Likes", "Recommended action"], rows, classes)
                        + "</div>", unsafe_allow_html=True)

    with tab3:
        export_df = build_export(df, opps)
        st.markdown(flat(f'''<div class="lens-card"><div class="card-title">Export processed analysis</div>
        <div class="card-sub">UTF-8 with BOM so Tamil script opens correctly in Excel</div>
        <div class="stat-grid" style="margin-top:14px">
        <div class="stat"><b>{len(export_df)}</b><span>Comments</span></div>
        <div class="stat"><b>{int(export_df["flagged"].sum())}</b><span>Flagged</span></div>
        <div class="stat"><b>{len(opps)}</b><span>Opportunities</span></div>
        <div class="stat"><b>{len(export_df.columns)}</b><span>Columns</span></div></div></div>'''), unsafe_allow_html=True)
        b1, b2, _ = st.columns([1, 1, 1.4])
        with b1:
            show_download("⬇  Download full analysis (CSV)", export_df.to_csv(index=False).encode("utf-8-sig"),
                          "creatorlens_analysis.csv", "text/csv", "dl_full")
        with b2:
            opp_csv = opps.drop(columns=["evidence"]).to_csv(index=False).encode("utf-8-sig") if not opps.empty else b""
            show_download("⬇  Download opportunity ranking (CSV)", opp_csv, "creatorlens_opportunities.csv", "text/csv", "dl_opps")
        prev = export_df.head(8)
        rows = [[esc(r.comment_id), esc(trunc(r.text, 70)), esc(r.language), esc(r.sentiment), esc(r.intent), esc(r.topic), f"{r.opportunity_score}"] for r in prev.itertuples()]
        st.markdown('<div class="section-title">Preview</div>', unsafe_allow_html=True)
        st.markdown('<div class="lens-card">' + html_table(["ID", "Comment", "Language", "Sentiment", "Intent", "Topic", "Opp. score"], rows) + "</div>", unsafe_allow_html=True)


def build_export(df: pd.DataFrame, opps: pd.DataFrame) -> pd.DataFrame:
    score_map = dict(zip(opps["topic"], opps["score"])) if not opps.empty else {}
    out = df[[
        "comment_id", "platform", "timestamp", "text", "likes", "language", "emojis", "sentiment", "sentiment_score",
        "intent", "confidence", "topic", "flagged", "flag_severity", "flag_reason", "recommended_action",
    ]].copy()
    out["opportunity_score"] = out["topic"].map(score_map).fillna(0.0)
    return out


def render_explorer(df):
    render_header("Comment Explorer", "Every comment with its normalised features and model labels")
    q = st.text_input("Search comments", placeholder="Search text, topic or intent…", label_visibility="collapsed")
    view = df
    if q:
        ql = q.lower()
        view = df[df.apply(lambda r: ql in (r["text"] + " " + r["topic"] + " " + r["intent"]).lower(), axis=1)]
    rows = []
    for r in view.sort_values("timestamp", ascending=False).itertuples():
        rows.append([
            esc(r.comment_id), f'<div style="max-width:420px;line-height:1.5">{esc(r.text)}</div>',
            chip(r.language, LANG_CHIP.get(r.language, "gray")), chip(r.sentiment, SENT_CHIP[r.sentiment]),
            chip(r.intent, INTENT_CHIP[r.intent]), esc(r.topic), f"♥ {r.likes}", f"{r.confidence:.2f}",
        ])
    st.markdown(f'<div class="lens-card"><div class="card-title">{len(view)} comments</div>'
                + html_table(["ID", "Comment", "Language", "Sentiment", "Intent", "Topic", "Likes", "Conf."], rows) + "</div>",
                unsafe_allow_html=True)


def render_language(df):
    render_header("Language Intelligence", "How your audience mixes English, Tamil and Tanglish and how each group feels")
    c1, c2 = st.columns(2)
    lang_counts = df["language"].value_counts().rename_axis("language").reset_index(name="comments")
    with c1:
        with card("langdist"):
            st.markdown('<div class="card-title">Language distribution</div><div class="card-sub">Comments per detected language</div>', unsafe_allow_html=True)
            ch = alt.Chart(lang_counts).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, size=26).encode(
                y=alt.Y("language:N", sort="-x", axis=alt.Axis(title=None)),
                x=alt.X("comments:Q", axis=alt.Axis(title="Comments", tickMinStep=1)),
                color=alt.value(C["coral"]), tooltip=["language", "comments"])
            show_chart(dark_chart(ch, 240))
    with c2:
        with card("langsent"):
            st.markdown('<div class="card-title">Sentiment by language</div><div class="card-sub">Stacked comment counts</div>', unsafe_allow_html=True)
            ct = df.groupby(["language", "sentiment"]).size().reset_index(name="comments")
            ch = alt.Chart(ct).mark_bar(size=34).encode(
                x=alt.X("language:N", axis=alt.Axis(title=None, labelAngle=0)),
                y=alt.Y("comments:Q", axis=alt.Axis(title="Comments", tickMinStep=1)),
                color=alt.Color("sentiment:N", scale=alt.Scale(domain=["Positive", "Neutral", "Negative"],
                                                               range=[C["olive"], "#5B5B66", C["coral"]]), legend=alt.Legend(title=None, orient="bottom")),
                tooltip=["language", "sentiment", "comments"])
            show_chart(dark_chart(ch, 240))
    emo = pd.Series([e for s in df["emojis"] for e in s.split()]).value_counts().head(12)
    chips_html = "".join(f'<span class="chip chip-gray" style="font-size:15px;padding:6px 12px">{esc(e)} <b style="color:#fff">{n}</b></span>' for e, n in emo.items())
    st.markdown(f'<div class="lens-card"><div class="card-title">Emoji signal</div><div class="card-sub">Extracted as a separate feature before text normalisation</div><div style="margin-top:12px">{chips_html or "No emojis in current selection"}</div></div>', unsafe_allow_html=True)


def render_pipeline(df):
    render_header("Pipeline", "How a raw comment becomes a ranked content idea")
    mods = [
        ("Module 1 · Preprocessing & Normalisation", "Strips URLs and mentions, extracts emojis into their own feature, lowercases, collapses elongated words and maps Tanglish spelling variants. A script and vocabulary heuristic tags each comment as English, Tamil or Code-Mixed."),
        ("Module 2 · Sentiment & Intent Classifier", "Lexicon, emoji and pattern engine that emits the same contract as a multilingual transformer: sentiment, intent (Question, Feature Request, Complaint, Casual Engagement) and a confidence score. Swap classify_comment() for XLM-R / MuRIL inference."),
        ("Module 3 · Semantic Grouping & FAQ Extraction", "Character n-gram TF-IDF vectors compared by cosine similarity against request profiles, so Tanglish and Tamil variants land in the same cluster. The most-liked phrasing becomes each cluster's canonical FAQ."),
        ("Module 4 · Opportunity Ranker", "Score = (similar requests × 1.5) + (average likes × 0.8) + severity weight (+10 when a cluster holds a structural complaint). Tiers P1 to P3 are set relative to the top score."),
    ]
    cards = "".join(f'<div class="lens-card mod"><h4>{t}</h4><p>{d}</p>{chip("Active · heuristic mode", "green")}</div>' for t, d in mods)
    st.markdown(f'<div class="mod-grid">{cards}</div>', unsafe_allow_html=True)
    rows = []
    for r in df.head(6).itertuples():
        rows.append([
            f'<div style="max-width:300px">{esc(r.text)}</div>', f'<div style="max-width:260px;color:#A1A1AA">{esc(r.clean_text)}</div>',
            esc(r.emojis) or "—", chip(r.language, LANG_CHIP.get(r.language, "gray")),
            chip(r.sentiment, SENT_CHIP[r.sentiment]) + chip(r.intent, INTENT_CHIP[r.intent]), esc(r.topic),
        ])
    st.markdown('<div class="section-title">Live transformation samples</div>', unsafe_allow_html=True)
    st.markdown('<div class="lens-card">' + html_table(["Raw comment", "Normalised", "Emoji feature", "Language", "Labels", "Cluster"], rows) + "</div>", unsafe_allow_html=True)


# =============================================================================
# APP SHELL
# =============================================================================
def main():
    df_all = load_pipeline()
    trend = build_trend_data()
    icons = {"Dashboard": "▦", "Comment Explorer": "☰", "Language Intelligence": "◍", "Pipeline": "⚙"}

    with st.sidebar:
        st.markdown(flat('''<div class="brand"><div class="brand-logo">🔴</div><div><div class="brand-name">CreatorLens AI</div>
        <div class="brand-sub">VERSION 2.0 · MULTILINGUAL</div></div></div>'''), unsafe_allow_html=True)
        st.markdown('<div class="side-label">Menu</div>', unsafe_allow_html=True)
        page = st.radio("Navigation", list(icons), format_func=lambda x: f"{icons[x]}   {x}", label_visibility="collapsed")
        st.markdown('<div class="side-label">Filters</div>', unsafe_allow_html=True)
        langs = st.multiselect("Language", ["English", "Tamil", "Code-Mixed"], default=["English", "Tamil", "Code-Mixed"])
        sents = st.multiselect("Sentiment", ["Positive", "Neutral", "Negative"], default=["Positive", "Neutral", "Negative"])
        st.markdown(flat(f'''<div class="side-status"><b>Engine status</b><br>● Classifier: heuristic (XLM-R compatible)<br>
        ● Clustering: TF-IDF char n-grams<br>● Dataset: {len(df_all)} demo comments</div>'''), unsafe_allow_html=True)

    df = df_all[df_all["language"].isin(langs) & df_all["sentiment"].isin(sents)].copy()
    if df.empty:
        st.warning("No comments match the current filters. Widen the language or sentiment selection in the sidebar.")
        return

    opps = rank_opportunities(df)
    faqs = extract_faqs(df)

    if page == "Dashboard":
        render_dashboard(df, opps, faqs, trend)
    elif page == "Comment Explorer":
        render_explorer(df)
    elif page == "Language Intelligence":
        render_language(df)
    else:
        render_pipeline(df)


main()
