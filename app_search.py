#!/usr/bin/env python3
"""
Job Search — search offers, get KZIS occupations + ESCO skills
"""

import streamlit as st
import sqlite3
import numpy as np
import json
import faiss
import pickle
import os
import warnings
import io
import collections
import pandas as pd
from dotenv import load_dotenv
import voyageai
import plotly.graph_objects as pgo
from statsmodels.tsa.arima.model import ARIMA

load_dotenv()

st.set_page_config(
    page_title="Skill Demand and Supply",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
.main .block-container { padding-top: 2.5rem; padding-bottom: 3rem; max-width: 860px; }
#MainMenu, footer, header { visibility: hidden; }
.stApp { background: #f8f9fa; }

/* Header */
.app-header { text-align: center; margin-bottom: 2rem; }
.app-title { font-size: 2.2rem; font-weight: 300; color: #1a1a2e; margin: 0; letter-spacing: -0.03em; }
.app-sub { font-size: 1rem; color: #888; margin-top: 0.4rem; }

/* Section labels */
.sec-label { font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: #888; margin-bottom: 0.5rem; margin-top: 0.4rem; }

/* Result card */
.rcard { background: #fff; border: 1px solid #e4e4ea; border-radius: 12px; padding: 1.3rem 1.5rem; margin-bottom: 0.7rem; }
.rcard-title { font-size: 1.15rem; font-weight: 500; color: #1a1a2e; margin-bottom: 0.25rem; line-height: 1.4; }
.rcard-meta { font-size: 0.85rem; color: #888; margin-bottom: 0; }

/* KZIS */
.kzis-sec { margin-top: 0.9rem; padding-top: 0.9rem; border-top: 1px solid #f0f0f5; }
.kzis-lbl { font-size: 0.68rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: #999; margin-bottom: 0.4rem; }
.kr { display: flex; align-items: center; justify-content: space-between; padding: 0.45rem 0.7rem; margin: 0.25rem 0; background: #f7f8fa; border-radius: 7px; border: 1px solid #eeeff3; }
.kr-n { font-size: 0.9rem; color: #222; flex: 1; line-height: 1.35; }
.kr-s { font-size: 0.78rem; font-weight: 600; color: #2d6a2d; background: #e4f2e4; padding: 0.15rem 0.5rem; border-radius: 4px; margin-left: 0.8rem; white-space: nowrap; }

/* Similar */
.sr { display: flex; align-items: center; justify-content: space-between; padding: 0.45rem 0.7rem; margin: 0.25rem 0; background: #f7f8fa; border-radius: 7px; border: 1px solid #eeeff3; }
.sr-n { font-size: 0.9rem; color: #222; flex: 1; line-height: 1.35; }
.sr-s { font-size: 0.78rem; font-weight: 600; color: #3a4a7a; background: #e4eaf5; padding: 0.15rem 0.5rem; border-radius: 4px; margin-left: 0.8rem; white-space: nowrap; }

/* Skills */
.skill-row { display: flex; align-items: center; justify-content: space-between; padding: 0.45rem 0.7rem; margin: 0.25rem 0; background: #f7f8fa; border-radius: 7px; border: 1px solid #eeeff3; }
.skill-name { font-size: 0.9rem; color: #222; flex: 1; line-height: 1.35; }
.skill-score { font-size: 0.78rem; font-weight: 600; color: #7a5a1a; background: #f5edd5; padding: 0.15rem 0.5rem; border-radius: 4px; margin-left: 0.8rem; white-space: nowrap; }
.skill-type-tag { font-size: 0.68rem; color: #666; background: #eee; padding: 0.1rem 0.4rem; border-radius: 3px; margin-left: 0.5rem; }

/* Item row */
.item-row { padding: 0.55rem 0; border-bottom: 1px solid #f0f0f5; font-size: 0.9rem; color: #222; line-height: 1.5; }
.item-row:last-child { border-bottom: none; }
.item-type { font-size: 0.65rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; padding: 0.1rem 0.35rem; border-radius: 3px; margin-right: 0.4rem; display: inline-block; vertical-align: middle; }
.item-type-req { color: #3a4a7a; background: #e4eaf5; }
.item-type-resp { color: #7a5a1a; background: #f5edd5; }
.item-skill { font-size: 0.82rem; color: #666; margin-top: 0.15rem; }
.item-skill-name { color: #444; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    flex-wrap: wrap !important;
    overflow: visible !important;
    mask-image: none !important;
    -webkit-mask-image: none !important;
    background-image: none !important;
}
.stTabs [data-baseweb="tab-list"] > div {
    display: none !important;
}
.stTabs [data-baseweb="tab-list"]::before,
.stTabs [data-baseweb="tab-list"]::after {
    display: none !important;
    background: transparent !important;
    box-shadow: none !important;
}
.stTabs [data-testid="stHorizontalBlock"] {
    overflow: visible !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    box-shadow: none !important;
}
.stTabs [data-baseweb="tab"] {
    padding: 0.6rem 1.1rem;
    font-size: 0.92rem;
    color: #333 !important;
    border-bottom: 3px solid transparent !important;
}
.stTabs [data-baseweb="tab"] p { color: #333 !important; }
.stTabs [aria-selected="true"] {
    color: #1a1a2e !important;
    font-weight: 500;
    border-bottom: 3px solid #e55b52 !important;
}
.stTabs [aria-selected="true"] p { color: #1a1a2e !important; font-weight: 500; }

/* Radio */
div[role="radiogroup"] label { color: #333 !important; }
div[role="radiogroup"] label p { color: #333 !important; }

/* Buttons — primary */
.stButton > button[kind="primary"],
[data-testid="baseButton-primary"],
.stButton button[data-testid="baseButton-primary"] {
    background: #1a1a2e !important; color: white !important;
    border: none !important; border-radius: 8px !important;
    padding: 0.55rem 1.8rem !important; font-size: 0.95rem !important; font-weight: 400 !important;
}
.stButton > button[kind="primary"]:hover,
[data-testid="baseButton-primary"]:hover {
    background: #2a2a4e !important;
}

/* Buttons — secondary (including download buttons) */
.stButton > button[kind="secondary"],
[data-testid="baseButton-secondary"],
.stButton button[data-testid="baseButton-secondary"],
.stDownloadButton > button,
[data-testid="stDownloadButton"] button,
[data-testid="baseButton-secondary"] {
    background: #eaf3ff !important; color: #31557a !important;
    border: 1px solid #c7d9ef !important; border-radius: 999px !important;
    padding: 0.5rem 1.05rem !important; font-size: 0.88rem !important; font-weight: 600 !important;
    box-shadow: 0 1px 2px rgba(49, 85, 122, 0.08) !important;
}
.stButton > button[kind="secondary"]:hover,
[data-testid="baseButton-secondary"]:hover,
.stDownloadButton > button:hover,
[data-testid="stDownloadButton"] button:hover {
    background: #dcecff !important; color: #23486d !important;
    border-color: #aecaec !important;
}

/* Selectbox - force white bg and dark text */
.stSelectbox div[data-baseweb="select"] { 
    background-color: #fff !important; 
    background: #fff !important;
}
.stSelectbox div[data-baseweb="select"] * { color: #222 !important; }
.stSelectbox div[data-baseweb="input"] {
    background-color: #fff !important;
    background: #fff !important;
}
.stSelectbox input { 
    color: #222 !important; 
    -webkit-text-fill-color: #222 !important;
    background-color: #fff !important;
    background: #fff !important;
}
.stSelectbox [role="combobox"] { 
    background-color: #fff !important; 
    color: #222 !important;
}
.stSelectbox [role="listbox"] { 
    background-color: #fff !important; 
}
.stSelectbox [role="option"] { 
    color: #222 !important; 
    background-color: #fff !important;
}
.stSelectbox [role="option"]:hover { 
    background-color: #f0f0f0 !important; 
}
.stSelectbox div[data-testid="stMarkdownContainer"] { 
    color: #222 !important; 
}
/* Override any dark theme */
[data-baseweb="select"] > div { 
    background-color: #fff !important; 
    color: #222 !important; 
}

/* Text input */
.stTextInput input { 
    color: #222 !important; 
    -webkit-text-fill-color: #222 !important;
    background-color: #fff !important;
    background: #fff !important;
}
.stTextInput input::placeholder { 
    color: #999 !important; 
    opacity: 1 !important;
    -webkit-text-fill-color: #999 !important;
}
.stTextInput div[data-baseweb="base-input"] {
    background-color: #fff !important;
}

.info-box { background: #f0f2f8; border-radius: 8px; padding: 0.8rem 1rem; font-size: 0.85rem; color: #555; text-align: center; margin: 1rem 0; }
.divider { border: none; border-top: 1px solid #eeeff3; margin: 0.3rem 0; }

/* Chart iframe spacing */
iframe { margin-bottom: 0 !important; }

/* Remove whitespace around Datawrapper iframes in tab1 and tab2 */
div[data-testid="stIFrame"] {
    margin: 0 !important;
    padding: 0 !important;
    line-height: 0 !important;
}
div[data-testid="stIFrame"] iframe {
    display: block !important;
    margin: 0 !important;
    padding: 0 !important;
}
div[data-testid="stElementContainer"]:has(div[data-testid="stIFrame"]) {
    margin-bottom: 0 !important;
    padding-bottom: 0 !important;
}
div[data-testid="stElementContainer"]:has(div[data-testid="stIFrame"]) + div[data-testid="stElementContainer"] {
    margin-top: 0 !important;
}

/* Dialog */
div[data-testid="stDialog"] div[role="dialog"] {
    background: #fbfcfe !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 18px !important;
    box-shadow: 0 20px 50px rgba(71, 85, 105, 0.18) !important;
}
div[data-testid="stDialog"] div[role="dialog"] * {
    color: #334155 !important;
}
div[data-testid="stDialog"] button[aria-label="Close"] {
    color: #64748b !important;
}

/* NACE */
.nace-lead {
    background: linear-gradient(135deg, #f8fbff 0%, #f1f6fd 100%);
    border: 1px solid #d9e6f4;
    border-radius: 14px 14px 0 0;
    border-bottom: none;
    padding: 1rem 1.1rem 0.8rem 1.1rem;
    margin: 0.35rem 0 0 0;
}
.nace-lead-title {
    font-size: 1rem;
    font-weight: 600;
    color: #294766;
    margin-bottom: 0.2rem;
}
.nace-lead-text {
    font-size: 0.9rem;
    line-height: 1.6;
    color: #5b6b7f;
}
.nace-microcopy {
    font-size: 0.84rem;
    color: #7b8796;
    margin-top: 0.25rem;
}
.nace-card-footer-anchor {
    height: 0;
}
div[data-testid="stElementContainer"]:has(.nace-card-footer-anchor) {
    display: none;
}
div[data-testid="stElementContainer"]:has(.nace-card-footer-anchor) + div[data-testid="stElementContainer"] {
    margin-top: 0 !important;
    margin-bottom: 1rem !important;
    background: linear-gradient(135deg, #f8fbff 0%, #f1f6fd 100%);
    border: 1px solid #d9e6f4;
    border-top: none;
    border-radius: 0 0 14px 14px;
    padding: 0 1.1rem 1rem 1.1rem;
    width: 100% !important;
    box-sizing: border-box !important;
}
div[data-testid="stElementContainer"]:has(.nace-card-footer-anchor) + div[data-testid="stElementContainer"] > div {
    width: 100% !important;
}
div[data-testid="stElementContainer"]:has(.nace-card-footer-anchor) + div[data-testid="stElementContainer"] button {
    background: #eaf3ff !important;
    color: #31557a !important;
    border: 1px solid #c7d9ef !important;
    border-radius: 999px !important;
    padding: 0.5rem 1.05rem !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 2px rgba(49, 85, 122, 0.08) !important;
}
div[data-testid="stElementContainer"]:has(.nace-card-footer-anchor) + div[data-testid="stElementContainer"] button:hover {
    background: #dcecff !important;
    color: #23486d !important;
    border-color: #aecaec !important;
}
</style>
""", unsafe_allow_html=True)


# ── Data ──────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "app_deploy")
APP_DATA_DB  = os.path.join(DATA_DIR, "app_data.db")
REQ_RESP_DB  = os.path.join(DATA_DIR, "req_resp_slim.db")
TRENDS_DB    = os.path.join(DATA_DIR, "skill_trends.db")
FAISS_DIR    = os.path.join(DATA_DIR, "faiss_indexes")

@st.cache_resource
def load_faiss_indexes():
    job_index  = faiss.read_index(os.path.join(FAISS_DIR, "job_titles.index"))
    kzis_index = faiss.read_index(os.path.join(FAISS_DIR, "kzis_occupations.index"))
    with open(os.path.join(FAISS_DIR, "job_titles_metadata.pkl"), 'rb') as f:
        job_meta = pickle.load(f)
    with open(os.path.join(FAISS_DIR, "kzis_occupations_metadata.pkl"), 'rb') as f:
        kzis_meta = pickle.load(f)
    return job_index, job_meta, kzis_index, kzis_meta


@st.cache_data
def get_all_titles_sorted():
    _, meta, _, _ = load_faiss_indexes()
    titles = sorted(meta['titles'])
    return titles, [t.lower() for t in titles]


def filter_titles(query, limit=50):
    if len(query) < 3:
        return []
    titles, titles_lower = get_all_titles_sorted()
    q = query.lower()
    prefix, substring = [], []
    for t, tl in zip(titles, titles_lower):
        if tl.startswith(q):
            prefix.append(t)
        elif q in tl:
            substring.append(t)
        if len(prefix) + len(substring) >= limit:
            break
    return prefix + substring


def search_jobs_by_text(query_text, top_k=30):
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        return None, "VOYAGE_API_KEY not found in .env"
    vo = voyageai.Client(api_key=api_key)
    result = vo.embed(texts=[query_text], model="voyage-4-large", input_type="query", output_dimension=1024)
    qv = np.array(result.embeddings[0], dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(qv)
    job_index, job_meta, _, _ = load_faiss_indexes()
    distances, indices = job_index.search(qv, top_k)
    return [{'title': job_meta['titles'][i], 'similarity': float(d)} for d, i in zip(distances[0], indices[0])], None


@st.cache_data(ttl=3600)
def get_kzis_matches(job_title):
    conn = sqlite3.connect(APP_DATA_DB)
    c = conn.cursor()
    c.execute("SELECT kzis_occupation_name, similarity_score, rank FROM job_kzis_matches WHERE job_title = ? ORDER BY rank", (job_title,))
    rows = c.fetchall()
    conn.close()
    return rows


@st.cache_data(ttl=3600)
def get_job_count(job_title):
    conn = sqlite3.connect(APP_DATA_DB)
    c = conn.cursor()
    c.execute("SELECT count FROM job_title_counts WHERE title = ?", (job_title,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else 0


# ── Skills Data ───────────────────────────────────────────────

@st.cache_data
def get_sample_offers():
    conn = sqlite3.connect(REQ_RESP_DB)
    c = conn.cursor()
    c.execute("SELECT job_id, title FROM sample_offers ORDER BY title")
    rows = c.fetchall()
    conn.close()
    return rows


@st.cache_data(ttl=3600)
def get_offer_skills(job_id):
    conn = sqlite3.connect(REQ_RESP_DB)
    c = conn.cursor()
    c.execute("""
        SELECT item_type, item_text, skill_label, skill_type, similarity
        FROM skill_matches WHERE job_id = ? AND rank = 1
        ORDER BY item_type, similarity DESC
    """, (job_id,))
    top1 = c.fetchall()
    c.execute("""
        SELECT skill_label, skill_type, MAX(similarity), COUNT(*)
        FROM skill_matches WHERE job_id = ? AND rank = 1
        GROUP BY skill_label ORDER BY MAX(similarity) DESC
    """, (job_id,))
    unique_skills = c.fetchall()
    conn.close()
    return top1, unique_skills


@st.cache_data
def get_sample_titles_for_filter():
    offers = get_sample_offers()
    titles = sorted(set(t for _, t in offers))
    return titles, [t.lower() for t in titles]


def filter_sample_titles(query, limit=50):
    if len(query) < 3:
        return []
    titles, titles_lower = get_sample_titles_for_filter()
    q = query.lower()
    prefix, substring = [], []
    for t, tl in zip(titles, titles_lower):
        if tl.startswith(q):
            prefix.append(t)
        elif q in tl:
            substring.append(t)
        if len(prefix) + len(substring) >= limit:
            break
    return prefix + substring


# ── Skills Stats Data ─────────────────────────────────────────

SKILLS_CACHE_PATH = os.path.join(DATA_DIR, "skills_stats_cache.json")

def _cache_mtime():
    return os.path.getmtime(SKILLS_CACHE_PATH)

@st.cache_data
def load_skills_cache(_mtime):
    with open(SKILLS_CACHE_PATH, encoding="utf-8") as f:
        return json.load(f)


def _is_skills_code(code: str) -> bool:
    return code.startswith("S") or code.startswith("T") or code.startswith("L")


def _is_knowledge_code(code: str) -> bool:
    return len(code) > 0 and code[0].isdigit()


def _node_count(node: dict) -> int:
    return node.get("count", 0)


def build_treemap_data(cache: dict, view: str = "skills", top_leaves: int = 20):
    """
    Recursive treemap builder for nested tree with up to 4 hierarchy levels.
    Filters L1 codes by view: "skills" (S*/T*/L*) or "knowledge" (digit codes).
    """
    tree = cache["tree"]

    ids, labels, parents, values, custom = [], [], [], [], []

    def add(id_: str, label: str, parent: str, value: int, info: str = ""):
        ids.append(id_)
        labels.append(label)
        parents.append(parent)
        values.append(value)
        custom.append(info)

    def walk(node_children: dict, parent_id: str, depth: int):
        """Recursively add children. At the deepest level, only top N by count."""
        items = sorted(node_children.items(), key=lambda x: -_node_count(x[1]))
        has_deeper = any(n.get("children") for _, n in items)

        if not has_deeper:
            items = items[:top_leaves]

        for code, node in items:
            cnt = _node_count(node)
            if cnt == 0:
                continue
            node_id = f"{parent_id}/{code}"
            children = node.get("children", {})

            if children and has_deeper:
                add(node_id, f"{code}  {node['title']}", parent_id, cnt,
                    f"{code} · {cnt:,}")
                walk(children, node_id, depth + 1)
            else:
                add(node_id, node["title"], parent_id, cnt, f"{cnt:,}")

    is_target = _is_skills_code if view == "skills" else _is_knowledge_code

    matched_codes = [c for c in tree if is_target(c)]
    lang_codes = [c for c in matched_codes if c.startswith("L")] if view == "skills" else []
    regular_codes = [c for c in matched_codes if c not in lang_codes]

    def sort_key(code):
        if code.startswith("S"):
            return (0, code)
        elif code.startswith("T"):
            return (1, code)
        else:
            return (2, code)

    # Compute "Other" total first so root value includes it
    other_codes = [c for c in tree if not _is_skills_code(c) and not _is_knowledge_code(c)]
    other_total = sum(_node_count(tree[c]) for c in other_codes)
    other_total += cache["meta"].get("unmatched_mentions", 0)

    ROOT = "root"
    total_view = sum(_node_count(tree[c]) for c in matched_codes) + other_total
    root_label = "Skills & Competences" if view == "skills" else "Knowledge"
    add(ROOT, root_label, "", total_view, f"{total_view:,} mentions")

    for l1_code in sorted(regular_codes, key=sort_key):
        l1_node = tree[l1_code]
        cnt = _node_count(l1_node)
        l1_id = f"{ROOT}/{l1_code}"
        add(l1_id, f"{l1_code}  {l1_node['title']}", ROOT, cnt,
            f"{l1_code} · {cnt:,}")
        walk(l1_node.get("children", {}), l1_id, 2)

    if lang_codes:
        lang_total = sum(_node_count(tree[lc]) for lc in lang_codes)
        lang_id = f"{ROOT}/LANG"
        add(lang_id, "Languages", ROOT, lang_total, f"{lang_total:,} mentions")
        for lc in sorted(lang_codes):
            lc_node = tree[lc]
            lc_cnt = _node_count(lc_node)
            add(f"{lang_id}/{lc}", lc_node["title"], lang_id, lc_cnt, f"{lc_cnt:,}")

    if other_total > 0:
        add(f"{ROOT}/other", "Other", ROOT, other_total, f"{other_total:,} mentions")

    return ids, labels, parents, values, custom


# ── Skill Trends Data ─────────────────────────────────────────

@st.cache_data(ttl=3600)
def get_period_totals():
    conn = sqlite3.connect(TRENDS_DB)
    c = conn.cursor()
    c.execute("SELECT period, total_offers FROM period_totals ORDER BY period")
    rows = c.fetchall()
    conn.close()
    return {p: t for p, t in rows}


@st.cache_data(ttl=3600)
def search_skill_labels(query: str, limit: int = 30):
    if len(query) < 2:
        return []
    conn = sqlite3.connect(TRENDS_DB)
    c = conn.cursor()
    c.execute(
        "SELECT skill_id, label, total_mentions, total_offers "
        "FROM skill_labels WHERE label LIKE ? "
        "ORDER BY total_mentions DESC LIMIT ?",
        (f"%{query}%", limit),
    )
    rows = c.fetchall()
    conn.close()
    return rows


@st.cache_data(ttl=3600)
def get_skill_trend(skill_id: int):
    conn = sqlite3.connect(TRENDS_DB)
    c = conn.cursor()
    c.execute(
        "SELECT period, mention_count, offer_count "
        "FROM skill_period WHERE skill_id = ? ORDER BY period",
        (skill_id,),
    )
    rows = c.fetchall()
    conn.close()
    return rows


@st.cache_data(ttl=3600)
def get_top_skills(limit: int = 200):
    conn = sqlite3.connect(TRENDS_DB)
    c = conn.cursor()
    c.execute(
        "SELECT skill_id, label, total_mentions, total_offers "
        "FROM skill_labels ORDER BY total_mentions DESC LIMIT ?",
        (limit,),
    )
    rows = c.fetchall()
    conn.close()
    return rows


# ── Rendering ─────────────────────────────────────────────────

def get_score_color(score):
    """Return (bg_color, text_color) based on similarity score."""
    score_pct = score * 100
    if score_pct >= 90:
        return "#2d5a2d", "#fff"  # dark green
    elif score_pct >= 80:
        return "#4a7a4a", "#fff"  # green
    elif score_pct >= 75:
        return "#76b076", "#fff"  # light green
    else:
        return "#d97d3a", "#fff"  # orange


def build_kzis_html(kzis_matches):
    if not kzis_matches:
        return ""
    rows = ""
    for name, sim, rank in kzis_matches:
        bg, fg = get_score_color(sim)
        rows += f'<div class="kr"><span class="kr-n">{rank}. {name}</span><span class="kr-s" style="background:{bg};color:{fg}">{sim:.2f}</span></div>'
    return f'<div class="kzis-sec"><div class="kzis-lbl">Standardized KZIS Occupations</div>{rows}</div>'


def render_card(title, count, similarity=None, kzis_matches=None):
    meta = f"{count} job offer{'s' if count != 1 else ''}"
    if similarity is not None:
        meta += f" &middot; match: {similarity:.2f}"
    kzis_html = build_kzis_html(kzis_matches)
    st.markdown(f'<div class="rcard"><div class="rcard-title">{title}</div><div class="rcard-meta">{meta}</div>{kzis_html}</div>', unsafe_allow_html=True)


def render_offer_with_skills(job_id, title):
    count = get_job_count(title)
    kzis = get_kzis_matches(title)
    top1_items, unique_skills = get_offer_skills(job_id)

    kzis_html = build_kzis_html(kzis)
    st.markdown(f'<div class="rcard"><div class="rcard-title">{title}</div><div class="rcard-meta">{count} job offer{"s" if count != 1 else ""}</div>{kzis_html}</div>', unsafe_allow_html=True)

    # Unique skills
    if unique_skills:
        html = ""
        for label, stype, sim, cnt in unique_skills[:20]:
            tag = (stype or "").replace("skill/competence", "competence")
            bg, fg = get_score_color(sim)
            html += f'<div class="skill-row"><span class="skill-name">{label}</span><span class="skill-type-tag">{tag}</span><span class="skill-score" style="background:{bg};color:{fg}">{sim:.2f}</span></div>'
        extra = len(unique_skills) - 20
        if extra > 0:
            html += f'<div class="info-box">+ {extra} more skills</div>'
        st.markdown(f'<div class="sec-label">Mapped ESCO Skills ({len(unique_skills)} unique)</div>{html}', unsafe_allow_html=True)

    # Items
    if top1_items:
        req = [(t, s, sim) for typ, t, s, _, sim in top1_items if typ == "requirement"]
        resp = [(t, s, sim) for typ, t, s, _, sim in top1_items if typ == "responsibility"]

        if req:
            rows = ""
            for text, skill, sim in req:
                bg, fg = get_score_color(sim)
                rows += f'<div class="item-row"><span class="item-type item-type-req">req</span>{text}<div class="item-skill">&#8594; <span class="item-skill-name">{skill}</span> <span class="skill-score" style="background:{bg};color:{fg}">{sim:.2f}</span></div></div>'
            st.markdown(f'<div class="sec-label">Requirements ({len(req)})</div><div class="rcard">{rows}</div>', unsafe_allow_html=True)

        if resp:
            rows = ""
            for text, skill, sim in resp:
                bg, fg = get_score_color(sim)
                rows += f'<div class="item-row"><span class="item-type item-type-resp">resp</span>{text}<div class="item-skill">&#8594; <span class="item-skill-name">{skill}</span> <span class="skill-score" style="background:{bg};color:{fg}">{sim:.2f}</span></div></div>'
            st.markdown(f'<div class="sec-label">Responsibilities ({len(resp)})</div><div class="rcard">{rows}</div>', unsafe_allow_html=True)


# ── NACE Methodology Dialog ───────────────────────────────────

@st.dialog("NACE Assignment — Methodology", width="large")
def _show_nace_methodology():
    st.markdown("""
### Overview

This workflow assigns each job ad to a **NACE Rev. 2 economic sector**
so labour demand can be analysed not only by occupation, but also by the
type of economic activity behind the vacancy.

Job ads do not contain NACE codes directly. We therefore infer them in three steps:
we identify the likely occupation, retrieve candidate sectors from the official EU
crosswalk, and then use the text of the ad to choose the best-fitting sector for that
specific offer.

---

### How the assignment works
""", unsafe_allow_html=True)

    st.markdown("""
<div style="
  display:flex; align-items:stretch; justify-content:center;
  gap:0.6rem; margin:0.7rem 0 1.5rem 0; flex-wrap:nowrap;
">
  <div style="
    flex:1; text-align:center; padding:1rem 0.95rem;
    background:linear-gradient(135deg, #f7fbff 0%, #eaf3ff 100%);
    border:1px solid #d6e5f5; border-radius:14px;
    box-shadow:0 6px 16px rgba(59,130,246,0.08);
  ">
    <div style="font-size:1.55rem; color:#5a7ea6; margin-bottom:0.35rem;">1</div>
    <div style="font-weight:600; font-size:0.96rem; color:#3f648b;">Match the occupation</div>
    <div style="font-size:0.84rem; color:#617285; margin-top:0.35rem; line-height:1.45;">
      The job title is matched to the<br><b>closest ESCO occupation</b><br>
      using contextual meaning
    </div>
  </div>
  <div style="display:flex; align-items:center; font-size:1.25rem; color:#a1b2c3;">›</div>
  <div style="
    flex:1; text-align:center; padding:1rem 0.95rem;
    background:linear-gradient(135deg, #f7fcf8 0%, #eaf7ee 100%);
    border:1px solid #d7eadb; border-radius:14px;
    box-shadow:0 6px 16px rgba(34,197,94,0.08);
  ">
    <div style="font-size:1.55rem; color:#62916e; margin-bottom:0.35rem;">2</div>
    <div style="font-weight:600; font-size:0.96rem; color:#467352;">Get candidate sectors</div>
    <div style="font-size:0.84rem; color:#617285; margin-top:0.35rem; line-height:1.45;">
      The occupation is linked to one or more<br><b>NACE sector candidates</b><br>
      using the official EU crosswalk
    </div>
  </div>
  <div style="display:flex; align-items:center; font-size:1.25rem; color:#a1b2c3;">›</div>
  <div style="
    flex:1; text-align:center; padding:1rem 0.95rem;
    background:linear-gradient(135deg, #fffaf4 0%, #ffedd8 100%);
    border:1px solid #f1d9b8; border-radius:14px;
    box-shadow:0 6px 16px rgba(249,115,22,0.08);
  ">
    <div style="font-size:1.55rem; color:#b07a45; margin-bottom:0.35rem;">3</div>
    <div style="font-weight:600; font-size:0.96rem; color:#8d6232;">Resolve ambiguity</div>
    <div style="font-size:0.84rem; color:#617285; margin-top:0.35rem; line-height:1.45;">
      If there is more than one candidate,<br>the offer's <b>responsibilities</b><br>
      are used to choose the best sector
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
---

### Why contextual embeddings matter

Contextual embeddings do not read a title as an isolated string.
They place it in context and encode meaning based on the surrounding language.
This is especially useful for short, generic or English-language titles that are easy
to misread if we rely only on surface similarity.
""")

    _EXAMPLES_CTX = [
        {"title":"Inżynier robót","title_en":"Site Works Engineer",
         "responsibilities":["coordinate construction works & supervision","manage subcontractors on-site","prepare as-built documentation","quality & safety inspections","resolve on-site technical issues"],
         "match_plain":"robotics engineer","match_plain_pl":"inżynier robotyki",
         "match_ctx":"construction engineer","match_ctx_pl":"inżynier budownictwa"},
        {"title":"Data Science Specialist (Energy Market)","title_en":"Specjalista ds. Data Science (Rynek Energii)",
         "responsibilities":["model energy price forecasts with ML","build predictive pipelines for demand/RES","deploy ML models to production","scenario and what-if simulations","ensure data quality for model inputs"],
         "match_plain":"energy trading specialist","match_plain_pl":"specjalista ds. handlu energią",
         "match_ctx":"data scientist","match_ctx_pl":"specjalista ds. inteligentnej analizy danych"},
        {"title":"Analityk Business Intelligence","title_en":"BI Analyst",
         "responsibilities":["explore and visualize large datasets","identify trends and KPI deviations","build automated reporting dashboards","maintain data consistency and completeness","develop data quality improvement strategies"],
         "match_plain":"business analyst","match_plain_pl":"analityk biznesowy",
         "match_ctx":"data analyst","match_ctx_pl":"analityk danych"},
        {"title":"Finance Manager (Kontroler Finansowy)","title_en":"Finance Manager (Financial Controller)",
         "responsibilities":["supervise external accounting firm","manage annual audit process","prepare monthly financial statements","verify accuracy of accounting books","budget preparation and forecasting"],
         "match_plain":"financial director (CFO)","match_plain_pl":"dyrektor finansowy",
         "match_ctx":"financial controller","match_ctx_pl":"kontroler finansowy"},
        {"title":"Asystent","title_en":"Assistant",
         "responsibilities":["assist dentist during clinical procedures","prepare treatment room and instruments","patient scheduling and record management","dental X-rays and treatment documentation","sterilization and infection control"],
         "match_plain":"assistant (academic)","match_plain_pl":"asystent (akademicki)",
         "match_ctx":"dental assistant","match_ctx_pl":"asystent stomatologiczny"},
        {"title":"Specjalista ds. HR (Dział Spraw Pracowniczych)","title_en":"HR Specialist (Employee Affairs)",
         "responsibilities":["run end-to-end recruitment for technical & admin roles","source candidates and manage candidate experience","coordinate annual employee training plan","support HR business partners and managers","manage employee benefits administration"],
         "match_plain":"HR department manager","match_plain_pl":"kierownik działu zarządzania zasobami ludzkimi",
         "match_ctx":"recruitment specialist","match_ctx_pl":"specjalista ds. rekrutacji pracowników"},
        {"title":"Risk Manager","title_en":"Risk Manager",
         "responsibilities":["assess and monitor financial risk exposures","credit risk modelling and portfolio analysis","prepare risk reports for senior management","ensure Basel / IFRS9 compliance","stress testing and scenario analysis"],
         "match_plain":"corporate risk manager","match_plain_pl":"menedżer ryzyka korporacyjnego",
         "match_ctx":"financial risk manager","match_ctx_pl":"menedżer ryzyka finansowego"},
        {"title":"Kierownik produkcji / Production Manager","title_en":"Production Manager",
         "responsibilities":["manage kitchen, packaging and warehouse departments","ensure food safety and HACCP compliance","production planning based on sales forecasts","Lean / 5S / Kaizen process optimisation","CAPEX projects — new production lines"],
         "match_plain":"production manager","match_plain_pl":"kierownik ds. produkcji",
         "match_ctx":"food production manager","match_ctx_pl":"kierownik produkcji żywności"},
        {"title":"Specjalista ds. Data Governance","title_en":"Data Governance Specialist",
         "responsibilities":["define data quality rules and validation logic","maintain data lineage, catalog and metadata","identify and remediate data quality issues","develop governance frameworks and policies","collaborate with BI and data engineering teams"],
         "match_plain":"data management specialist","match_plain_pl":"specjalista ds. zarządzania danymi",
         "match_ctx":"data quality specialist","match_ctx_pl":"specjalista ds. zapewnienia jakości danych"},
        {"title":"IT Analityk Biznesowy","title_en":"IT Business Analyst",
         "responsibilities":["gather and analyse client business requirements","translate requirements into user stories and use cases","create and maintain the development backlog","model business processes and system architecture","support developers with domain clarifications"],
         "match_plain":"ICT business analyst","match_plain_pl":"analityk biznesowy w dziedzinie ICT",
         "match_ctx":"ICT systems analyst","match_ctx_pl":"analityk systemów informacyjno-telekomunikacyjnych"},
    ]

    def _pill_ctx(text: str) -> str:
        return (
            f'<span style="display:inline-block;background:#deeaf7;color:#2d5e8e;'
            f'border:1px solid #b8d4ee;border-radius:999px;padding:0.17rem 0.52rem;'
            f'font-size:0.72rem;font-weight:500;margin:0.15rem 0.12rem 0 0;'
            f'line-height:1.4;white-space:nowrap;">{text}</span>'
        )

    _ctx_scroll = (
        '<div style="display:flex;gap:0;overflow-x:auto;'
        'padding:0.3rem 0 1.1rem 0;scroll-snap-type:x mandatory;'
        '-webkit-overflow-scrolling:touch;scroll-behavior:smooth;">'
    )
    for _ex in _EXAMPLES_CTX:
        _pills_row = "".join(_pill_ctx(r) for r in _ex["responsibilities"])
        _ctx_scroll += (
            f'<div style="min-width:100%;max-width:100%;flex-shrink:0;'
            f'scroll-snap-align:start;display:flex;gap:0.65rem;align-items:stretch;padding:0 0.1rem;">'
            f'<div style="flex:1;background:#fff7f8;border:1px solid #f1d6dc;'
            f'border-radius:14px;padding:0.9rem 1rem;display:flex;'
            f'flex-direction:column;gap:0.5rem;">'
            f'<div style="font-size:0.67rem;font-weight:700;letter-spacing:0.08em;'
            f'text-transform:uppercase;color:#ad6a78;">Without context</div>'
            f'<div style="font-size:0.94rem;font-weight:700;color:#5a3a48;line-height:1.3;">'
            f'{_ex["title"]}'
            f'<span style="font-weight:400;color:#b08898;font-size:0.76rem;'
            f'display:block;margin-top:0.07rem;">{_ex["title_en"]}</span></div>'
            f'<div style="flex:1"></div>'
            f'<div style="background:#fff;border:1px solid #efd9de;border-radius:9px;'
            f'padding:0.55rem 0.65rem;">'
            f'<div style="font-size:0.63rem;color:#aa7580;text-transform:uppercase;'
            f'letter-spacing:0.06em;margin-bottom:0.15rem;">Title-only match</div>'
            f'<div style="font-size:0.9rem;font-weight:700;color:#7b4a5a;">'
            f'{_ex["match_plain"]}</div>'
            f'<div style="font-size:0.74rem;color:#a07585;font-style:italic;'
            f'margin-top:0.05rem;">{_ex["match_plain_pl"]}</div>'
            f'</div>'
            f'</div>'
            f'<div style="flex:1.3;background:#f7fbff;border:1px solid #d8e6f3;'
            f'border-radius:14px;padding:0.9rem 1rem;display:flex;'
            f'flex-direction:column;gap:0.5rem;">'
            f'<div style="font-size:0.67rem;font-weight:700;letter-spacing:0.08em;'
            f'text-transform:uppercase;color:#5d83ab;">With contextual embeddings</div>'
            f'<div style="font-size:0.94rem;font-weight:700;color:#2c4f72;line-height:1.3;">'
            f'{_ex["title"]}'
            f'<span style="font-weight:400;color:#8099b2;font-size:0.76rem;'
            f'display:block;margin-top:0.07rem;">{_ex["title_en"]}</span></div>'
            f'<div style="background:#fff;border:1px solid #dbe7f4;border-radius:9px;'
            f'padding:0.55rem 0.65rem;">'
            f'<div style="font-size:0.63rem;color:#72869b;text-transform:uppercase;'
            f'letter-spacing:0.06em;margin-bottom:0.2rem;">Responsibilities</div>'
            f'<div style="line-height:1.8;">{_pills_row}</div>'
            f'</div>'
            f'<div style="background:#fff;border:1px solid #dbe7f4;border-radius:9px;'
            f'padding:0.55rem 0.65rem;margin-top:auto;">'
            f'<div style="font-size:0.63rem;color:#72869b;text-transform:uppercase;'
            f'letter-spacing:0.06em;margin-bottom:0.15rem;">Context-aware match ✓</div>'
            f'<div style="font-size:0.9rem;font-weight:700;color:#1e4d7a;">'
            f'{_ex["match_ctx"]}</div>'
            f'<div style="font-size:0.74rem;color:#5a7a9a;font-style:italic;'
            f'margin-top:0.05rem;">{_ex["match_ctx_pl"]}</div>'
            f'</div>'
            f'</div>'
            f'</div>'
        )
    _ctx_scroll += '</div>'
    st.markdown(_ctx_scroll, unsafe_allow_html=True)

    st.markdown("""
In other words, the model does not rely only on word overlap.
It reads the **meaning of the title in context**, which gives a more reliable occupational
match before the NACE sector is assigned.

This is particularly helpful when:

- titles are short or generic, such as *specialist*, *manager*, *analyst* or *installer*;
- titles are in English while the rest of the ad is in Polish;
- one occupation can reasonably appear in several sectors, so sector choice must be made at the offer level.

---

### Example of NACE disambiguation

Once the occupation is identified, some occupations still map to more than one economic sector.
In those cases, the final sector is selected at the **job-ad level** using the responsibilities text.
This means that two ads with the same title can still receive different NACE assignments if the work context differs.
""")

    st.markdown("""
<div style="
  display:flex; gap:0.9rem; align-items:stretch; margin:0.7rem 0 1rem 0;
">
  <div style="
    flex:1.05; background:#f8fbff; border:1px solid #d9e6f4; border-radius:14px;
    padding:1rem 1rem 0.95rem 1rem;
  ">
    <div style="font-size:0.74rem; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:#5d83ab;">Offer-level example</div>
    <div style="font-size:1.02rem; font-weight:600; color:#3e648a; margin-top:0.45rem;">Job title: “Cloud Engineer” (<i>Inżynier Chmurowy</i>)</div>
    <div style="font-size:0.88rem; line-height:1.58; color:#607184; margin-top:0.55rem;">
      Responsibilities mention cloud architecture, migration, client needs,
      landing zones and solution design.
    </div>
    <div style="
      margin-top:0.75rem; background:#fff; border:1px solid #dbe7f4; border-radius:10px;
      padding:0.75rem 0.8rem;
    ">
      <div style="font-size:0.74rem; color:#72869b; text-transform:uppercase; letter-spacing:0.06em;">Matched occupation</div>
      <div style="font-size:0.95rem; font-weight:600; color:#3e648a; margin-top:0.18rem;">cloud engineer (<i>inżynier ds. chmury</i>)</div>
    </div>
  </div>
  <div style="
    flex:1.2; background:#fffaf4; border:1px solid #f1d9b8; border-radius:14px;
    padding:1rem 1rem 0.95rem 1rem;
  ">
    <div style="font-size:0.74rem; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:#ad7a4f;">Sector choice</div>
    <div style="
      margin-top:0.45rem; background:#fff; border:1px solid #f2e1c7; border-radius:10px;
      padding:0.75rem 0.8rem;
    ">
      <div style="font-size:0.74rem; color:#9b7f62; text-transform:uppercase; letter-spacing:0.06em;">Candidate NACE sectors from crosswalk</div>
      <div style="font-size:0.89rem; line-height:1.58; color:#667587; margin-top:0.25rem;">
        <b>K62.1</b> Programming (<i>Działalność w zakresie programowania</i>)<br>
        <b>K62.2</b> IT consultancy and infrastructure management
        (<i>Działalność związana z doradztwem w zakresie informatyki...</i>)
      </div>
    </div>
    <div style="
      margin-top:0.75rem; background:#fff; border:1px solid #f2e1c7; border-radius:10px;
      padding:0.75rem 0.8rem;
    ">
      <div style="font-size:0.74rem; color:#9b7f62; text-transform:uppercase; letter-spacing:0.06em;">Assigned sector</div>
      <div style="font-size:0.95rem; font-weight:600; color:#8d6232; margin-top:0.18rem;">
        K62.2 — IT consultancy and infrastructure management
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
Here the final choice is not made from the title alone.
The responsibilities point to advisory and architecture work with clients and cloud environments,
which makes the consultancy-oriented sector a better fit than general programming.
""")

    st.markdown("""---

### Technical details

| Component | Detail |
|:----------|:-------|
| **Occupation matching** | Contextual embeddings over 3,043 ESCO occupations with cosine similarity search in FAISS |
| **Sector candidates** | Official EU ESCO–NACE Rev. 2 crosswalk (4,564 mappings; 706 unique NACE codes) |
| **Offer-level disambiguation** | Embedding similarity between each offer's responsibilities and Polish NACE sector descriptions |
| **Coverage** | 100% of 749,569 job ads assigned; 64.3% direct (single NACE), 35.7% disambiguated |
| **Models used** | ESCO matching: `voyage-context-3` (2048 dimensions); sector disambiguation: `voyage-4` (2048 dimensions) |
""")


# ── Main ──────────────────────────────────────────────────────

def main():
    st.markdown(
        '<div class="app-header">'
        '<div class="app-title">Skill Demand and Supply in Poland 2025</div>'
        '<div class="app-sub">KZIS occupations, ESCO skills, NACE sectors</div>'
        '</div>',
        unsafe_allow_html=True
    )

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Job titles", "Skills Search", "Skills Stats", "Skill Trends", "NACE",
        "UA-Targeted", "AI Skills",
    ])

    # ── Tab 1: Job Titles ──
    with tab1:
        # KZIS Categories Chart
        st.markdown('<div class="sec-label">Job Offers by KZIS Category</div>', unsafe_allow_html=True)
        st.components.v1.iframe(
            "https://datawrapper.dwcdn.net/U2ao5/11/",
            height=320,
            scrolling=False
        )
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        
        st.markdown('<div class="sec-label">Search job offers by title or keyword</div>', unsafe_allow_html=True)

        query = st.text_input(
            "Search query",
            placeholder="Type a job title or keyword, e.g. pielęgniarz, data scientist...",
            label_visibility="collapsed"
        )

        col1, col2 = st.columns([1, 5])
        with col1:
            go = st.button("Search", type="primary", use_container_width=True)

        if go and query:
            with st.spinner("Searching..."):
                results, err = search_jobs_by_text(query, top_k=30)
            if err:
                st.error(err)
            elif results:
                seen = {}
                for r in results:
                    tl = r['title'].lower()
                    if tl not in seen or r['similarity'] > seen[tl]['similarity']:
                        seen[tl] = r

                enriched = []
                total_matching = 0
                for data in seen.values():
                    t = data['title']
                    cnt = get_job_count(t)
                    total_matching += cnt
                    enriched.append((t, cnt, data['similarity'], get_kzis_matches(t)))
                enriched.sort(key=lambda x: x[2], reverse=True)

                st.markdown(f'<div class="sec-label">Matching job offers &mdash; {total_matching:,} offers across {len(enriched)} titles</div>', unsafe_allow_html=True)

                for t, cnt, sim, kzis in enriched[:5]:
                    render_card(t, cnt, sim, kzis)

                if len(enriched) > 5:
                    if st.button(f"Show {len(enriched) - 5} more results", use_container_width=True):
                        for t, cnt, sim, kzis in enriched[5:]:
                            render_card(t, cnt, sim, kzis)
            else:
                st.markdown('<div class="info-box">No results found.</div>', unsafe_allow_html=True)

    # ── Tab 2: Skills ──
    with tab2:
        # Top Skills Chart
        st.markdown('<div class="sec-label">Top Skills in Job Offers</div>', unsafe_allow_html=True)
        st.components.v1.iframe(
            "https://datawrapper.dwcdn.net/FMGJD/5/",
            height=320,
            scrolling=False
        )
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        
        st.markdown('<div class="sec-label">Explore job offers with mapped ESCO skills</div>', unsafe_allow_html=True)

        all_titles = get_sample_titles_for_filter()[0]
        
        chosen_title = st.selectbox(
            "Select an offer",
            [""] + all_titles,
            index=0,
            placeholder="Start typing to filter 4,000 sample offers...",
            label_visibility="collapsed",
            key="skills_pick"
        )

        if chosen_title:
            offers = get_sample_offers()
            matching = [(jid, t) for jid, t in offers if t == chosen_title]
            if matching:
                job_id, title = matching[0]
                render_offer_with_skills(job_id, title)

    # ── Tab 3: Skills Stats ──
    with tab3:
        cache = load_skills_cache(_cache_mtime())
        meta = cache["meta"]

        matched_pct = meta["matched_mentions"] / meta["total_mentions"] * 100

        st.markdown(
            f'<div class="info-box">'
            f'<b>{meta["total_mentions"]:,}</b> ESCO skill mentions across '
            f'<b>{meta["total_offers"]:,}</b> job offers &mdash; '
            f'<b>{meta["matched_mentions"]:,}</b> ({matched_pct:.0f}%) '
            f'matched to ESCO hierarchy.<br>'
            f'<span style="font-size:0.8em;color:#888">Click a tile to drill down. '
            f'Click the header bar to go back up.</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        view = st.radio(
            "View",
            ["Skills & Competences", "Knowledge"],
            horizontal=True,
            label_visibility="collapsed",
            key="stats_view",
        )
        view_key = "skills" if view == "Skills & Competences" else "knowledge"

        ids, labels_tm, parents, values, custom = build_treemap_data(cache, view=view_key)

        VIEW_COLORS = {
            "skills": {
                "root": "#1a1a2e",
                "S": "#3a6b8c",
                "T": "#2d6a2d",
                "LANG": "#6a2d5a",
            },
            "knowledge": {
                "root": "#1a1a2e",
                "default": "#7a5a1a",
            },
        }
        palette = VIEW_COLORS[view_key]

        node_colors = []
        for node_id in ids:
            if node_id == "root":
                node_colors.append(palette["root"])
            else:
                l1_part = node_id.split("/")[1] if "/" in node_id else ""
                if l1_part == "other":
                    node_colors.append("#999")
                elif view_key == "skills":
                    if l1_part.startswith("S"):
                        node_colors.append(palette["S"])
                    elif l1_part.startswith("T"):
                        node_colors.append(palette["T"])
                    else:
                        node_colors.append(palette["LANG"])
                else:
                    node_colors.append(palette["default"])

        fig = pgo.Figure(pgo.Treemap(
            ids=ids,
            labels=labels_tm,
            parents=parents,
            values=values,
            customdata=custom,
            branchvalues="total",
            marker=dict(
                colors=node_colors,
                line=dict(width=1.5, color="#fff"),
            ),
            hovertemplate="<b>%{label}</b><br>%{customdata}<extra></extra>",
            textinfo="label+percent parent",
            textfont=dict(size=13),
            maxdepth=3,
            pathbar=dict(visible=True),
        ))
        fig.update_layout(
            height=700,
            margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor="#fff",
            font=dict(size=12, color="#333"),
        )
        st.plotly_chart(fig, width="stretch")

    # ── Tab 4: Skill Trends ──
    with tab4:
        st.markdown('<div class="sec-label">Search for a skill or knowledge area to see weekly trends</div>', unsafe_allow_html=True)

        top_skills = get_top_skills(500)
        all_options = [f"{label}  ({total_m:,})" for sid, label, total_m, _ in top_skills]
        all_map = {opt: (sid, label) for opt, (sid, label, _, _) in zip(all_options, top_skills)}

        chosen = st.selectbox(
            "Select skill",
            [""] + all_options,
            index=0,
            placeholder="Type to search skills, e.g. Python, zarządzanie, Excel…",
            label_visibility="collapsed",
            key="trend_pick",
        )

        sid = None
        label = None
        if chosen and chosen in all_map:
            sid, label = all_map[chosen]

        if sid is not None:
            trend = get_skill_trend(sid)
            period_totals = get_period_totals()

            if trend:
                weeks = [r[0] for r in trend]
                mentions = [r[1] for r in trend]
                pct = [r[2] / period_totals.get(r[0], 1) * 100 for r in trend]
                week_labels = [w.replace("2025-", "") for w in weeks]

                # 4-week rolling average (≈ monthly smoothing)
                WIN = 4
                def rolling_avg(series, window=WIN):
                    out = []
                    for i in range(len(series)):
                        start = max(0, i - window + 1)
                        out.append(sum(series[start:i+1]) / (i - start + 1))
                    return out

                smooth_mentions = rolling_avg(mentions)
                smooth_pct = rolling_avg(pct)

                FORECAST_N = 4
                def arima_forecast(series, steps=FORECAST_N):
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            model = ARIMA(series, order=(1, 1, 1))
                            fit = model.fit()
                            fc = fit.forecast(steps=steps)
                            ci = fit.get_forecast(steps=steps).conf_int(alpha=0.2)
                            return list(fc), list(ci[:, 0]), list(ci[:, 1])
                    except Exception:
                        return None, None, None

                fc_labels = ["W01*", "W02*", "W03*", "W04*"]
                fc_m_vals, fc_lo_m, fc_hi_m = arima_forecast(smooth_mentions)
                fc_p_vals, fc_lo_p, fc_hi_p = arima_forecast(smooth_pct)

                st.markdown(f'<div class="rcard-title" style="margin-top:1rem">{label}</div>', unsafe_allow_html=True)

                TICK = dict(size=14, color="#000")
                AXIS_TITLE = dict(size=14, color="#000")
                CHART_FONT = dict(size=14, color="#000")

                # ── Mentions chart ──
                fig = pgo.Figure()
                fig.add_trace(pgo.Scatter(
                    x=week_labels, y=mentions,
                    name="Weekly",
                    mode="lines+markers",
                    line=dict(color="rgba(58,107,140,0.25)", width=1),
                    marker=dict(size=3, color="rgba(58,107,140,0.35)"),
                    hovertemplate="%{x}<br><b>%{y:,}</b> mentions<extra></extra>",
                ))
                fig.add_trace(pgo.Scatter(
                    x=week_labels, y=smooth_mentions,
                    name="Smoothed (4w avg)",
                    mode="lines",
                    line=dict(color="#3a6b8c", width=3.5),
                    hovertemplate="%{x}<br><b>%{y:,.0f}</b> (4w avg)<extra></extra>",
                ))
                if fc_m_vals is not None:
                    ci_x = [week_labels[-1]] + fc_labels + fc_labels[::-1] + [week_labels[-1]]
                    ci_y = [smooth_mentions[-1]] + fc_hi_m + fc_lo_m[::-1] + [smooth_mentions[-1]]
                    fig.add_trace(pgo.Scatter(
                        x=ci_x, y=ci_y,
                        fill="toself", fillcolor="rgba(58,107,140,0.18)",
                        line=dict(width=0), mode="none", name="Forecast range",
                        hoverinfo="skip",
                    ))
                all_m = mentions + (fc_hi_m if fc_hi_m else [])
                y_lo_m = min(mentions + (fc_lo_m if fc_lo_m else [])) * 0.85
                y_hi_m = max(all_m) * 1.08
                fig.update_layout(
                    title=dict(text="Mentions", font=dict(size=16, color="#111")),
                    height=370,
                    margin=dict(l=60, r=20, t=45, b=70),
                    plot_bgcolor="#fff", paper_bgcolor="#fff",
                    font=CHART_FONT,
                    xaxis=dict(showgrid=False, tickfont=TICK, tickangle=-45),
                    yaxis=dict(showgrid=True, gridcolor="#e0e0e0",
                               title=dict(text="Mentions", font=AXIS_TITLE),
                               tickfont=TICK, range=[y_lo_m, y_hi_m]),
                    hovermode="x unified",
                    legend=dict(orientation="h", y=1.12, x=0, font=dict(size=13, color="#000")),
                )
                st.plotly_chart(fig, width="stretch")

                # ── % chart ──
                fig2 = pgo.Figure()
                fig2.add_trace(pgo.Scatter(
                    x=week_labels, y=pct,
                    name="Weekly",
                    mode="lines+markers",
                    line=dict(color="rgba(45,106,45,0.25)", width=1),
                    marker=dict(size=3, color="rgba(45,106,45,0.35)"),
                    hovertemplate="%{x}<br><b>%{y:.2f}%</b> of offers<extra></extra>",
                ))
                fig2.add_trace(pgo.Scatter(
                    x=week_labels, y=smooth_pct,
                    name="Smoothed (4w avg)",
                    mode="lines",
                    line=dict(color="#2d6a2d", width=3.5),
                    hovertemplate="%{x}<br><b>%{y:.2f}%</b> (4w avg)<extra></extra>",
                ))
                if fc_p_vals is not None:
                    ci_x = [week_labels[-1]] + fc_labels + fc_labels[::-1] + [week_labels[-1]]
                    ci_y = [smooth_pct[-1]] + fc_hi_p + fc_lo_p[::-1] + [smooth_pct[-1]]
                    fig2.add_trace(pgo.Scatter(
                        x=ci_x, y=ci_y,
                        fill="toself", fillcolor="rgba(45,106,45,0.18)",
                        line=dict(width=0), mode="none", name="Forecast range",
                        hoverinfo="skip",
                    ))
                all_p = pct + (fc_hi_p if fc_hi_p else [])
                y_lo_p = min(pct + (fc_lo_p if fc_lo_p else [])) * 0.85
                y_hi_p = max(all_p) * 1.08
                fig2.update_layout(
                    title=dict(text="% of job offers", font=dict(size=16, color="#111")),
                    height=370,
                    margin=dict(l=60, r=20, t=45, b=70),
                    plot_bgcolor="#fff", paper_bgcolor="#fff",
                    font=CHART_FONT,
                    xaxis=dict(showgrid=False, tickfont=TICK, tickangle=-45),
                    yaxis=dict(showgrid=True, gridcolor="#e0e0e0",
                               title=dict(text="% of offers", font=AXIS_TITLE),
                               tickfont=TICK, range=[y_lo_p, y_hi_p]),
                    hovermode="x unified",
                    legend=dict(orientation="h", y=1.12, x=0, font=dict(size=13, color="#000")),
                )
                st.plotly_chart(fig2, width="stretch")

                # ── Export to .dta ──
                df_export = pd.DataFrame({
                    "week": weeks,
                    "mentions": mentions,
                    "mentions_smooth_4w": smooth_mentions,
                    "offer_count": [r[2] for r in trend],
                    "total_offers": [period_totals.get(r[0], 0) for r in trend],
                    "pct_offers": pct,
                    "pct_offers_smooth_4w": smooth_pct,
                })
                if fc_m_vals is not None:
                    df_fc = pd.DataFrame({
                        "week": fc_labels,
                        "fc_mentions_smooth": fc_m_vals,
                        "fc_mentions_lo": fc_lo_m,
                        "fc_mentions_hi": fc_hi_m,
                        "fc_pct_smooth": fc_p_vals if fc_p_vals else [None]*FORECAST_N,
                        "fc_pct_lo": fc_lo_p if fc_lo_p else [None]*FORECAST_N,
                        "fc_pct_hi": fc_hi_p if fc_hi_p else [None]*FORECAST_N,
                    })
                    df_export = pd.concat([df_export, df_fc], ignore_index=True)

                buf = io.BytesIO()
                df_export.to_stata(buf, write_index=False, version=118)
                buf.seek(0)
                safe_label = label.replace(" ", "_").replace("/", "_")[:40]
                st.download_button(
                    label="Export to .dta",
                    data=buf,
                    file_name=f"skill_trend_{safe_label}.dta",
                    mime="application/x-stata",
                )

    # ── Tab 5: NACE ──
    with tab5:
        st.markdown('<div class="sec-label">NACE sector assignment</div>', unsafe_allow_html=True)

        st.markdown(
            '<div class="nace-lead">'
            '<div class="nace-lead-title">From job title to economic sector</div>'
            '<div class="nace-lead-text">'
            'We assign each vacancy to a <b>NACE economic sector</b> by first matching the job title '
            'to an ESCO occupation and then linking that occupation to one or more candidate sectors. '
            'Where several sectors are possible, the final choice is made at the <b>individual job-ad level</b> '
            'using the responsibilities text.'
            '</div>'
            '<div class="nace-microcopy">The charts below show how broad those sector mappings are and how job ads are distributed across the NACE hierarchy.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="nace-card-footer-anchor"></div>', unsafe_allow_html=True)
        show_method = st.button(
            "How it works",
            help="Open methodology and worked examples",
            key="nace_info",
            type="secondary",
        )

        if show_method:
            _show_nace_methodology()

        conn_nd = sqlite3.connect(APP_DATA_DB)
        hist_rows = conn_nd.execute(
            "SELECT bin, count_all, count_ours, COALESCE(offers_ours, 0) FROM nace_histogram ORDER BY rowid"
        ).fetchall()
        conn_nd.close()

        bins        = [r[0] for r in hist_rows]
        count_all   = [r[1] for r in hist_rows]
        count_ours  = [r[2] for r in hist_rows]
        offers_ours = [r[3] for r in hist_rows]
        total_all   = sum(count_all)
        total_ours  = sum(count_ours)
        total_offers_ours = sum(offers_ours)

        TICK_S  = dict(size=13, color="#425466")
        AXIS_T  = dict(size=13, color="#425466")
        FONT_S  = dict(size=13, color="#425466")

        fig = pgo.Figure()

        pct_all  = [v / total_all  * 100 for v in count_all]
        pct_ours = [v / total_ours * 100 for v in count_ours]

        fig.add_trace(pgo.Bar(
            x=bins,
            y=pct_all,
            name=f"All ESCO occupations (n={total_all:,})",
            marker_color="#93C5FD",
            marker_line=dict(width=1, color="#fff"),
            text=[f"{v:.1f}%" for v in pct_all],
            textposition="outside",
            textfont=dict(size=11, color="#333"),
            customdata=list(zip(count_all, [f"{v:.1f}" for v in pct_all])),
            hovertemplate=(
                "<b>%{x} NACE code(s)</b><br>"
                "All ESCO: <b>%{customdata[0]:,}</b> occupations "
                "(<b>%{customdata[1]}%</b>)"
                "<extra></extra>"
            ),
        ))

        pct_offers_ours = [
            v / total_offers_ours * 100 if total_offers_ours else 0
            for v in offers_ours
        ]
        fig.add_trace(pgo.Bar(
            x=bins,
            y=pct_ours,
            name=f"Job titles in our dataset (n={total_ours:,})",
            marker_color="#1D4ED8",
            marker_line=dict(width=1, color="#fff"),
            text=[f"{v:.1f}%" for v in pct_ours],
            textposition="outside",
            textfont=dict(size=11, color="#1D4ED8"),
            customdata=list(zip(
                count_ours,
                [f"{v:.1f}" for v in pct_ours],
                offers_ours,
                [f"{v:.1f}" for v in pct_offers_ours],
            )),
            hovertemplate=(
                "<b>%{x} NACE code(s)</b><br>"
                "Our job ads: <b>%{customdata[0]:,}</b> occupations "
                "(<b>%{customdata[1]}%</b>)<br>"
                "Offers covered: <b>%{customdata[2]:,}</b> "
                "(<b>%{customdata[3]}%</b> of all offers)"
                "<extra></extra>"
            ),
        ))

        y_max = max(max(pct_all), max(pct_ours)) * 1.18

        fig.update_layout(
            barmode="group",
            height=480,
            margin=dict(l=60, r=20, t=30, b=60),
            plot_bgcolor="#fff",
            paper_bgcolor="#fff",
            font=FONT_S,
            xaxis=dict(
                title=dict(text="Number of NACE codes per ESCO occupation", font=AXIS_T),
                tickfont=TICK_S,
                showgrid=False,
            ),
            yaxis=dict(
                title=dict(text="% of ESCO occupations", font=AXIS_T),
                tickfont=TICK_S,
                showgrid=True,
                gridcolor="#e8e8e8",
                ticksuffix="%",
                range=[0, y_max],
            ),
            legend=dict(
                orientation="h",
                y=1.05,
                x=0,
                font=dict(size=13, color="#516274"),
            ),
            bargap=0.2,
            bargroupgap=0.05,
        )

        st.plotly_chart(fig, use_container_width=True)
        _dta_btn(
            pd.DataFrame({
                "nace_codes_per_occupation": bins,
                "pct_all_esco": [round(v, 2) for v in pct_all],
                "n_all_esco": count_all,
                "pct_our_job_ads": [round(v, 2) for v in pct_ours],
                "n_our_job_ads": count_ours,
                "n_offers_ours": offers_ours,
                "pct_offers_ours": [round(v, 2) for v in pct_offers_ours],
            }),
            "nace_histogram.dta", "dta_nace_hist",
        )

        # ── NACE Treemap ─────────────────────────────────────────
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="sec-label">Job offers by NACE sector</div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:0.88rem;color:#778596;margin-bottom:0.8rem">'
            'Click a box to drill down into the hierarchy. Labels show sector names; the NACE codes appear in the tooltip.'
            '</div>',
            unsafe_allow_html=True,
        )

        conn_tm = sqlite3.connect(APP_DATA_DB)
        tm_json = conn_tm.execute("SELECT data FROM nace_treemap").fetchone()[0]
        conn_tm.close()
        tm = json.loads(tm_json)


        # ── English NACE Rev 2 labels — section letters ───────────
        _NACE_EN_SEC = {
            'A': 'Agriculture, forestry and fishing',
            'B': 'Mining and quarrying',
            'C': 'Manufacturing',
            'D': 'Electricity, gas, steam and air conditioning supply',
            'E': 'Water supply; sewerage, waste management and remediation',
            'F': 'Construction',
            'G': 'Wholesale and retail trade',
            'H': 'Transportation and storage',
            'I': 'Accommodation and food service activities',
            'J': 'Information and communication',
            'K': 'Financial and insurance activities',
            'L': 'Real estate activities',
            'M': 'Professional, scientific and technical activities',
            'N': 'Administrative and support service activities',
            'O': 'Public administration and defence',
            'P': 'Education',
            'Q': 'Human health and social work activities',
            'R': 'Arts, entertainment and recreation',
            'S': 'Other service activities',
            'T': 'Activities of households as employers',
            'U': 'Activities of extraterritorial organisations',
            'V': 'Other',
        }

        # ── English NACE Rev 2 labels — keyed by 2-digit numeric code ─
        # Division numbers are globally unique in NACE Rev 2, so we can
        # strip the section letter and look up purely by number.
        _NACE_EN_DIV = {
            '01': 'Crop and animal production, hunting and related service activities',
            '02': 'Forestry and logging',
            '03': 'Fishing and aquaculture',
            '05': 'Mining of coal and lignite',
            '06': 'Extraction of crude petroleum and natural gas',
            '07': 'Mining of metal ores',
            '08': 'Other mining and quarrying',
            '09': 'Mining support service activities',
            '10': 'Manufacture of food products',
            '11': 'Manufacture of beverages',
            '12': 'Manufacture of tobacco products',
            '13': 'Manufacture of textiles',
            '14': 'Manufacture of wearing apparel',
            '15': 'Manufacture of leather and related products',
            '16': 'Manufacture of wood and of products of wood and cork',
            '17': 'Manufacture of paper and paper products',
            '18': 'Printing and reproduction of recorded media',
            '19': 'Manufacture of coke and refined petroleum products',
            '20': 'Manufacture of chemicals and chemical products',
            '21': 'Manufacture of basic pharmaceutical products and preparations',
            '22': 'Manufacture of rubber and plastic products',
            '23': 'Manufacture of other non-metallic mineral products',
            '24': 'Manufacture of basic metals',
            '25': 'Manufacture of fabricated metal products, except machinery',
            '26': 'Manufacture of computer, electronic and optical products',
            '27': 'Manufacture of electrical equipment',
            '28': 'Manufacture of machinery and equipment n.e.c.',
            '29': 'Manufacture of motor vehicles, trailers and semi-trailers',
            '30': 'Manufacture of other transport equipment',
            '31': 'Manufacture of furniture',
            '32': 'Other manufacturing',
            '33': 'Repair and installation of machinery and equipment',
            '35': 'Electricity, gas, steam and air conditioning supply',
            '36': 'Water collection, treatment and supply',
            '37': 'Sewerage',
            '38': 'Waste collection, treatment and disposal; materials recovery',
            '39': 'Remediation activities and other waste management services',
            '41': 'Construction of buildings',
            '42': 'Civil engineering',
            '43': 'Specialised construction activities',
            '45': 'Wholesale and retail trade and repair of motor vehicles',
            '46': 'Wholesale trade, except of motor vehicles and motorcycles',
            '47': 'Retail trade, except of motor vehicles and motorcycles',
            '49': 'Land transport and transport via pipelines',
            '50': 'Water transport',
            '51': 'Air transport',
            '52': 'Warehousing and support activities for transportation',
            '53': 'Postal and courier activities',
            '55': 'Accommodation',
            '56': 'Food and beverage service activities',
            '58': 'Publishing activities',
            '59': 'Motion picture, video and TV programme production',
            '60': 'Programming and broadcasting activities',
            '61': 'Telecommunications',
            '62': 'Computer programming, consultancy and related activities',
            '63': 'Information service activities',
            '64': 'Financial service activities, except insurance and pension funding',
            '65': 'Insurance, reinsurance and pension funding',
            '66': 'Activities auxiliary to financial services and insurance',
            '68': 'Real estate activities',
            '69': 'Legal and accounting activities',
            '70': 'Activities of head offices; management consultancy activities',
            '71': 'Architectural and engineering activities; technical testing and analysis',
            '72': 'Scientific research and development',
            '73': 'Advertising and market research',
            '74': 'Other professional, scientific and technical activities',
            '75': 'Veterinary activities',
            '77': 'Rental and leasing activities',
            '78': 'Employment activities',
            '79': 'Travel agency, tour operator and reservation service activities',
            '80': 'Security and investigation activities',
            '81': 'Services to buildings and landscape activities',
            '82': 'Office administrative, office support and business support activities',
            '84': 'Public administration and defence; compulsory social security',
            '85': 'Education',
            '86': 'Human health activities',
            '87': 'Residential care activities',
            '88': 'Social work activities without accommodation',
            '90': 'Creative, arts and entertainment activities',
            '91': 'Libraries, archives, museums and other cultural activities',
            '92': 'Gambling and betting activities',
            '93': 'Sports activities and amusement and recreation activities',
            '94': 'Activities of membership organisations',
            '95': 'Repair of computers and personal and household goods',
            '96': 'Other personal service activities',
            '97': 'Activities of households as employers of domestic personnel',
            '98': 'Undifferentiated goods- and services-producing activities of households for own use',
            '99': 'Activities of extraterritorial organisations and bodies',
        }

        def _nace_en_label(nid: str, pl_fallback: str) -> str:
            """Return English NACE label for any node id; Polish fallback."""
            if nid == 'root':
                return pl_fallback
            # Section letter only (depth-1)
            if len(nid) == 1:
                return _NACE_EN_SEC.get(nid, pl_fallback)
            # Division or group — strip leading letters to get numeric part
            numeric = nid.lstrip('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
            # Use first 2 digits for the division lookup
            div2 = numeric[:2] if len(numeric) >= 2 else numeric
            en = _NACE_EN_DIV.get(div2)
            if en:
                return en
            return pl_fallback

        fixed_labels = [
            _nace_en_label(nid, label)
            for nid, label in zip(tm['ids'], tm['labels'])
        ]

        # ── Color palette — distinct per section letter ───────────
        _SEC_COLORS = {
            'N': '#1B4F9C',  # deep blue    – Business admin
            'G': '#16803D',  # forest green – Trade
            'C': '#C2410C',  # burnt orange – Manufacturing
            'H': '#0E7490',  # dark cyan    – Transport
            'K': '#5B21B6',  # violet       – Finance
            'O': '#3730A3',  # indigo       – Public admin
            'J': '#0369A1',  # blue         – IT / comms
            'M': '#065F46',  # dark emerald – Professional svcs
            'F': '#92400E',  # brown        – Construction
            'Q': '#991B1B',  # dark red     – Health
            'P': '#0C4A6E',  # navy         – Education
            'L': '#831843',  # dark rose    – Real estate
            'I': '#9A3412',  # rust         – Accommodation
            'R': '#7E22CE',  # purple       – Culture & sport
            'S': '#3F6212',  # olive        – Other services
            'T': '#78350F',  # dark brown   – Repair / personal
            'D': '#BE185D',  # dark pink    – Energy
            'A': '#0F766E',  # teal         – Agriculture
            'E': '#155E75',  # dark cyan    – Utilities
            'B': '#374151',  # dark grey    – Mining
            'U': '#475569',  # slate        – Households
            'V': '#64748B',  # light slate  – Extraterritorial
        }

        def _lighten(hex_c: str, f: float) -> str:
            h = hex_c.lstrip('#')
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return '#{:02x}{:02x}{:02x}'.format(
                int(r + (255 - r) * f),
                int(g + (255 - g) * f),
                int(b + (255 - b) * f),
            )

        def _wrap_tm_label(label: str, width: int = 18, max_lines: int = 2) -> str:
            parts = label.split()
            if not parts:
                return label
            lines = []
            current = ""
            consumed = 0
            for part in parts:
                candidate = part if not current else f"{current} {part}"
                if len(candidate) <= width:
                    current = candidate
                    consumed += 1
                else:
                    if current:
                        lines.append(current)
                    current = part
                    consumed += 1
                    if len(lines) >= max_lines - 1:
                        break
            if current and len(lines) < max_lines:
                lines.append(current)
            if consumed < len(parts):
                last = lines[-1] if lines else label[:width]
                if len(last) > max(width - 3, 3):
                    last = last[:width - 3].rstrip()
                lines[-1] = last.rstrip(".") + "..."
            return "<br>".join(lines)

        _parent_map_tm = dict(zip(tm['ids'], tm['parents']))


        def _sec_letter(nid: str) -> str:
            curr = nid
            while curr and curr != 'root':
                p = _parent_map_tm.get(curr, 'root')
                if p in ('root', ''):
                    return curr[0] if curr else ''
                curr = p
            return ''

        def _depth_tm(nid: str) -> int:
            if nid == 'root':
                return 0
            p = _parent_map_tm.get(nid, 'root')
            if p in ('root', ''):
                return 1
            pp = _parent_map_tm.get(p, 'root')
            if pp in ('root', ''):
                return 2
            return 3

        node_colors = []
        display_text = []
        for nid, label, c in zip(tm['ids'], fixed_labels, tm['custom']):
            d = _depth_tm(nid)
            sec = _sec_letter(nid)
            base = _SEC_COLORS.get(sec, '#64748B')
            code = c.get('code', '') or nid
            if nid == 'root':
                node_colors.append('#334155')
                display_text.append('')
            elif d == 1:
                node_colors.append(base)
                display_text.append(f"<b>{_wrap_tm_label(label, width=18, max_lines=2)}</b>")
            elif d == 2:
                node_colors.append(_lighten(base, 0.28))
                display_text.append(code)
            else:
                node_colors.append(_lighten(base, 0.52))
                display_text.append(code)

        tm_custom = [
            f"{c['code']}<br>{c['cnt']:,} offers<br>{c['pct']}% of total"
            for c in tm['custom']
        ]

        fig_tm = pgo.Figure(pgo.Treemap(
            ids=tm['ids'],
            labels=fixed_labels,
            text=display_text,
            parents=tm['parents'],
            values=tm['values'],
            customdata=tm_custom,
            branchvalues='total',
            marker=dict(
                colors=node_colors,
                line=dict(width=1.5, color='rgba(255,255,255,0.6)'),
            ),
            hovertemplate='<b>%{label}</b><br>%{customdata}<extra></extra>',
            textinfo='text',
            texttemplate='%{text}',
            textfont=dict(size=12, color='#ffffff', family='Arial Black, Arial, sans-serif'),
            insidetextfont=dict(size=12, color='#ffffff'),
            maxdepth=3,
            pathbar=dict(visible=True, thickness=24),
        ))
        fig_tm.update_layout(
            height=720,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor='#f8f9fa',
            uniformtext=dict(minsize=7, mode='hide'),
            font=dict(size=12, color='#ffffff'),
        )
        st.plotly_chart(fig_tm, use_container_width=True)
        _dta_btn(
            pd.DataFrame({
                "nace_id":     tm["ids"],
                "sector_name": tm["labels"],
                "parent_id":   tm["parents"],
                "job_count":   tm["values"],
                "nace_code":   [c.get("code", "") for c in tm["custom"]],
                "pct_of_total":[c.get("pct",  0.0) for c in tm["custom"]],
            }),
            "nace_treemap.dta", "dta_nace_tm",
        )

        # ── Monthly NACE section share ────────────────────────────
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown(
            '<div class="sec-label">Monthly share of job offers by NACE section</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="font-size:0.88rem;color:#778596;margin-bottom:0.6rem">'
            'Shows the percentage of monthly job offers assigned to each top-level NACE section. '
            'Select up to 5 sectors to compare.'
            '</div>',
            unsafe_allow_html=True,
        )

        conn_mo = sqlite3.connect(APP_DATA_DB)
        mo_json = conn_mo.execute("SELECT data FROM nace_monthly").fetchone()[0]
        conn_mo.close()
        mo = json.loads(mo_json)

        _month_totals: dict[str, int] = collections.defaultdict(int)
        _sec_month_cnt: dict[str, dict[str, int]] = collections.defaultdict(lambda: collections.defaultdict(int))
        for m, s, c in zip(mo['months'], mo['sections'], mo['counts']):
            _month_totals[m] += c
            _sec_month_cnt[s][m] += c

        _all_months = sorted(_month_totals.keys())

        _sec_totals = {s: sum(cnts.values()) for s, cnts in _sec_month_cnt.items()}
        _ranked_secs = sorted(_sec_totals, key=lambda s: _sec_totals[s], reverse=True)

        _sec_options = [
            f"{s} — {_NACE_EN_SEC.get(s, s)}" for s in _ranked_secs
        ]
        _option_to_letter = {opt: opt[0] for opt in _sec_options}

        _default_top5 = _sec_options[:5]

        chosen = st.multiselect(
            "Sectors (max 5)",
            options=_sec_options,
            default=_default_top5,
            max_selections=5,
            key="nace_monthly_sel",
            label_visibility="collapsed",
        )

        chosen_letters = [_option_to_letter[o] for o in chosen]

        if chosen_letters:
            # pills showing current selection
            _pill_colors = {s: _SEC_COLORS.get(s, '#64748B') for s in chosen_letters}
            _pills_row = "".join(
                f'<span style="display:inline-block;background:{_pill_colors[s]};color:#fff;'
                f'font-size:0.78rem;font-weight:600;border-radius:999px;'
                f'padding:0.22rem 0.75rem;margin:0 0.3rem 0.4rem 0;">'
                f'{s} — {_NACE_EN_SEC.get(s, s)}</span>'
                for s in chosen_letters
            )
            st.markdown(
                f'<div style="margin-bottom:0.5rem;line-height:2;">{_pills_row}</div>',
                unsafe_allow_html=True,
            )

            fig_mo = pgo.Figure()
            _mo_export_rows = []

            for sec in chosen_letters:
                cnts = _sec_month_cnt[sec]
                pcts = [
                    round(cnts.get(m, 0) / _month_totals[m] * 100, 2) if _month_totals[m] else 0
                    for m in _all_months
                ]
                label = _NACE_EN_SEC.get(sec, sec)
                color = _SEC_COLORS.get(sec, '#64748B')

                fig_mo.add_trace(pgo.Scatter(
                    x=_all_months,
                    y=pcts,
                    name=f"{sec} — {label}",
                    mode='lines+markers',
                    line=dict(width=2.5, color=color),
                    marker=dict(size=7, color=color),
                    hovertemplate=(
                        f"<b>{sec} — {label}</b><br>"
                        "%{x}<br>"
                        "Share: <b>%{y:.2f}%</b>"
                        "<extra></extra>"
                    ),
                ))

                for m, p in zip(_all_months, pcts):
                    _mo_export_rows.append({
                        "month": m,
                        "nace_section": sec,
                        "section_name": label,
                        "pct_of_monthly_offers": p,
                        "count": cnts.get(m, 0),
                    })

            _month_labels = [
                m.split("-")[1] + "/" + m.split("-")[0][2:]
                for m in _all_months
            ]

            fig_mo.update_layout(
                height=440,
                margin=dict(l=55, r=20, t=20, b=55),
                plot_bgcolor="#fff",
                paper_bgcolor="#fff",
                font=dict(size=13, color="#425466"),
                xaxis=dict(
                    tickvals=_all_months,
                    ticktext=_month_labels,
                    tickfont=dict(size=12, color="#425466"),
                    showgrid=False,
                    tickangle=-45,
                ),
                yaxis=dict(
                    title=dict(text="% of monthly offers", font=dict(size=13, color="#425466")),
                    tickfont=dict(size=12, color="#425466"),
                    showgrid=True,
                    gridcolor="#eaeaea",
                    ticksuffix="%",
                    rangemode="tozero",
                ),
                legend=dict(
                    orientation="h",
                    y=-0.22,
                    x=0.5,
                    xanchor="center",
                    font=dict(size=11.5, color="#516274"),
                ),
                hovermode="x unified",
            )

            st.plotly_chart(fig_mo, use_container_width=True)

            _dta_btn(
                pd.DataFrame(_mo_export_rows),
                "nace_monthly_share.dta", "dta_nace_mo",
            )
        else:
            st.info("Select at least one NACE section above to display the chart.")

    # ── Tab 6: Ukrainian-Targeted Job Ads ──────────────────────────
    with tab6:
        _render_ua_tab()

    with tab7:
        _render_ai_tab()


# ── AI Skills tab helpers ──────────────────────────────────────────────────


AI_TAB_CACHE = os.path.join(DATA_DIR, "ai_tab_cache.json")

_AI_COLOR = "#E55B52"
_ALL_COLOR_AI = "#94A3B8"
_ICT_COLOR = "#3B82F6"

_CAT_COLORS = {
    "Core AI & Machine Learning": "#E55B52",
    "Natural Language Processing": "#10B981",
    "Computer Vision & Image Recognition": "#8B5CF6",
    "Data Science & Analytics": "#3B82F6",
    "AI Applications & Domain-Specific": "#F59E0B",
    "AI Applications & Predictive Modeling": "#EC4899",
    "AI Governance / Responsible AI": "#6366F1",
}

_AI_REGEX_PATTERNS = [
    r"\bartificial intelligence\b", r"\bsztuczna inteligencja\b",
    r"\bmachine learning\b", r"\buczenie maszynowe\b",
    r"\bdeep learning\b", r"\buczenie głębokie\b",
    r"\bneural network\b", r"\bsieć neuronow\w*\b",
    r"\bcomputer vision\b", r"\bnatural language processing\b",
    r"\bNLP\b", r"\bLLM\b", r"\blarge language model\b",
    r"\bGPT\b", r"\bChatGPT\b", r"\bgenerative ai\b", r"\bgen ai\b",
    r"\bMLOps\b", r"\bml engineer\b", r"\bai engineer\b",
    r"\bdata scien\w+\b", r"\btensorflow\b", r"\bpytorch\b",
    r"\bscikit-learn\b", r"\bkeras\b",
    r"\breinforcement learning\b", r"\btransfer learning\b",
    r"\bspeech recognition\b", r"\brozpoznawanie mowy\b",
    r"\bimage recognition\b", r"\bpredictive model\w*\b",
    r"\bmodel\w* predykcyjn\w*\b", r"\bbig data\b",
    r"\bdata mining\b", r"\bexploracja danych\b",
    r"\brecommender system\b", r"\brecommendation system\b",
    r"\bcognitive computing\b", r"\bautonomous vehicle\b",
    r"\bself-driving\b", r"\brobotic process automation\b",
]


@st.cache_data
def _load_ai_cache():
    with open(AI_TAB_CACHE, "r", encoding="utf-8") as f:
        return json.loads(f.read())


@st.cache_data
def _build_ai_keyword_sources_xlsx(ai_data: dict) -> bytes:
    esco_df = pd.DataFrame(ai_data.get("skills_freq", []))
    if not esco_df.empty:
        rename_map = {
            "name_en": "esco_skill_en",
            "name_pl": "esco_skill_pl",
            "category": "category",
            "scope": "scope",
            "n": "mentions_in_ai_offers",
            "pct_ai": "pct_ai_offers",
        }
        esco_df = esco_df.rename(columns=rename_map)
        wanted = [
            "esco_skill_en", "esco_skill_pl", "category",
            "scope", "mentions_in_ai_offers", "pct_ai_offers",
        ]
        esco_df = esco_df[[c for c in wanted if c in esco_df.columns]]
    regex_df = pd.DataFrame(
        {"regex_pattern": _AI_REGEX_PATTERNS, "source": "AI keyword matching"}
    )

    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        esco_df.to_excel(writer, sheet_name="ESCO_skills", index=False)
        regex_df.to_excel(writer, sheet_name="Regex_keywords", index=False)
    out.seek(0)
    return out.getvalue()


def _ai_grouped_bar(
    title,
    labels,
    pct_ai,
    pct_all,
    color_a=_AI_COLOR,
    color_b=_ALL_COLOR_AI,
    name_a="AI offers",
    name_b="All offers",
    height=None,
):
    fig = pgo.Figure()
    _max_val = max([0.0] + [float(v) for v in pct_ai] + [float(v) for v in pct_all])
    _x_max = (_max_val + max(8, _max_val * 0.22)) if _max_val > 0 else 1.0
    fig.add_trace(pgo.Bar(
        y=labels, x=pct_ai, name=name_a, orientation="h",
        marker=dict(color=color_a, cornerradius=4),
        text=[f"{v:.1f}%" for v in pct_ai],
        textposition="outside", textfont=dict(size=13, color=color_a),
        cliponaxis=False,
    ))
    fig.add_trace(pgo.Bar(
        y=labels, x=pct_all, name=name_b, orientation="h",
        marker=dict(color=color_b, cornerradius=4),
        text=[f"{v:.1f}%" for v in pct_all],
        textposition="outside", textfont=dict(size=13, color=color_b),
        cliponaxis=False,
    ))
    fig_h = height if height is not None else max(240, len(labels) * 30 + 70)
    fig.update_layout(
        barmode="group", height=fig_h,
        margin=dict(l=10, r=10, t=8, b=6),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(autorange="reversed", tickfont=dict(size=11, color="#425466")),
        xaxis=dict(visible=False, range=[0, _x_max]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                    font=dict(size=10, color="#425466")),
        bargap=0.3, bargroupgap=0.06,
    )
    return fig


@st.dialog("AI Offer Identification — Methodology", width="large")
def _show_ai_methodology():
    st.markdown("""
### How are AI offers identified?

A job offer is classified as **AI-related** if it meets **either** of two independent criteria:

---

#### Method 1: ESCO Skill Matching

Each job ad's requirements are matched to the ESCO skills taxonomy using **contextual embeddings**
(Voyage AI). If any of the matched ESCO skills belongs to a curated list of **35 AI-related skills**
(e.g. *machine learning*, *deep learning*, *natural language processing*, *computer vision*),
the offer is flagged.

The 35 skills are divided into two tiers:

| Tier | Description | Example skills |
|---|---|---|
| **Strict / Core AI** | Skills that are unambiguously about AI | machine learning, deep learning, neural networks, NLP, computer vision |
| **Extended AI** | Skills closely related to AI but also used outside it | data mining, data science, statistical modelling, robotics, predictive models |

#### Method 2: Keyword Detection

A regex-based search scans the **job title**, **requirements** and **responsibilities** fields
for ~40 AI-specific terms and phrases, including both English and Polish variants:

`machine learning`, `deep learning`, `artificial intelligence`, `sztuczna inteligencja`,
`NLP`, `LLM`, `GPT`, `ChatGPT`, `generative AI`, `computer vision`,
`TensorFlow`, `PyTorch`, `scikit-learn`, `MLOps`, `data scientist`, `big data`,
`reinforcement learning`, `predictive model`, etc.


---

#### Combined result

An offer is marked as **AI** if **either** method matches. This dual approach catches both
offers where AI skills were formally identified in the ESCO matching and those where AI terms
appear in free text but were not captured as a specific ESCO skill.

#### ICT offers

A separate **ICT** flag is defined as any offer that has a non-empty *technologies* field
(i.e., the employer explicitly listed technology requirements such as Python, AWS, Docker, etc.).
This serves as a proxy for ICT/tech job offers.
""")


@st.dialog("Overrepresentation Ratio — Methodology", width="large")
def _show_overrep_methodology():
    st.markdown("""
### How is the overrepresentation ratio calculated?

The ratio measures how much more likely a technology (or skill) is to appear in AI job offers
compared to non-AI job offers.

---

#### Formula

""")
    st.latex(r"\text{Ratio} = \frac{\text{Frequency in AI offers}}{\text{Frequency in non-AI offers}}")
    st.markdown("""

Where **frequency** = percentage of offers in that group that mention the item.

#### Example

If **Python** appears in 45% of AI offers but only 5% of non-AI offers,
the ratio is 45 / 5 = **9.0x** — meaning Python is 9 times more likely to be
listed in an AI job offer.

#### Sampling

To keep computation tractable, non-AI frequencies are estimated from a random sample
of 100,000 non-AI offers (out of ~733k). The sample is drawn with a fixed random seed
for reproducibility.

#### Minimum threshold

Only items appearing in at least **20 AI offers** are shown, to avoid noise from
very rare technologies.

#### Interpretation

- **Ratio > 3x**: strongly associated with AI offers
- **Ratio ~ 1x**: appears equally in AI and non-AI offers
- **Ratio < 1x**: more common in non-AI offers (rare in this view since we sort descending)
""")


@st.dialog("Co-occurring Skills — Methodology", width="large")
def _show_cooccur_methodology():
    st.markdown("""
### How are co-occurring skills identified?

This analysis finds **non-AI ESCO skills** that are disproportionately present in AI offers
compared to the rest of the job market. It reveals the complementary competencies that
employers seek alongside AI expertise.

---

#### Process

1. **Parse all ESCO skills** from the `skills_esco_contextual` field of each job ad
   (matched via contextual embeddings against ESCO v1.2.1)
2. **Exclude the 35 AI skills** themselves — we only want to see what *else* accompanies AI
3. **Calculate frequencies** in AI offers vs. a random sample of 100k non-AI offers
4. **Compute the overrepresentation ratio**: frequency-in-AI / frequency-in-non-AI
5. **Filter**: only skills appearing in ≥ 50 AI offers are shown

#### Transversal vs. domain-specific skills

The results are split into two groups:

- **Domain-specific skills** — technical or sector-specific competencies
  (e.g. *Python programming*, *cloud computing*, *database administration*)
- **Transversal skills** — cross-cutting soft skills from the ESCO transversal skills collection
  (e.g. *think analytically*, *work in teams*, *solve problems*)

#### What the ratio means

A ratio of **5.2x** means the skill appears 5.2 times more often in AI offers than in
the average non-AI offer. High-ratio skills represent the distinctive skill profile
of AI-related roles.
""")


def _render_ai_tab():
    data = _load_ai_cache()
    ov = data["overview"]

    # ── Header block: description on left, controls/stats on right ──
    _kpi_style = (
        'background:#fff;border:1px solid #e4e4ea;border-radius:12px;'
        'padding:0.8rem 0.7rem;text-align:center;white-space:nowrap;'
    )
    _kpi_val = 'font-size:1.55rem;font-weight:700;color:#1a1a2e;margin:0;line-height:1.15;'
    _kpi_lbl = 'font-size:0.72rem;color:#888;margin:0.2rem 0 0;'

    col_left, col_right = st.columns([2.25, 1.75])

    with col_left:
        st.markdown(
            '<div style="font-size:0.8rem;font-weight:700;letter-spacing:.06em;'
            'text-transform:uppercase;color:#8896a7;margin-bottom:0.3rem">'
            'AI Skills Demand in Polish Job Market — 2025</div>'
            '<div style="font-size:0.95rem;color:#4d5d6d;line-height:1.45">'
            'AI offers identified by ESCO skill matching and keyword detection '
            'in job titles, requirements and responsibilities.<br>'
            f'Based on <b>{ov["total_offers"]:,}</b> job ads collected in 2025.'
            '</div>',
            unsafe_allow_html=True,
        )

    with col_right:
        k1, k2 = st.columns(2)
        with k1:
            st.markdown(
                f'<div style="{_kpi_style}">'
                f'<p style="{_kpi_val}">{ov["pct_ai_all"]}%</p>'
                f'<p style="{_kpi_lbl}">of all 2025 offers</p></div>',
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                f'<div style="{_kpi_style}">'
                f'<p style="{_kpi_val}">{ov["pct_ai_ict"]}%</p>'
                f'<p style="{_kpi_lbl}">of ICT offers</p></div>',
                unsafe_allow_html=True,
            )
        st.markdown('<div style="height:0.45rem"></div>', unsafe_allow_html=True)
        btn_col, dl_col = st.columns(2)
        with btn_col:
            if st.button("Methodology", key="btn_ai_method", type="secondary",
                         use_container_width=True):
                _show_ai_methodology()
        with dl_col:
            st.download_button(
                "AI keywords",
                data=_build_ai_keyword_sources_xlsx(data),
                file_name="ai_keyword_sources.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_ai_keyword_sources",
                type="secondary",
                use_container_width=True,
            )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)


    # ── 1. AI ESCO Skills frequency ──
    st.markdown(
        '<div style="font-size:1.02rem;font-weight:600;color:#1a1a2e;margin:0.5rem 0 0.45rem 0;">'
        'Most Demanded AI Skills (ESCO)</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-size:0.85rem;color:#778596;margin-bottom:0.8rem">'
        'How often each AI-related skill appears in AI job offers. '
        'Dark red = Strict AI (core ML, deep learning), light red = Extended AI (data science, analytics).'
        '</div>',
        unsafe_allow_html=True,
    )

    sf = data["skills_freq"]
    sf_top = sf[:15]
    _sf_max = max([float(s["pct_ai"]) for s in sf_top]) if sf_top else 0
    sf_x_max = (_sf_max + max(6, _sf_max * 0.22)) if _sf_max > 0 else 1.0

    _SCOPE_COLORS = {"Strict AI": _AI_COLOR, "Extended AI": "#F3A7A2"}
    scope_order = ["Strict AI", "Extended AI"]
    fig_skills = pgo.Figure()
    for scope in scope_order:
        items = [s for s in sf_top if s["scope"] == scope]
        if not items:
            continue
        fig_skills.add_trace(pgo.Bar(
            y=[s["name_en"] for s in items],
            x=[s["pct_ai"] for s in items],
            name=scope,
            orientation="h",
            marker=dict(color=_SCOPE_COLORS[scope], cornerradius=4),
            text=[f"{s['pct_ai']:.1f}%" for s in items],
            textposition="outside", textfont=dict(size=13),
            hovertemplate="<b>%{y}</b><br>%{x:.2f}% of AI offers<extra></extra>",
        ))
    fig_skills.update_traces(cliponaxis=False)
    fig_skills.update_layout(
        height=max(360, len(sf_top) * 22 + 100),
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(autorange="reversed", tickfont=dict(size=10, color="#425466"), categoryorder="array",
                   categoryarray=[s["name_en"] for s in sf_top]),
        xaxis=dict(visible=False, range=[0, sf_x_max]),
        barmode="overlay",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="left", x=0,
            font=dict(size=11, color="#425466"),
            tracegroupgap=0,
        ),
        bargap=0.25,
    )
    st.plotly_chart(fig_skills, use_container_width=True, config={"displayModeBar": False})

    sf_df = pd.DataFrame(sf)
    sf_df.columns = ["Skill (EN)", "Skill (PL)", "Category", "Scope",
                     "N offers", "% AI offers", "% all offers", "% ICT offers"]
    _export_chart_and_dta(
        fig_skills, sf_df,
        "ai_skills_esco_freq.dta", "dta_ai_skills",
        "ai_skills_esco_freq.png", "png_ai_skills",
    )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── 2. Keywords frequency ──
    st.markdown(
        '<div style="font-size:1.02rem;font-weight:600;color:#1a1a2e;margin:0.5rem 0 0.45rem 0;">'
        'Top AI Keywords Found in Offers</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-size:0.85rem;color:#778596;margin-bottom:0.8rem">'
        'How often each AI-related keyword appears across job titles, requirements and responsibilities in AI offers.'
        '</div>',
        unsafe_allow_html=True,
    )

    kf = data["kw_freq"][:15]
    _kw_max = max([float(k["pct_ai"]) for k in kf]) if kf else 0
    kw_x_max = (_kw_max + max(6, _kw_max * 0.22)) if _kw_max > 0 else 1.0
    fig_kw = pgo.Figure(pgo.Bar(
        y=[k["keyword"] for k in kf],
        x=[k["pct_ai"] for k in kf],
        orientation="h",
        marker=dict(color=_AI_COLOR, cornerradius=4),
        text=[f'{k["pct_ai"]:.1f}%' for k in kf],
        textposition="outside", textfont=dict(size=13, color="#555"),
        hovertemplate="<b>%{y}</b><br>%{x:.2f}% of AI offers<br>n = %{customdata:,}<extra></extra>",
        customdata=[k["n"] for k in kf],
        cliponaxis=False,
    ))
    fig_kw.update_layout(
        height=max(300, len(kf) * 22 + 60),
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(autorange="reversed", tickfont=dict(size=11, color="#425466")),
        xaxis=dict(visible=False, range=[0, kw_x_max]),
    )
    st.plotly_chart(fig_kw, use_container_width=True, config={"displayModeBar": False})

    _export_chart_and_dta(
        fig_kw, pd.DataFrame(kf),
        "ai_keywords_freq.dta", "dta_ai_kw",
        "ai_keywords_freq.png", "png_ai_kw",
    )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── 3a. Seniority ──
    st.markdown(
        '<div style="font-size:1.02rem;font-weight:600;color:#1a1a2e;margin:0.5rem 0 0.45rem 0;">'
        'Seniority Level in AI Offers</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-size:0.85rem;color:#778596;margin-bottom:0.8rem">'
        'Distribution of seniority levels in AI offers compared to all offers.'
        '</div>',
        unsafe_allow_html=True,
    )
    sen = data["seniority"]
    fig_sen = _ai_grouped_bar(
        "Seniority",
        [s["level"] for s in sen],
        [s["pct_ai"] for s in sen],
        [s["pct_all"] for s in sen],
        height=250,
    )
    st.plotly_chart(fig_sen, use_container_width=True, config={"displayModeBar": False})
    _export_chart_and_dta(
        fig_sen, pd.DataFrame(sen),
        "ai_seniority.dta", "dta_ai_sen",
        "ai_seniority.png", "png_ai_sen",
    )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── 3b. Contract Type ──
    st.markdown(
        '<div style="font-size:1.02rem;font-weight:600;color:#1a1a2e;margin:0.5rem 0 0.45rem 0;">'
        'Contract Type in AI Offers</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-size:0.85rem;color:#778596;margin-bottom:0.8rem">'
        'Distribution of contract types in AI offers compared to all offers.'
        '</div>',
        unsafe_allow_html=True,
    )
    ct = data["contracts"]
    fig_ct = _ai_grouped_bar(
        "Contract",
        [c["type"] for c in ct],
        [c["pct_ai"] for c in ct],
        [c["pct_all"] for c in ct],
        height=250,
    )
    st.plotly_chart(fig_ct, use_container_width=True, config={"displayModeBar": False})
    _export_chart_and_dta(
        fig_ct, pd.DataFrame(ct),
        "ai_contracts.dta", "dta_ai_ct",
        "ai_contracts.png", "png_ai_ct",
    )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── 4. Location ──
    st.markdown(
        '<div style="font-size:1.02rem;font-weight:600;color:#1a1a2e;margin:0.5rem 0 0.45rem 0;">'
        'Location of AI Job Offers</div>',
        unsafe_allow_html=True,
    )


    loc_all = data["locations"]
    loc_filtered = [l for l in loc_all if str(l.get("city", "")).strip().lower() != "other"]
    loc = loc_filtered[:10]
    loc_h = max(300, len(loc) * 30 + 75)
    fig_loc = _ai_grouped_bar(
        "Location",
        [l["city"] for l in loc],
        [l["pct_ai"] for l in loc],
        [l["pct_all"] for l in loc],
        height=loc_h,
    )
    st.plotly_chart(fig_loc, use_container_width=True, config={"displayModeBar": False})
    _export_chart_and_dta(
        fig_loc, pd.DataFrame(loc_filtered),
        "ai_locations.dta", "dta_ai_loc",
        "ai_locations.png", "png_ai_loc",
    )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── 5. Transversal skills ──
    co = data.get("cooccur", [])
    co_trans = [
        c for c in co
        if c.get("transversal")
        and "apply basic programming" not in str(c.get("name_en", "")).lower()
    ]
    if co_trans:
        co_trans = sorted(co_trans, key=lambda x: float(x.get("pct_ai", 0)) - float(x.get("pct_non_ai", 0)), reverse=True)[:10]
        st.markdown(
            '<div style="font-size:1.02rem;font-weight:600;color:#1a1a2e;margin:0.5rem 0 0.45rem 0;">'
            'Transversal Skills in AI Offers</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="font-size:0.85rem;color:#778596;margin-bottom:0.6rem">'
            'Comparison of soft skills in AI offers vs non-AI offers. '
            'Sorted by percentage point difference.'
            '</div>',
            unsafe_allow_html=True,
        )

        fig_trans = _ai_grouped_bar(
            "Transversal",
            [c["name_en"] or c["name_pl"] for c in co_trans],
            [c["pct_ai"] for c in co_trans],
            [c["pct_non_ai"] for c in co_trans],
            name_a="% in AI offers",
            name_b="% in non-AI offers",
            height=max(280, len(co_trans) * 30 + 70),
        )
        st.plotly_chart(fig_trans, use_container_width=True, config={"displayModeBar": False})
        _export_chart_and_dta(
            fig_trans, pd.DataFrame(co_trans),
            "ai_transversal_skills.dta", "dta_ai_trans",
            "ai_transversal_skills.png", "png_ai_trans",
        )

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── 7. ESCO Job Titles ──
    st.markdown(
        '<div style="font-size:1.02rem;font-weight:600;color:#1a1a2e;margin:0.5rem 0 0.45rem 0;">'
        'Top ESCO Occupations in AI Offers</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-size:0.85rem;color:#778596;margin-bottom:0.8rem">'
        'Which standardised occupations are most common among AI-related job offers.'
        '</div>',
        unsafe_allow_html=True,
    )

    esco_titles_all = data["esco_titles"]
    esco_titles_filtered = [
        t for t in esco_titles_all
        if "youth programme director" not in str(t.get("title_en", "")).strip().lower()
    ]
    et = esco_titles_filtered[:15]
    _et_max = max([float(t["pct_ai"]) for t in et]) if et else 0
    et_x_max = (_et_max + max(6, _et_max * 0.22)) if _et_max > 0 else 1.0
    fig_et = pgo.Figure(pgo.Bar(
        y=[t["title_en"] or t["title_pl"] for t in et],
        x=[t["pct_ai"] for t in et],
        orientation="h",
        marker=dict(color=_AI_COLOR, cornerradius=4),
        text=[f'{t["pct_ai"]:.1f}%' for t in et],
        textposition="outside", textfont=dict(size=13, color=_AI_COLOR),
        hovertemplate="<b>%{y}</b><br>%{x:.2f}% of AI offers<br>n = %{customdata:,}<extra></extra>",
        customdata=[t["n"] for t in et],
        cliponaxis=False,
    ))
    fig_et.update_layout(
        height=max(320, len(et) * 22 + 60),
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(autorange="reversed", tickfont=dict(size=10, color="#425466")),
        xaxis=dict(visible=False, range=[0, et_x_max]),
    )
    st.plotly_chart(fig_et, use_container_width=True, config={"displayModeBar": False})

    _export_chart_and_dta(
        fig_et, pd.DataFrame(esco_titles_filtered),
        "ai_esco_titles.dta", "dta_ai_titles",
        "ai_esco_titles.png", "png_ai_titles",
    )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── 8. NACE Sections ──
    st.markdown(
        '<div style="font-size:1.02rem;font-weight:600;color:#1a1a2e;margin:0.5rem 0 0.45rem 0;">'
        'NACE Sectors of AI Job Offers</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-size:0.85rem;color:#778596;margin-bottom:0.8rem">'
        'Which economic sectors post the most AI-related job offers.'
        '</div>',
        unsafe_allow_html=True,
    )

    ns = data["nace_sections"]
    _ns_max = max([float(s["pct_ai"]) for s in ns]) if ns else 0
    ns_x_max = (_ns_max + max(6, _ns_max * 0.22)) if _ns_max > 0 else 1.0
    nace_name_overrides = {
        "T": "Activities of households as employers",
    }
    ns_labels = [
        f'{s["section"]} — {nace_name_overrides.get(s["section"], s["name"])}'
        for s in ns
    ]
    fig_ns = pgo.Figure(pgo.Bar(
        y=ns_labels,
        x=[s["pct_ai"] for s in ns],
        orientation="h",
        marker=dict(color=_AI_COLOR, cornerradius=4),
        text=[f'{s["pct_ai"]:.1f}%' for s in ns],
        textposition="outside", textfont=dict(size=13, color=_AI_COLOR),
        hovertemplate="<b>%{y}</b><br>%{x:.2f}% of AI offers<br>n = %{customdata:,}<extra></extra>",
        customdata=[s["n"] for s in ns],
        cliponaxis=False,
    ))
    fig_ns.update_layout(
        height=max(280, len(ns) * 22 + 60),
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(autorange="reversed", tickfont=dict(size=10, color="#425466")),
        xaxis=dict(visible=False, range=[0, ns_x_max]),
    )
    st.plotly_chart(fig_ns, use_container_width=True, config={"displayModeBar": False})

    _export_chart_and_dta(
        fig_ns, pd.DataFrame(ns),
        "ai_nace_sections.dta", "dta_ai_nace",
        "ai_nace_sections.png", "png_ai_nace",
    )


# ── Ukrainian-Targeted Job Ads helpers ─────────────────────────────────────


def _dta_btn(df: pd.DataFrame, filename: str, key: str) -> None:
    """Render a small .dta download button for the given DataFrame."""
    buf = io.BytesIO()
    df.to_stata(buf, write_index=False, version=118)
    buf.seek(0)
    st.download_button(
        label="Export .dta",
        data=buf,
        file_name=filename,
        mime="application/x-stata",
        key=key,
        type="secondary",
        use_container_width=True,
    )


def _export_chart_and_dta(
    fig: pgo.Figure,
    df: pd.DataFrame,
    dta_filename: str,
    dta_key: str,
    png_filename: str,
    png_key: str,
) -> None:
    """Render side-by-side exports: Stata data + hi-res PNG chart."""
    buf = io.BytesIO()
    df.to_stata(buf, write_index=False, version=118)
    buf.seek(0)

    png_bytes = None
    try:
        import copy
        fig_export = copy.deepcopy(fig)
        fig_h = int(fig.layout.height) if fig.layout.height else 500
        cur_margin = fig.layout.margin or {}
        export_font = 22
        export_tick = 20
        fig_export.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color="#111111", size=export_font),
            margin=dict(
                l=max((cur_margin.l or 0), 180),
                r=max((cur_margin.r or 0), 160),
                t=max((cur_margin.t or 0), 80),
                b=max((cur_margin.b or 0), 60),
            ),
        )
        fig_export.update_xaxes(tickfont=dict(size=export_tick, color="#111111"))
        fig_export.update_yaxes(tickfont=dict(size=export_tick, color="#111111"))
        fig_export.update_traces(textfont=dict(size=export_font), selector=dict(type="bar"))
        png_bytes = fig_export.to_image(
            format="png",
            width=2000,
            height=max(900, fig_h * 3),
            scale=2,
        )
    except Exception:
        png_bytes = None

    c1, c2, _ = st.columns([1.3, 1.45, 5.25])
    with c1:
        st.download_button(
            label="Export .dta",
            data=buf,
            file_name=dta_filename,
            mime="application/x-stata",
            key=dta_key,
            type="secondary",
        )
    with c2:
        if png_bytes:
            st.download_button(
                label="Export chart",
                data=png_bytes,
                file_name=png_filename,
                mime="image/png",
                key=png_key,
                type="secondary",
            )
        else:
            st.button("Export chart", key=f"{png_key}_disabled", disabled=True, type="secondary")


_UA_COLOR  = "#FBBF24"   # amber  – Ukrainian-targeted ads
_ALL_COLOR = "#3B82F6"   # blue   – all jobs

_UA_DATA = {
    "Position Level": {
        "labels": [
            "Intern / Trainee",
            "Assistant",
            "Junior Specialist",
            "Specialist (Mid)",
            "Senior Specialist",
            "Expert",
            "Manager / Coordinator",
            "Senior Manager",
            "Director",
            "Manual Worker",
        ],
        "all_pct":  [0.8,  3.0,  9.3, 41.5, 11.6, 2.6, 8.2, 4.3, 1.1, 17.5],
        "ua_pct":   [0.9,  2.2,  7.7, 23.9,  9.4, 1.7, 7.5, 1.7, 0.2, 44.9],
    },
    "Contract Type": {
        "labels": [
            "Employment Contract",
            "Work-for-Hire",
            "Commission Contract",
            "B2B Contract",
            "Substitute Contract",
            "Agency Contract",
            "Temporary Employment",
            "Internship / Apprenticeship",
        ],
        "all_pct": [61.2, 0.8, 13.3, 21.9, 0.7, 1.0, 0.6, 0.5],
        "ua_pct":  [51.9, 0.8, 24.3, 19.1, 0.3, 0.7, 1.8, 1.1],
    },
    "Work Dimension": {
        "labels": ["Part-time", "Additional / Temporary", "Full-time"],
        "all_pct": [8.5,  2.8, 88.7],
        "ua_pct":  [15.0, 5.8, 79.2],
    },
    "Work Mode": {
        "labels": ["On-site", "Hybrid", "Remote", "Mobile"],
        "all_pct": [56.1, 23.5,  8.0, 12.4],
        "ua_pct":  [62.0, 18.5,  6.7, 12.8],
    },
    "Salary Transparency": {
        "labels": ["Job ads showing salary range"],
        # all: 21 124 / 73 507 tryb-pracy total ≈ 28.7 %
        # ua:  2 653 / 5 360 tryb-pracy total  ≈ 49.5 %
        "all_pct": [28.7],
        "ua_pct":  [49.5],
    },
}

_UA_TICK  = dict(size=12, color="#425466")
_UA_AXIS  = dict(size=12, color="#425466")
_UA_FONT  = dict(size=12, color="#425466")

_UA_LAYOUT_BASE = dict(
    plot_bgcolor="#fff",
    paper_bgcolor="#fff",
    font=_UA_FONT,
    margin=dict(l=10, r=20, t=58, b=10),
    legend=dict(
        orientation="h",
        y=1.03,
        yanchor="bottom",
        x=0,
        xanchor="left",
        font=dict(size=12, color="#516274"),
        bgcolor="rgba(0,0,0,0)",
    ),
    bargroupgap=0.08,
)


def _ua_grouped_bar_h(title: str, labels, all_pct, ua_pct):
    """Horizontal grouped bar – good for many categories."""
    # Sort ascending by avg of both series; autorange="reversed" puts highest at top
    order = sorted(range(len(labels)), key=lambda i: (all_pct[i] + ua_pct[i]) / 2)
    labels  = [labels[i]  for i in order]
    all_pct = [all_pct[i] for i in order]
    ua_pct  = [ua_pct[i]  for i in order]

    fig = pgo.Figure()
    fig.add_trace(pgo.Bar(
        y=labels,
        x=all_pct,
        name="All job ads",
        orientation="h",
        marker_color=_ALL_COLOR,
        marker_line=dict(width=0),
        text=[f"{v:.1f}%" for v in all_pct],
        textposition="outside",
        textfont=dict(size=10, color=_ALL_COLOR),
        hovertemplate="<b>%{y}</b><br>All jobs: <b>%{x:.1f}%</b><extra></extra>",
    ))
    fig.add_trace(pgo.Bar(
        y=labels,
        x=ua_pct,
        name="Ukrainian-targeted ads",
        orientation="h",
        marker_color=_UA_COLOR,
        marker_line=dict(width=0),
        text=[f"{v:.1f}%" for v in ua_pct],
        textposition="outside",
        textfont=dict(size=10, color="#B45309"),
        hovertemplate="<b>%{y}</b><br>Ukrainian-targeted: <b>%{x:.1f}%</b><extra></extra>",
    ))
    x_max = max(max(all_pct), max(ua_pct)) * 1.22
    n = len(labels)
    height = max(340, n * 52)
    fig.update_layout(
        **_UA_LAYOUT_BASE,
        barmode="group",
        height=height,
        xaxis=dict(
            ticksuffix="%",
            tickfont=_UA_TICK,
            showgrid=True,
            gridcolor="#eeeeee",
            range=[0, x_max],
        ),
        yaxis=dict(
            tickfont=dict(size=11, color="#425466"),
            autorange="reversed",
        ),
    )
    return fig


def _ua_grouped_bar_v(title: str, labels, all_pct, ua_pct):
    """Vertical grouped bar – good for few categories."""
    # Sort descending by avg of both series; highest bar appears first (left)
    order = sorted(range(len(labels)), key=lambda i: (all_pct[i] + ua_pct[i]) / 2, reverse=True)
    labels  = [labels[i]  for i in order]
    all_pct = [all_pct[i] for i in order]
    ua_pct  = [ua_pct[i]  for i in order]

    fig = pgo.Figure()
    fig.add_trace(pgo.Bar(
        x=labels,
        y=all_pct,
        name="All job ads",
        marker_color=_ALL_COLOR,
        marker_line=dict(width=0),
        text=[f"{v:.1f}%" for v in all_pct],
        textposition="outside",
        textfont=dict(size=11, color=_ALL_COLOR),
        hovertemplate="<b>%{x}</b><br>All jobs: <b>%{y:.1f}%</b><extra></extra>",
    ))
    fig.add_trace(pgo.Bar(
        x=labels,
        y=ua_pct,
        name="Ukrainian-targeted ads",
        marker_color=_UA_COLOR,
        marker_line=dict(width=0),
        text=[f"{v:.1f}%" for v in ua_pct],
        textposition="outside",
        textfont=dict(size=11, color="#B45309"),
        hovertemplate="<b>%{x}</b><br>Ukrainian-targeted: <b>%{y:.1f}%</b><extra></extra>",
    ))
    y_max = max(max(all_pct), max(ua_pct)) * 1.22
    fig.update_layout(
        **_UA_LAYOUT_BASE,
        barmode="group",
        height=360,
        xaxis=dict(tickfont=dict(size=11, color="#425466"), showgrid=False),
        yaxis=dict(
            ticksuffix="%",
            tickfont=_UA_TICK,
            showgrid=True,
            gridcolor="#eeeeee",
            range=[0, y_max],
        ),
    )
    return fig


def _render_ua_tab():
    st.markdown(
        '<div class="sec-label">Ukrainian-Targeted vs All Job Ads — share within each category (%)</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-size:0.88rem;color:#778596;margin-bottom:1.4rem">'
        'Comparing the distribution of <b>All job ads</b> vs '
        '<b>Ukrainian-targeted ads</b> (<b>Робота у Польщі</b>). '
        'All values are percentages within each filter category.'
        '</div>',
        unsafe_allow_html=True,
    )

    # Position Level — horizontal (11 categories)
    d = _UA_DATA["Position Level"]
    st.markdown(
        '<div style="font-size:1.02rem;font-weight:600;color:#1a1a2e;margin:0.15rem 0 0.45rem 0;">Position Level</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        _ua_grouped_bar_h("Position Level", d["labels"], d["all_pct"], d["ua_pct"]),
        use_container_width=True,
    )
    _dta_btn(
        pd.DataFrame({"option": d["labels"], "pct_all_jobs": d["all_pct"], "pct_ua_targeted": d["ua_pct"]}),
        "ua_position_level.dta", "dta_pos",
    )

    st.markdown('<hr style="border:none;border-top:1px solid #eee;margin:0.5rem 0 1rem">', unsafe_allow_html=True)

    # Contract Type — horizontal (8 categories)
    d = _UA_DATA["Contract Type"]
    st.markdown(
        '<div style="font-size:1.02rem;font-weight:600;color:#1a1a2e;margin:0.15rem 0 0.45rem 0;">Contract Type</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        _ua_grouped_bar_h("Contract Type", d["labels"], d["all_pct"], d["ua_pct"]),
        use_container_width=True,
    )
    _dta_btn(
        pd.DataFrame({"option": d["labels"], "pct_all_jobs": d["all_pct"], "pct_ua_targeted": d["ua_pct"]}),
        "ua_contract_type.dta", "dta_contract",
    )

    st.markdown('<hr style="border:none;border-top:1px solid #eee;margin:0.5rem 0 1rem">', unsafe_allow_html=True)

    # Work Dimension + Work Mode — side by side
    col1, col2 = st.columns(2)
    with col1:
        d = _UA_DATA["Work Dimension"]
        st.markdown(
            '<div style="font-size:1.02rem;font-weight:600;color:#1a1a2e;margin:0.15rem 0 0.45rem 0;">Work Dimension</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            _ua_grouped_bar_v("Work Dimension", d["labels"], d["all_pct"], d["ua_pct"]),
            use_container_width=True,
        )
        _dta_btn(
            pd.DataFrame({"option": d["labels"], "pct_all_jobs": d["all_pct"], "pct_ua_targeted": d["ua_pct"]}),
            "ua_work_dimension.dta", "dta_dim",
        )
    with col2:
        d = _UA_DATA["Work Mode"]
        st.markdown(
            '<div style="font-size:1.02rem;font-weight:600;color:#1a1a2e;margin:0.15rem 0 0.45rem 0;">Work Mode</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            _ua_grouped_bar_v("Work Mode", d["labels"], d["all_pct"], d["ua_pct"]),
            use_container_width=True,
        )
        _dta_btn(
            pd.DataFrame({"option": d["labels"], "pct_all_jobs": d["all_pct"], "pct_ua_targeted": d["ua_pct"]}),
            "ua_work_mode.dta", "dta_mode",
        )

    st.markdown('<hr style="border:none;border-top:1px solid #eee;margin:0.5rem 0 1rem">', unsafe_allow_html=True)

    # Salary Transparency
    d = _UA_DATA["Salary Transparency"]
    st.markdown(
        '<div style="font-size:1.02rem;font-weight:600;color:#1a1a2e;margin:0.15rem 0 0.45rem 0;">Salary Transparency</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        _ua_grouped_bar_v("Salary Transparency", d["labels"], d["all_pct"], d["ua_pct"]),
        use_container_width=True,
    )
    _dta_btn(
        pd.DataFrame({"option": d["labels"], "pct_all_jobs": d["all_pct"], "pct_ua_targeted": d["ua_pct"]}),
        "ua_salary_transparency.dta", "dta_salary",
    )


if __name__ == "__main__":
    main()
