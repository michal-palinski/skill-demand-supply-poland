#!/usr/bin/env python3
"""
Skill Demand and Supply in Poland 2025 — Streamlit Cloud deployment version.

Differences from app_search.py:
  - No FAISS, no VoyageAI, no API keys required.
  - Job-title search uses SQLite FTS5 (built by prepare_deploy.py).
  - Data is read from the deploy/ subfolder (~144 MB total).
"""

import streamlit as st
import sqlite3
import numpy as np
import json
import os
import re
import warnings
import io
import collections
import pandas as pd
import plotly.graph_objects as pgo
from statsmodels.tsa.arima.model import ARIMA

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

/* Remove whitespace around Datawrapper iframes */
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


# ── Data paths ─────────────────────────────────────────────────

DATA_DIR     = os.path.join(os.path.dirname(__file__), "deploy")
APP_DATA_DB  = os.path.join(DATA_DIR, "app_data.db")
REQ_RESP_DB  = os.path.join(DATA_DIR, "req_resp_slim.db")
TRENDS_DB    = os.path.join(DATA_DIR, "skill_trends.db")


# ── FTS5 job-title search (replaces FAISS + VoyageAI) ─────────

def _fts_query(text: str) -> str:
    """Build a safe FTS5 MATCH expression from user input."""
    words = re.split(r"\s+", text.strip())
    terms = []
    for w in words:
        safe = w.replace('"', '""')
        if safe:
            terms.append(f'"{safe}"*')
    return " OR ".join(terms) if terms else '""'


@st.cache_data(ttl=300, show_spinner=False)
def search_jobs_fts(query: str, top_k: int = 30):
    """Full-text search over job titles using SQLite FTS5 (BM25 ranking)."""
    q = query.strip()
    if not q:
        return [], None

    fts_expr = _fts_query(q)
    conn = sqlite3.connect(APP_DATA_DB)
    try:
        rows = conn.execute(
            "SELECT title, rank FROM job_titles_fts WHERE job_titles_fts MATCH ? ORDER BY rank LIMIT ?",
            (fts_expr, top_k),
        ).fetchall()
    except Exception as exc:
        conn.close()
        return None, str(exc)
    conn.close()

    if not rows:
        return [], None

    # BM25 rank: more negative = better match. Normalise to 0.5–0.95 range.
    ranks = [r[1] for r in rows]
    best  = min(ranks)
    span  = (max(ranks) - best) or 1.0
    results = []
    for title, rank in rows:
        norm = 1.0 - (rank - best) / span   # 1.0 best → 0.0 worst in this set
        similarity = 0.5 + 0.45 * norm      # maps to 0.95 … 0.50
        results.append({"title": title, "similarity": similarity})
    return results, None


@st.cache_data(ttl=3600)
def get_kzis_matches(job_title):
    conn = sqlite3.connect(APP_DATA_DB)
    c = conn.cursor()
    c.execute(
        "SELECT kzis_occupation_name, similarity_score, rank "
        "FROM job_kzis_matches WHERE job_title = ? ORDER BY rank",
        (job_title,),
    )
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
    c.execute(
        """
        SELECT item_type, item_text, skill_label, skill_type, similarity
        FROM skill_matches WHERE job_id = ? AND rank = 1
        ORDER BY item_type, similarity DESC
        """,
        (job_id,),
    )
    top1 = c.fetchall()
    c.execute(
        """
        SELECT skill_label, skill_type, MAX(similarity), COUNT(*)
        FROM skill_matches WHERE job_id = ? AND rank = 1
        GROUP BY skill_label ORDER BY MAX(similarity) DESC
        """,
        (job_id,),
    )
    unique_skills = c.fetchall()
    conn.close()
    return top1, unique_skills


@st.cache_data
def get_sample_titles_for_filter():
    offers = get_sample_offers()
    titles = sorted(set(t for _, t in offers))
    return titles, [t.lower() for t in titles]


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
    tree = cache["tree"]
    ids, labels, parents, values, custom = [], [], [], [], []

    def add(id_: str, label: str, parent: str, value: int, info: str = ""):
        ids.append(id_)
        labels.append(label)
        parents.append(parent)
        values.append(value)
        custom.append(info)

    def walk(node_children: dict, parent_id: str, depth: int):
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
                add(node_id, f"{code}  {node['title']}", parent_id, cnt, f"{code} · {cnt:,}")
                walk(children, node_id, depth + 1)
            else:
                add(node_id, node["title"], parent_id, cnt, f"{cnt:,}")

    is_target = _is_skills_code if view == "skills" else _is_knowledge_code
    matched_codes = [c for c in tree if is_target(c)]
    lang_codes    = [c for c in matched_codes if c.startswith("L")] if view == "skills" else []
    regular_codes = [c for c in matched_codes if c not in lang_codes]

    def sort_key(code):
        if code.startswith("S"):
            return (0, code)
        elif code.startswith("T"):
            return (1, code)
        return (2, code)

    other_codes = [c for c in tree if not _is_skills_code(c) and not _is_knowledge_code(c)]
    other_total  = sum(_node_count(tree[c]) for c in other_codes)
    other_total += cache["meta"].get("unmatched_mentions", 0)

    ROOT = "root"
    total_view = sum(_node_count(tree[c]) for c in matched_codes) + other_total
    root_label  = "Skills & Competences" if view == "skills" else "Knowledge"
    add(ROOT, root_label, "", total_view, f"{total_view:,} mentions")

    for l1_code in sorted(regular_codes, key=sort_key):
        l1_node = tree[l1_code]
        cnt = _node_count(l1_node)
        l1_id = f"{ROOT}/{l1_code}"
        add(l1_id, f"{l1_code}  {l1_node['title']}", ROOT, cnt, f"{l1_code} · {cnt:,}")
        walk(l1_node.get("children", {}), l1_id, 2)

    if lang_codes:
        lang_total = sum(_node_count(tree[lc]) for lc in lang_codes)
        lang_id = f"{ROOT}/LANG"
        add(lang_id, "Languages", ROOT, lang_total, f"{lang_total:,} mentions")
        for lc in sorted(lang_codes):
            lc_node = tree[lc]
            lc_cnt  = _node_count(lc_node)
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


# ── Rendering ─────────────────────────────────────────────────

def get_score_color(score):
    score_pct = score * 100
    if score_pct >= 90:
        return "#2d5a2d", "#fff"
    elif score_pct >= 80:
        return "#4a7a4a", "#fff"
    elif score_pct >= 75:
        return "#76b076", "#fff"
    return "#d97d3a", "#fff"


def build_kzis_html(kzis_matches):
    if not kzis_matches:
        return ""
    rows = ""
    for name, sim, rank in kzis_matches:
        bg, fg = get_score_color(sim)
        rows += (
            f'<div class="kr"><span class="kr-n">{rank}. {name}</span>'
            f'<span class="kr-s" style="background:{bg};color:{fg}">{sim:.2f}</span></div>'
        )
    return f'<div class="kzis-sec"><div class="kzis-lbl">Standardized KZIS Occupations</div>{rows}</div>'


def render_card(title, count, similarity=None, kzis_matches=None):
    meta = f"{count} job offer{'s' if count != 1 else ''}"
    if similarity is not None:
        meta += f" &middot; relevance: {similarity:.2f}"
    kzis_html = build_kzis_html(kzis_matches)
    st.markdown(
        f'<div class="rcard"><div class="rcard-title">{title}</div>'
        f'<div class="rcard-meta">{meta}</div>{kzis_html}</div>',
        unsafe_allow_html=True,
    )


def render_offer_with_skills(job_id, title):
    count = get_job_count(title)
    kzis  = get_kzis_matches(title)
    top1_items, unique_skills = get_offer_skills(job_id)

    kzis_html = build_kzis_html(kzis)
    st.markdown(
        f'<div class="rcard"><div class="rcard-title">{title}</div>'
        f'<div class="rcard-meta">{count} job offer{"s" if count != 1 else ""}</div>'
        f'{kzis_html}</div>',
        unsafe_allow_html=True,
    )

    if unique_skills:
        html = ""
        for label, stype, sim, cnt in unique_skills[:20]:
            tag = (stype or "").replace("skill/competence", "competence")
            bg, fg = get_score_color(sim)
            html += (
                f'<div class="skill-row"><span class="skill-name">{label}</span>'
                f'<span class="skill-type-tag">{tag}</span>'
                f'<span class="skill-score" style="background:{bg};color:{fg}">{sim:.2f}</span></div>'
            )
        extra = len(unique_skills) - 20
        if extra > 0:
            html += f'<div class="info-box">+ {extra} more skills</div>'
        st.markdown(
            f'<div class="sec-label">Mapped ESCO Skills ({len(unique_skills)} unique)</div>{html}',
            unsafe_allow_html=True,
        )

    if top1_items:
        req  = [(t, s, sim) for typ, t, s, _, sim in top1_items if typ == "requirement"]
        resp = [(t, s, sim) for typ, t, s, _, sim in top1_items if typ == "responsibility"]

        if req:
            rows = ""
            for text, skill, sim in req:
                bg, fg = get_score_color(sim)
                rows += (
                    f'<div class="item-row"><span class="item-type item-type-req">req</span>'
                    f'{text}<div class="item-skill">&#8594; <span class="item-skill-name">{skill}</span> '
                    f'<span class="skill-score" style="background:{bg};color:{fg}">{sim:.2f}</span></div></div>'
                )
            st.markdown(
                f'<div class="sec-label">Requirements ({len(req)})</div><div class="rcard">{rows}</div>',
                unsafe_allow_html=True,
            )

        if resp:
            rows = ""
            for text, skill, sim in resp:
                bg, fg = get_score_color(sim)
                rows += (
                    f'<div class="item-row"><span class="item-type item-type-resp">resp</span>'
                    f'{text}<div class="item-skill">&#8594; <span class="item-skill-name">{skill}</span> '
                    f'<span class="skill-score" style="background:{bg};color:{fg}">{sim:.2f}</span></div></div>'
                )
            st.markdown(
                f'<div class="sec-label">Responsibilities ({len(resp)})</div><div class="rcard">{rows}</div>',
                unsafe_allow_html=True,
            )


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
    <div style="font-size:1.02rem; font-weight:600; color:#3e648a; margin-top:0.45rem;">Job title: "Cloud Engineer" (<i>Inżynier Chmurowy</i>)</div>
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
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Job titles", "Skills Search", "Skills Stats", "Skill Trends", "NACE",
        "Ukrainian-Targeted Job Ads", "AI Skills",
    ])

    # ── Tab 1: Job Titles ──────────────────────────────────────
    with tab1:
        st.markdown('<div class="sec-label">Job Offers by KZIS Category</div>', unsafe_allow_html=True)
        st.components.v1.iframe(
            "https://datawrapper.dwcdn.net/U2ao5/11/",
            height=320,
            scrolling=False,
        )
        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        st.markdown('<div class="sec-label">Search job offers by title or keyword</div>', unsafe_allow_html=True)

        query = st.text_input(
            "Search query",
            placeholder="Type a job title or keyword, e.g. pielęgniarz, data scientist...",
            label_visibility="collapsed",
        )

        col1, col2 = st.columns([1, 5])
        with col1:
            go = st.button("Search", type="primary", use_container_width=True)

        if go and query:
            with st.spinner("Searching..."):
                results, err = search_jobs_fts(query, top_k=30)
            if err:
                st.error(f"Search error: {err}")
            elif results:
                seen = {}
                for r in results:
                    tl = r["title"].lower()
                    if tl not in seen or r["similarity"] > seen[tl]["similarity"]:
                        seen[tl] = r

                enriched = []
                total_matching = 0
                for data in seen.values():
                    t   = data["title"]
                    cnt = get_job_count(t)
                    total_matching += cnt
                    enriched.append((t, cnt, data["similarity"], get_kzis_matches(t)))
                enriched.sort(key=lambda x: x[2], reverse=True)

                st.markdown(
                    f'<div class="sec-label">Matching job offers &mdash; '
                    f'{total_matching:,} offers across {len(enriched)} titles</div>',
                    unsafe_allow_html=True,
                )

                for t, cnt, sim, kzis in enriched[:5]:
                    render_card(t, cnt, sim, kzis)

                if len(enriched) > 5:
                    if st.button(f"Show {len(enriched) - 5} more results", use_container_width=True):
                        for t, cnt, sim, kzis in enriched[5:]:
                            render_card(t, cnt, sim, kzis)
            else:
                st.markdown('<div class="info-box">No results found.</div>', unsafe_allow_html=True)

    # ── Tab 2: Skills Search ───────────────────────────────────
    with tab2:
        st.markdown('<div class="sec-label">Top Skills in Job Offers</div>', unsafe_allow_html=True)
        st.components.v1.iframe(
            "https://datawrapper.dwcdn.net/FMGJD/5/",
            height=320,
            scrolling=False,
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
            key="skills_pick",
        )

        if chosen_title:
            offers  = get_sample_offers()
            matching = [(jid, t) for jid, t in offers if t == chosen_title]
            if matching:
                job_id, title = matching[0]
                render_offer_with_skills(job_id, title)

    # ── Tab 3: Skills Stats ────────────────────────────────────
    with tab3:
        cache = load_skills_cache(_cache_mtime())
        meta  = cache["meta"]

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
            "skills":    {"root": "#1a1a2e", "S": "#3a6b8c", "T": "#2d6a2d", "LANG": "#6a2d5a"},
            "knowledge": {"root": "#1a1a2e", "default": "#7a5a1a"},
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
            marker=dict(colors=node_colors, line=dict(width=1.5, color="#fff")),
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

    # ── Tab 4: Skill Trends ────────────────────────────────────
    with tab4:
        st.markdown(
            '<div class="sec-label">Search for a skill or knowledge area to see weekly trends</div>',
            unsafe_allow_html=True,
        )

        top_skills  = get_top_skills(500)
        all_options = [f"{label}  ({total_m:,})" for sid, label, total_m, _ in top_skills]
        all_map     = {opt: (sid, label) for opt, (sid, label, _, _) in zip(all_options, top_skills)}

        chosen = st.selectbox(
            "Select skill",
            [""] + all_options,
            index=0,
            placeholder="Type to search skills, e.g. Python, zarządzanie, Excel…",
            label_visibility="collapsed",
            key="trend_pick",
        )

        sid   = None
        label = None
        if chosen and chosen in all_map:
            sid, label = all_map[chosen]

        if sid is not None:
            trend        = get_skill_trend(sid)
            period_totals = get_period_totals()

            if trend:
                weeks    = [r[0] for r in trend]
                mentions = [r[1] for r in trend]
                pct      = [r[2] / period_totals.get(r[0], 1) * 100 for r in trend]
                week_labels = [w.replace("2025-", "") for w in weeks]

                WIN = 4
                def rolling_avg(series, window=WIN):
                    out = []
                    for i in range(len(series)):
                        start = max(0, i - window + 1)
                        out.append(sum(series[start:i+1]) / (i - start + 1))
                    return out

                smooth_mentions = rolling_avg(mentions)
                smooth_pct      = rolling_avg(pct)

                FORECAST_N = 4
                def arima_forecast(series, steps=FORECAST_N):
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            model = ARIMA(series, order=(1, 1, 1))
                            fit   = model.fit()
                            fc    = fit.forecast(steps=steps)
                            ci    = fit.get_forecast(steps=steps).conf_int(alpha=0.2)
                            return list(fc), list(ci[:, 0]), list(ci[:, 1])
                    except Exception:
                        return None, None, None

                fc_labels = ["W01*", "W02*", "W03*", "W04*"]
                fc_m_vals, fc_lo_m, fc_hi_m = arima_forecast(smooth_mentions)
                fc_p_vals, fc_lo_p, fc_hi_p = arima_forecast(smooth_pct)

                st.markdown(
                    f'<div class="rcard-title" style="margin-top:1rem">{label}</div>',
                    unsafe_allow_html=True,
                )

                TICK       = dict(size=14, color="#000")
                AXIS_TITLE = dict(size=14, color="#000")
                CHART_FONT = dict(size=14, color="#000")

                fig = pgo.Figure()
                fig.add_trace(pgo.Scatter(
                    x=week_labels, y=mentions, name="Weekly",
                    mode="lines+markers",
                    line=dict(color="rgba(58,107,140,0.25)", width=1),
                    marker=dict(size=3, color="rgba(58,107,140,0.35)"),
                    hovertemplate="%{x}<br><b>%{y:,}</b> mentions<extra></extra>",
                ))
                fig.add_trace(pgo.Scatter(
                    x=week_labels, y=smooth_mentions, name="Smoothed (4w avg)",
                    mode="lines",
                    line=dict(color="#3a6b8c", width=3.5),
                    hovertemplate="%{x}<br><b>%{y:,.0f}</b> (4w avg)<extra></extra>",
                ))
                if fc_m_vals is not None:
                    ci_x = [week_labels[-1]] + fc_labels + fc_labels[::-1] + [week_labels[-1]]
                    ci_y = [smooth_mentions[-1]] + fc_hi_m + fc_lo_m[::-1] + [smooth_mentions[-1]]
                    fig.add_trace(pgo.Scatter(
                        x=ci_x, y=ci_y, fill="toself",
                        fillcolor="rgba(58,107,140,0.18)",
                        line=dict(width=0), mode="none",
                        name="Forecast range", hoverinfo="skip",
                    ))
                all_m  = mentions + (fc_hi_m if fc_hi_m else [])
                y_lo_m = min(mentions + (fc_lo_m if fc_lo_m else [])) * 0.85
                y_hi_m = max(all_m) * 1.08
                fig.update_layout(
                    title=dict(text="Mentions", font=dict(size=16, color="#111")),
                    height=370,
                    margin=dict(l=60, r=20, t=45, b=70),
                    plot_bgcolor="#fff", paper_bgcolor="#fff",
                    font=CHART_FONT,
                    xaxis=dict(showgrid=False, tickfont=TICK, tickangle=-45),
                    yaxis=dict(
                        showgrid=True, gridcolor="#e0e0e0",
                        title=dict(text="Mentions", font=AXIS_TITLE),
                        tickfont=TICK, range=[y_lo_m, y_hi_m],
                    ),
                    hovermode="x unified",
                    legend=dict(orientation="h", y=1.12, x=0, font=dict(size=13, color="#000")),
                )
                st.plotly_chart(fig, width="stretch")

                fig2 = pgo.Figure()
                fig2.add_trace(pgo.Scatter(
                    x=week_labels, y=pct, name="Weekly",
                    mode="lines+markers",
                    line=dict(color="rgba(45,106,45,0.25)", width=1),
                    marker=dict(size=3, color="rgba(45,106,45,0.35)"),
                    hovertemplate="%{x}<br><b>%{y:.2f}%</b> of offers<extra></extra>",
                ))
                fig2.add_trace(pgo.Scatter(
                    x=week_labels, y=smooth_pct, name="Smoothed (4w avg)",
                    mode="lines",
                    line=dict(color="#2d6a2d", width=3.5),
                    hovertemplate="%{x}<br><b>%{y:.2f}%</b> (4w avg)<extra></extra>",
                ))
                if fc_p_vals is not None:
                    ci_x = [week_labels[-1]] + fc_labels + fc_labels[::-1] + [week_labels[-1]]
                    ci_y = [smooth_pct[-1]] + fc_hi_p + fc_lo_p[::-1] + [smooth_pct[-1]]
                    fig2.add_trace(pgo.Scatter(
                        x=ci_x, y=ci_y, fill="toself",
                        fillcolor="rgba(45,106,45,0.18)",
                        line=dict(width=0), mode="none",
                        name="Forecast range", hoverinfo="skip",
                    ))
                all_p  = pct + (fc_hi_p if fc_hi_p else [])
                y_lo_p = min(pct + (fc_lo_p if fc_lo_p else [])) * 0.85
                y_hi_p = max(all_p) * 1.08
                fig2.update_layout(
                    title=dict(text="% of job offers", font=dict(size=16, color="#111")),
                    height=370,
                    margin=dict(l=60, r=20, t=45, b=70),
                    plot_bgcolor="#fff", paper_bgcolor="#fff",
                    font=CHART_FONT,
                    xaxis=dict(showgrid=False, tickfont=TICK, tickangle=-45),
                    yaxis=dict(
                        showgrid=True, gridcolor="#e0e0e0",
                        title=dict(text="% of offers", font=AXIS_TITLE),
                        tickfont=TICK, range=[y_lo_p, y_hi_p],
                    ),
                    hovermode="x unified",
                    legend=dict(orientation="h", y=1.12, x=0, font=dict(size=13, color="#000")),
                )
                st.plotly_chart(fig2, width="stretch")

                df_export = pd.DataFrame({
                    "week":                 weeks,
                    "mentions":             mentions,
                    "mentions_smooth_4w":   smooth_mentions,
                    "offer_count":          [r[2] for r in trend],
                    "total_offers":         [period_totals.get(r[0], 0) for r in trend],
                    "pct_offers":           pct,
                    "pct_offers_smooth_4w": smooth_pct,
                })
                if fc_m_vals is not None:
                    df_fc = pd.DataFrame({
                        "week":             fc_labels,
                        "fc_mentions_smooth": fc_m_vals,
                        "fc_mentions_lo":   fc_lo_m,
                        "fc_mentions_hi":   fc_hi_m,
                        "fc_pct_smooth":    fc_p_vals if fc_p_vals else [None] * FORECAST_N,
                        "fc_pct_lo":        fc_lo_p   if fc_lo_p   else [None] * FORECAST_N,
                        "fc_pct_hi":        fc_hi_p   if fc_hi_p   else [None] * FORECAST_N,
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

    # ── Tab 5: NACE ────────────────────────────────────────────
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

        TICK_S = dict(size=13, color="#425466")
        AXIS_T = dict(size=13, color="#425466")
        FONT_S = dict(size=13, color="#425466")

        pct_all  = [v / total_all  * 100 for v in count_all]
        pct_ours = [v / total_ours * 100 for v in count_ours]

        fig = pgo.Figure()
        fig.add_trace(pgo.Bar(
            x=bins, y=pct_all,
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
                "(<b>%{customdata[1]}%</b>)<extra></extra>"
            ),
        ))

        pct_offers_ours = [
            v / total_offers_ours * 100 if total_offers_ours else 0
            for v in offers_ours
        ]
        fig.add_trace(pgo.Bar(
            x=bins, y=pct_ours,
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
                "(<b>%{customdata[3]}%</b> of all offers)<extra></extra>"
            ),
        ))

        y_max = max(max(pct_all), max(pct_ours)) * 1.18
        fig.update_layout(
            barmode="group",
            height=480,
            margin=dict(l=60, r=20, t=30, b=60),
            plot_bgcolor="#fff", paper_bgcolor="#fff",
            font=FONT_S,
            xaxis=dict(
                title=dict(text="Number of NACE codes per ESCO occupation", font=AXIS_T),
                tickfont=TICK_S, showgrid=False,
            ),
            yaxis=dict(
                title=dict(text="% of ESCO occupations", font=AXIS_T),
                tickfont=TICK_S, showgrid=True, gridcolor="#e8e8e8",
                ticksuffix="%", range=[0, y_max],
            ),
            legend=dict(orientation="h", y=1.05, x=0, font=dict(size=13, color="#516274")),
            bargap=0.2, bargroupgap=0.05,
        )
        st.plotly_chart(fig, use_container_width=True)
        _dta_btn(
            pd.DataFrame({
                "nace_codes_per_occupation": bins,
                "pct_all_esco":      [round(v, 2) for v in pct_all],
                "n_all_esco":        count_all,
                "pct_our_job_ads":   [round(v, 2) for v in pct_ours],
                "n_our_job_ads":     count_ours,
                "n_offers_ours":     offers_ours,
                "pct_offers_ours":   [round(v, 2) for v in pct_offers_ours],
            }),
            "nace_histogram.dta", "dta_nace_hist",
        )

        # ── NACE Treemap ──────────────────────────────────────
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
            if nid == 'root':
                return pl_fallback
            if len(nid) == 1:
                return _NACE_EN_SEC.get(nid, pl_fallback)
            numeric = nid.lstrip('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
            div2 = numeric[:2] if len(numeric) >= 2 else numeric
            en = _NACE_EN_DIV.get(div2)
            return en if en else pl_fallback

        fixed_labels = [
            _nace_en_label(nid, lbl)
            for nid, lbl in zip(tm['ids'], tm['labels'])
        ]

        _SEC_COLORS = {
            'N': '#1B4F9C', 'G': '#16803D', 'C': '#C2410C', 'H': '#0E7490',
            'K': '#5B21B6', 'O': '#3730A3', 'J': '#0369A1', 'M': '#065F46',
            'F': '#92400E', 'Q': '#991B1B', 'P': '#0C4A6E', 'L': '#831843',
            'I': '#9A3412', 'R': '#7E22CE', 'S': '#3F6212', 'T': '#78350F',
            'D': '#BE185D', 'A': '#0F766E', 'E': '#155E75', 'B': '#374151',
            'U': '#475569', 'V': '#64748B',
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
            lines, current, consumed = [], "", 0
            for part in parts:
                candidate = part if not current else f"{current} {part}"
                if len(candidate) <= width:
                    current  = candidate
                    consumed += 1
                else:
                    if current:
                        lines.append(current)
                    current  = part
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
            p  = _parent_map_tm.get(nid, 'root')
            if p in ('root', ''):
                return 1
            pp = _parent_map_tm.get(p, 'root')
            if pp in ('root', ''):
                return 2
            return 3

        node_colors  = []
        display_text = []
        for nid, lbl, c in zip(tm['ids'], fixed_labels, tm['custom']):
            d    = _depth_tm(nid)
            sec  = _sec_letter(nid)
            base = _SEC_COLORS.get(sec, '#64748B')
            code = c.get('code', '') or nid
            if nid == 'root':
                node_colors.append('#334155')
                display_text.append('')
            elif d == 1:
                node_colors.append(base)
                display_text.append(f"<b>{_wrap_tm_label(lbl, width=18, max_lines=2)}</b>")
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
            marker=dict(colors=node_colors, line=dict(width=1.5, color='rgba(255,255,255,0.6)')),
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
                "nace_id":      tm["ids"],
                "sector_name":  tm["labels"],
                "parent_id":    tm["parents"],
                "job_count":    tm["values"],
                "nace_code":    [c.get("code", "") for c in tm["custom"]],
                "pct_of_total": [c.get("pct", 0.0)  for c in tm["custom"]],
            }),
            "nace_treemap.dta", "dta_nace_tm",
        )

        # ── Monthly NACE section share ────────────────────────
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

        _month_totals: dict[str, int]             = collections.defaultdict(int)
        _sec_month_cnt: dict[str, dict[str, int]] = collections.defaultdict(lambda: collections.defaultdict(int))
        for m, s, c in zip(mo['months'], mo['sections'], mo['counts']):
            _month_totals[m]      += c
            _sec_month_cnt[s][m]  += c

        _all_months  = sorted(_month_totals.keys())
        _sec_totals  = {s: sum(cnts.values()) for s, cnts in _sec_month_cnt.items()}
        _ranked_secs = sorted(_sec_totals, key=lambda s: _sec_totals[s], reverse=True)

        _sec_options     = [f"{s} — {_NACE_EN_SEC.get(s, s)}" for s in _ranked_secs]
        _option_to_letter = {opt: opt[0] for opt in _sec_options}
        _default_top5    = _sec_options[:5]

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

            fig_mo       = pgo.Figure()
            _mo_export_rows = []

            for sec in chosen_letters:
                cnts  = _sec_month_cnt[sec]
                pcts  = [
                    round(cnts.get(m, 0) / _month_totals[m] * 100, 2) if _month_totals[m] else 0
                    for m in _all_months
                ]
                lbl_mo = _NACE_EN_SEC.get(sec, sec)
                color  = _SEC_COLORS.get(sec, '#64748B')

                fig_mo.add_trace(pgo.Scatter(
                    x=_all_months, y=pcts,
                    name=f"{sec} — {lbl_mo}",
                    mode='lines+markers',
                    line=dict(width=2.5, color=color),
                    marker=dict(size=7, color=color),
                    hovertemplate=(
                        f"<b>{sec} — {lbl_mo}</b><br>"
                        "%{x}<br>Share: <b>%{y:.2f}%</b><extra></extra>"
                    ),
                ))
                for m, p in zip(_all_months, pcts):
                    _mo_export_rows.append({
                        "month":                  m,
                        "nace_section":           sec,
                        "section_name":           lbl_mo,
                        "pct_of_monthly_offers":  p,
                        "count":                  cnts.get(m, 0),
                    })

            _month_labels = [
                m.split("-")[1] + "/" + m.split("-")[0][2:]
                for m in _all_months
            ]

            fig_mo.update_layout(
                height=440,
                margin=dict(l=55, r=20, t=20, b=55),
                plot_bgcolor="#fff", paper_bgcolor="#fff",
                font=dict(size=13, color="#425466"),
                xaxis=dict(
                    tickvals=_all_months, ticktext=_month_labels,
                    tickfont=dict(size=12, color="#425466"),
                    showgrid=False, tickangle=-45,
                ),
                yaxis=dict(
                    title=dict(text="% of monthly offers", font=dict(size=13, color="#425466")),
                    tickfont=dict(size=12, color="#425466"),
                    showgrid=True, gridcolor="#eaeaea",
                    ticksuffix="%", rangemode="tozero",
                ),
                legend=dict(
                    orientation="h", y=-0.22, x=0.5, xanchor="center",
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

    # ── Tab 6: Ukrainian-Targeted Job Ads ──────────────────────
    with tab6:
        _render_ua_tab()

    # ── Tab 7: AI Skills ───────────────────────────────────────
    with tab7:
        _render_ai_tab()


# ── AI Skills tab helpers ──────────────────────────────────────────────────


AI_TAB_CACHE = os.path.join(DATA_DIR, "ai_tab_cache.json")

_AI_COLOR = "#E55B52"
_ALL_COLOR_AI = "#94A3B8"

_CAT_COLORS = {
    "Core AI & Machine Learning": "#E55B52",
    "Natural Language Processing": "#10B981",
    "Computer Vision & Image Recognition": "#8B5CF6",
    "Data Science & Analytics": "#3B82F6",
    "AI Applications & Domain-Specific": "#F59E0B",
    "AI Applications & Predictive Modeling": "#EC4899",
    "AI Governance / Responsible AI": "#6366F1",
}


@st.cache_data
def _load_ai_cache():
    with open(AI_TAB_CACHE, "r", encoding="utf-8") as f:
        return json.loads(f.read())


def _ai_grouped_bar(title, labels, pct_ai, pct_all, color_a=_AI_COLOR, color_b=_ALL_COLOR_AI, name_a="AI offers", name_b="All offers"):
    fig = pgo.Figure()
    fig.add_trace(pgo.Bar(
        y=labels, x=pct_ai, name=name_a, orientation="h",
        marker=dict(color=color_a, cornerradius=4),
        text=[f"{v:.1f}%" for v in pct_ai],
        textposition="outside", textfont=dict(size=11, color=color_a),
    ))
    fig.add_trace(pgo.Bar(
        y=labels, x=pct_all, name=name_b, orientation="h",
        marker=dict(color=color_b, cornerradius=4),
        text=[f"{v:.1f}%" for v in pct_all],
        textposition="outside", textfont=dict(size=11, color=color_b),
    ))
    fig.update_layout(
        barmode="group", height=max(260, len(labels) * 42 + 80),
        margin=dict(l=10, r=40, t=10, b=30),
        plot_bgcolor="#fff", paper_bgcolor="#f8f9fa",
        yaxis=dict(autorange="reversed", tickfont=dict(size=12, color="#425466")),
        xaxis=dict(visible=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                    font=dict(size=12, color="#425466")),
        bargap=0.25, bargroupgap=0.08,
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

Word boundaries (`\\b`) are used to avoid false positives.

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

    st.markdown(
        '<div class="sec-label">AI Skills Demand in Polish Job Market</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-size:0.88rem;color:#778596;margin-bottom:0.6rem">'
        'Analysis of AI-related skills demand based on ESCO skill matching and keyword detection '
        'in job titles, requirements and responsibilities. '
        f'Based on <b>{ov["total_offers"]:,}</b> job offers.'
        '</div>',
        unsafe_allow_html=True,
    )

    if st.button("How are AI offers identified?", key="btn_ai_method", type="secondary"):
        _show_ai_methodology()

    c1, c2, c3, c4 = st.columns(4)
    _kpi_style = (
        'background:#fff;border:1px solid #e4e4ea;border-radius:12px;'
        'padding:1rem 1.2rem;text-align:center;'
    )
    _kpi_val = 'font-size:1.6rem;font-weight:700;color:#1a1a2e;margin:0;line-height:1.3;'
    _kpi_lbl = 'font-size:0.75rem;color:#888;margin:0;'

    with c1:
        st.markdown(
            f'<div style="{_kpi_style}"><p style="{_kpi_val}">{ov["ai_combined"]:,}</p>'
            f'<p style="{_kpi_lbl}">AI job offers</p></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div style="{_kpi_style}"><p style="{_kpi_val}">{ov["pct_ai_all"]}%</p>'
            f'<p style="{_kpi_lbl}">of all offers</p></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div style="{_kpi_style}"><p style="{_kpi_val}">{ov["pct_ai_ict"]}%</p>'
            f'<p style="{_kpi_lbl}">of ICT offers</p></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f'<div style="{_kpi_style}"><p style="{_kpi_val}">{ov["ai_strict"]:,}</p>'
            f'<p style="{_kpi_lbl}">Core AI offers</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.82rem;color:#778596;margin:0.5rem 0 1rem">'
        f'ESCO skill match: <b>{ov["ai_esco"]:,}</b> · '
        f'Keyword match: <b>{ov["ai_keyword"]:,}</b> · '
        f'Both: <b>{ov["ai_both"]:,}</b> · '
        f'Extended AI only: <b>{ov["ai_extended_only"]:,}</b>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── 1. AI ESCO Skills frequency ──
    st.markdown(
        '<div style="font-size:1.02rem;font-weight:600;color:#1a1a2e;margin:0.5rem 0 0.45rem 0;">'
        'Most Demanded AI Skills (ESCO)</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-size:0.85rem;color:#778596;margin-bottom:0.8rem">'
        'Frequency of ESCO-classified AI skills found in job offers. '
        'Skills are matched via contextual embeddings against the full ESCO taxonomy. '
        'Colors indicate skill category (e.g. Core ML, NLP, Data Science).'
        '</div>',
        unsafe_allow_html=True,
    )
    sf = data["skills_freq"]
    sf_labels = [s["name_en"] for s in sf[:20]]
    sf_pct = [s["pct_ai"] for s in sf[:20]]
    sf_colors = [_CAT_COLORS.get(s["category"], "#94A3B8") for s in sf[:20]]
    fig_skills = pgo.Figure(pgo.Bar(
        y=sf_labels, x=sf_pct, orientation="h",
        marker=dict(color=sf_colors, cornerradius=4),
        text=[f"{v:.1f}%" for v in sf_pct],
        textposition="outside", textfont=dict(size=11),
        hovertemplate="<b>%{y}</b><br>%{x:.2f}% of AI offers<extra></extra>",
    ))
    fig_skills.update_layout(
        height=max(400, len(sf_labels) * 28 + 60),
        margin=dict(l=10, r=50, t=10, b=20),
        plot_bgcolor="#fff", paper_bgcolor="#f8f9fa",
        yaxis=dict(autorange="reversed", tickfont=dict(size=11, color="#425466")),
        xaxis=dict(visible=False),
    )
    st.plotly_chart(fig_skills, use_container_width=True)
    sf_df = pd.DataFrame(sf)
    sf_df.columns = ["Skill (EN)", "Skill (PL)", "Category", "Scope",
                     "N offers", "% AI offers", "% all offers", "% ICT offers"]
    _dta_btn(sf_df, "ai_skills_esco_freq.dta", "dta_ai_skills")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── 2. Keywords frequency ──
    st.markdown(
        '<div style="font-size:1.02rem;font-weight:600;color:#1a1a2e;margin:0.5rem 0 0.45rem 0;">'
        'Top AI Keywords Found in Offers</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-size:0.85rem;color:#778596;margin-bottom:0.8rem">'
        'Keywords detected via regex pattern matching in job titles, requirements and responsibilities. '
        'Counts show how many AI offers contain each keyword.'
        '</div>',
        unsafe_allow_html=True,
    )
    kf = data["kw_freq"][:20]
    fig_kw = pgo.Figure(pgo.Bar(
        y=[k["keyword"] for k in kf], x=[k["pct_ai"] for k in kf], orientation="h",
        marker=dict(color=_AI_COLOR, cornerradius=4),
        text=[f'{k["n"]:,}' for k in kf],
        textposition="outside", textfont=dict(size=11, color="#555"),
    ))
    fig_kw.update_layout(
        height=max(360, len(kf) * 28 + 60),
        margin=dict(l=10, r=50, t=10, b=20),
        plot_bgcolor="#fff", paper_bgcolor="#f8f9fa",
        yaxis=dict(autorange="reversed", tickfont=dict(size=11, color="#425466")),
        xaxis=dict(visible=False),
    )
    st.plotly_chart(fig_kw, use_container_width=True)
    _dta_btn(pd.DataFrame(kf), "ai_keywords_freq.dta", "dta_ai_kw")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── 3. Seniority & Contract ──
    st.markdown(
        '<div style="font-size:0.85rem;color:#778596;margin-bottom:0.8rem">'
        'Distribution of seniority levels and contract types in AI offers compared to all offers. '
        'Original values are aggregated into broader categories (e.g. junior specialist + trainee + assistant → Junior).'
        '</div>',
        unsafe_allow_html=True,
    )
    col_s, col_c = st.columns(2)
    with col_s:
        st.markdown(
            '<div style="font-size:1.02rem;font-weight:600;color:#1a1a2e;margin:0.15rem 0 0.45rem 0;">'
            'Seniority Level</div>', unsafe_allow_html=True,
        )
        sen = data["seniority"]
        st.plotly_chart(
            _ai_grouped_bar("Seniority", [s["level"] for s in sen],
                            [s["pct_ai"] for s in sen], [s["pct_all"] for s in sen]),
            use_container_width=True,
        )
        _dta_btn(pd.DataFrame(sen), "ai_seniority.dta", "dta_ai_sen")
    with col_c:
        st.markdown(
            '<div style="font-size:1.02rem;font-weight:600;color:#1a1a2e;margin:0.15rem 0 0.45rem 0;">'
            'Contract Type</div>', unsafe_allow_html=True,
        )
        ct = data["contracts"]
        st.plotly_chart(
            _ai_grouped_bar("Contract", [c["type"] for c in ct],
                            [c["pct_ai"] for c in ct], [c["pct_all"] for c in ct]),
            use_container_width=True,
        )
        _dta_btn(pd.DataFrame(ct), "ai_contracts.dta", "dta_ai_ct")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── 4. Technologies ──
    st.markdown(
        '<div style="font-size:1.02rem;font-weight:600;color:#1a1a2e;margin:0.5rem 0 0.45rem 0;">'
        'Technologies Overrepresented in AI Offers</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-size:0.85rem;color:#778596;margin-bottom:0.4rem">'
        'Technologies from the job ad "technologies" field that appear disproportionately more '
        'in AI job offers compared to non-AI offers. Sorted by overrepresentation ratio.'
        '</div>',
        unsafe_allow_html=True,
    )
    if st.button("How is the overrepresentation ratio calculated?", key="btn_ai_overrep", type="secondary"):
        _show_overrep_methodology()
    tech = data["technologies"][:25]
    fig_tech = pgo.Figure()
    fig_tech.add_trace(pgo.Bar(
        y=[t["tech"] for t in tech], x=[t["pct_ai"] for t in tech],
        name="% in AI offers", orientation="h",
        marker=dict(color=_AI_COLOR, cornerradius=4),
        text=[f'{t["pct_ai"]:.1f}%' for t in tech],
        textposition="outside", textfont=dict(size=10, color=_AI_COLOR),
    ))
    fig_tech.add_trace(pgo.Bar(
        y=[t["tech"] for t in tech], x=[t["pct_non_ai"] for t in tech],
        name="% in non-AI offers", orientation="h",
        marker=dict(color=_ALL_COLOR_AI, cornerradius=4),
        text=[f'{t["pct_non_ai"]:.1f}%' for t in tech],
        textposition="outside", textfont=dict(size=10, color=_ALL_COLOR_AI),
    ))
    fig_tech.update_layout(
        barmode="group", height=max(420, len(tech) * 34 + 60),
        margin=dict(l=10, r=50, t=10, b=30),
        plot_bgcolor="#fff", paper_bgcolor="#f8f9fa",
        yaxis=dict(autorange="reversed", tickfont=dict(size=11, color="#425466")),
        xaxis=dict(visible=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                    font=dict(size=11, color="#425466")),
        bargap=0.2, bargroupgap=0.06,
    )
    st.plotly_chart(fig_tech, use_container_width=True)
    _dta_btn(pd.DataFrame(data["technologies"]), "ai_technologies.dta", "dta_ai_tech")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── 5. Co-occurring skills ──
    st.markdown(
        '<div style="font-size:1.02rem;font-weight:600;color:#1a1a2e;margin:0.5rem 0 0.45rem 0;">'
        'Co-occurring Skills (Overrepresented in AI Offers)</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-size:0.85rem;color:#778596;margin-bottom:0.4rem">'
        'Non-AI ESCO skills that appear significantly more often in AI offers than in other offers. '
        'The ratio label (e.g. 5.2x) shows how many times more likely a skill is to appear in an AI offer.'
        '</div>',
        unsafe_allow_html=True,
    )
    if st.button("How are co-occurring skills identified?", key="btn_ai_cooccur", type="secondary"):
        _show_cooccur_methodology()
    co = data["cooccur"]
    co_non_trans = [c for c in co if not c["transversal"]][:20]
    co_trans = [c for c in co if c["transversal"]][:15]
    fig_co = pgo.Figure()
    fig_co.add_trace(pgo.Bar(
        y=[c["name_en"] or c["name_pl"] for c in co_non_trans],
        x=[c["pct_ai"] for c in co_non_trans],
        orientation="h", marker=dict(color="#8B5CF6", cornerradius=4),
        text=[f'{c["ratio"]}x' for c in co_non_trans],
        textposition="outside", textfont=dict(size=10, color="#8B5CF6"),
    ))
    fig_co.update_layout(
        height=max(380, len(co_non_trans) * 30 + 60),
        margin=dict(l=10, r=50, t=10, b=20),
        plot_bgcolor="#fff", paper_bgcolor="#f8f9fa",
        yaxis=dict(autorange="reversed", tickfont=dict(size=11, color="#425466")),
        xaxis=dict(visible=False), showlegend=False,
    )
    st.plotly_chart(fig_co, use_container_width=True)
    _dta_btn(pd.DataFrame(co), "ai_cooccurring_skills.dta", "dta_ai_cooccur")

    if co_trans:
        st.markdown(
            '<div style="font-size:0.95rem;font-weight:600;color:#1a1a2e;margin:1rem 0 0.45rem 0;">'
            'Transversal Skills in AI Offers</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="font-size:0.85rem;color:#778596;margin-bottom:0.6rem">'
            'ESCO transversal skills (cross-sector soft skills like communication, teamwork, '
            'analytical thinking) overrepresented in AI job offers compared to the rest of the market.'
            '</div>',
            unsafe_allow_html=True,
        )
        fig_trans = pgo.Figure()
        fig_trans.add_trace(pgo.Bar(
            y=[c["name_en"] or c["name_pl"] for c in co_trans],
            x=[c["pct_ai"] for c in co_trans],
            orientation="h", marker=dict(color="#10B981", cornerradius=4),
            text=[f'{c["ratio"]}x' for c in co_trans],
            textposition="outside", textfont=dict(size=10, color="#10B981"),
        ))
        fig_trans.update_layout(
            height=max(300, len(co_trans) * 30 + 60),
            margin=dict(l=10, r=50, t=10, b=20),
            plot_bgcolor="#fff", paper_bgcolor="#f8f9fa",
            yaxis=dict(autorange="reversed", tickfont=dict(size=11, color="#425466")),
            xaxis=dict(visible=False), showlegend=False,
        )
        st.plotly_chart(fig_trans, use_container_width=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── 6. Location ──
    st.markdown(
        '<div style="font-size:1.02rem;font-weight:600;color:#1a1a2e;margin:0.5rem 0 0.45rem 0;">'
        'Location of AI Job Offers</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-size:0.85rem;color:#778596;margin-bottom:0.8rem">'
        'Cities extracted from the job ad location field using pattern matching. '
        'A single offer may appear in multiple cities if it lists several locations. '
        '"Remote" includes ads mentioning remote work, home office, or hybrid arrangements.'
        '</div>',
        unsafe_allow_html=True,
    )
    loc = data["locations"][:20]
    st.plotly_chart(
        _ai_grouped_bar("Location", [l["city"] for l in loc],
                        [l["pct_ai"] for l in loc], [l["pct_all"] for l in loc]),
        use_container_width=True,
    )
    _dta_btn(pd.DataFrame(data["locations"]), "ai_locations.dta", "dta_ai_loc")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── 7. ESCO Job Titles ──
    st.markdown(
        '<div style="font-size:1.02rem;font-weight:600;color:#1a1a2e;margin:0.5rem 0 0.45rem 0;">'
        'Top ESCO Occupations in AI Offers</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-size:0.85rem;color:#778596;margin-bottom:0.8rem">'
        'Each job ad title is matched to the closest ESCO occupation using contextual embeddings. '
        'Shows which standardised occupations are most common among AI-related job offers.'
        '</div>',
        unsafe_allow_html=True,
    )
    et = data["esco_titles"][:20]
    fig_et = pgo.Figure(pgo.Bar(
        y=[t["title_en"] or t["title_pl"] for t in et],
        x=[t["pct_ai"] for t in et], orientation="h",
        marker=dict(color="#3B82F6", cornerradius=4),
        text=[f'{t["n"]:,}' for t in et],
        textposition="outside", textfont=dict(size=11, color="#555"),
    ))
    fig_et.update_layout(
        height=max(400, len(et) * 30 + 60),
        margin=dict(l=10, r=50, t=10, b=20),
        plot_bgcolor="#fff", paper_bgcolor="#f8f9fa",
        yaxis=dict(autorange="reversed", tickfont=dict(size=11, color="#425466")),
        xaxis=dict(visible=False),
    )
    st.plotly_chart(fig_et, use_container_width=True)
    _dta_btn(pd.DataFrame(data["esco_titles"]), "ai_esco_titles.dta", "dta_ai_titles")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── 8. NACE Sections ──
    st.markdown(
        '<div style="font-size:1.02rem;font-weight:600;color:#1a1a2e;margin:0.5rem 0 0.45rem 0;">'
        'NACE Sectors of AI Job Offers</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-size:0.85rem;color:#778596;margin-bottom:0.8rem">'
        'Distribution of AI offers across NACE Rev. 2 economic sectors. '
        'Sector assignment is inferred from the matched ESCO occupation via the official EU crosswalk '
        '(see NACE tab for full methodology).'
        '</div>',
        unsafe_allow_html=True,
    )
    ns = data["nace_sections"]
    ns_labels = [f'{s["section"]} — {s["name"]}' for s in ns]
    fig_ns = pgo.Figure(pgo.Bar(
        y=ns_labels, x=[s["pct_ai"] for s in ns], orientation="h",
        marker=dict(color="#F59E0B", cornerradius=4),
        text=[f'{s["n"]:,}' for s in ns],
        textposition="outside", textfont=dict(size=11, color="#555"),
    ))
    fig_ns.update_layout(
        height=max(380, len(ns) * 30 + 60),
        margin=dict(l=10, r=50, t=10, b=20),
        plot_bgcolor="#fff", paper_bgcolor="#f8f9fa",
        yaxis=dict(autorange="reversed", tickfont=dict(size=11, color="#425466")),
        xaxis=dict(visible=False),
    )
    st.plotly_chart(fig_ns, use_container_width=True)
    _dta_btn(pd.DataFrame(ns), "ai_nace_sections.dta", "dta_ai_nace")


# ── Helpers ────────────────────────────────────────────────────

def _dta_btn(df: pd.DataFrame, filename: str, key: str) -> None:
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
    )


_UA_COLOR  = "#FBBF24"
_ALL_COLOR = "#3B82F6"

_UA_DATA = {
    "Position Level": {
        "labels": [
            "Intern / Trainee", "Assistant", "Junior Specialist",
            "Specialist (Mid)", "Senior Specialist", "Expert",
            "Manager / Coordinator", "Senior Manager", "Director", "Manual Worker",
        ],
        "all_pct": [0.8,  3.0,  9.3, 41.5, 11.6, 2.6, 8.2, 4.3, 1.1, 17.5],
        "ua_pct":  [0.9,  2.2,  7.7, 23.9,  9.4, 1.7, 7.5, 1.7, 0.2, 44.9],
    },
    "Contract Type": {
        "labels": [
            "Employment Contract", "Work-for-Hire", "Commission Contract",
            "B2B Contract", "Substitute Contract", "Agency Contract",
            "Temporary Employment", "Internship / Apprenticeship",
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
        "all_pct": [28.7],
        "ua_pct":  [49.5],
    },
}

_UA_TICK = dict(size=12, color="#425466")
_UA_AXIS = dict(size=12, color="#425466")
_UA_FONT = dict(size=12, color="#425466")

_UA_LAYOUT_BASE = dict(
    plot_bgcolor="#fff",
    paper_bgcolor="#fff",
    font=_UA_FONT,
    margin=dict(l=10, r=20, t=58, b=10),
    legend=dict(
        orientation="h", y=1.03, yanchor="bottom", x=0, xanchor="left",
        font=dict(size=12, color="#516274"),
        bgcolor="rgba(0,0,0,0)",
    ),
    bargroupgap=0.08,
)


def _ua_grouped_bar_h(title: str, labels, all_pct, ua_pct):
    order   = sorted(range(len(labels)), key=lambda i: (all_pct[i] + ua_pct[i]) / 2)
    labels  = [labels[i]  for i in order]
    all_pct = [all_pct[i] for i in order]
    ua_pct  = [ua_pct[i]  for i in order]

    fig = pgo.Figure()
    fig.add_trace(pgo.Bar(
        y=labels, x=all_pct, name="All job ads",
        orientation="h", marker_color=_ALL_COLOR, marker_line=dict(width=0),
        text=[f"{v:.1f}%" for v in all_pct], textposition="outside",
        textfont=dict(size=10, color=_ALL_COLOR),
        hovertemplate="<b>%{y}</b><br>All jobs: <b>%{x:.1f}%</b><extra></extra>",
    ))
    fig.add_trace(pgo.Bar(
        y=labels, x=ua_pct, name="Ukrainian-targeted ads",
        orientation="h", marker_color=_UA_COLOR, marker_line=dict(width=0),
        text=[f"{v:.1f}%" for v in ua_pct], textposition="outside",
        textfont=dict(size=10, color="#B45309"),
        hovertemplate="<b>%{y}</b><br>Ukrainian-targeted: <b>%{x:.1f}%</b><extra></extra>",
    ))
    x_max  = max(max(all_pct), max(ua_pct)) * 1.22
    n      = len(labels)
    height = max(340, n * 52)
    fig.update_layout(
        **_UA_LAYOUT_BASE,
        barmode="group", height=height,
        xaxis=dict(ticksuffix="%", tickfont=_UA_TICK, showgrid=True, gridcolor="#eeeeee", range=[0, x_max]),
        yaxis=dict(tickfont=dict(size=11, color="#425466"), autorange="reversed"),
    )
    return fig


def _ua_grouped_bar_v(title: str, labels, all_pct, ua_pct):
    order   = sorted(range(len(labels)), key=lambda i: (all_pct[i] + ua_pct[i]) / 2, reverse=True)
    labels  = [labels[i]  for i in order]
    all_pct = [all_pct[i] for i in order]
    ua_pct  = [ua_pct[i]  for i in order]

    fig = pgo.Figure()
    fig.add_trace(pgo.Bar(
        x=labels, y=all_pct, name="All job ads",
        marker_color=_ALL_COLOR, marker_line=dict(width=0),
        text=[f"{v:.1f}%" for v in all_pct], textposition="outside",
        textfont=dict(size=11, color=_ALL_COLOR),
        hovertemplate="<b>%{x}</b><br>All jobs: <b>%{y:.1f}%</b><extra></extra>",
    ))
    fig.add_trace(pgo.Bar(
        x=labels, y=ua_pct, name="Ukrainian-targeted ads",
        marker_color=_UA_COLOR, marker_line=dict(width=0),
        text=[f"{v:.1f}%" for v in ua_pct], textposition="outside",
        textfont=dict(size=11, color="#B45309"),
        hovertemplate="<b>%{x}</b><br>Ukrainian-targeted: <b>%{y:.1f}%</b><extra></extra>",
    ))
    y_max = max(max(all_pct), max(ua_pct)) * 1.22
    fig.update_layout(
        **_UA_LAYOUT_BASE,
        barmode="group", height=360,
        xaxis=dict(tickfont=dict(size=11, color="#425466"), showgrid=False),
        yaxis=dict(ticksuffix="%", tickfont=_UA_TICK, showgrid=True, gridcolor="#eeeeee", range=[0, y_max]),
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

    d = _UA_DATA["Position Level"]
    st.markdown(
        '<div style="font-size:1.02rem;font-weight:600;color:#1a1a2e;margin:0.15rem 0 0.45rem 0;">Position Level</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(_ua_grouped_bar_h("Position Level", d["labels"], d["all_pct"], d["ua_pct"]), use_container_width=True)
    _dta_btn(
        pd.DataFrame({"option": d["labels"], "pct_all_jobs": d["all_pct"], "pct_ua_targeted": d["ua_pct"]}),
        "ua_position_level.dta", "dta_pos",
    )

    st.markdown('<hr style="border:none;border-top:1px solid #eee;margin:0.5rem 0 1rem">', unsafe_allow_html=True)

    d = _UA_DATA["Contract Type"]
    st.markdown(
        '<div style="font-size:1.02rem;font-weight:600;color:#1a1a2e;margin:0.15rem 0 0.45rem 0;">Contract Type</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(_ua_grouped_bar_h("Contract Type", d["labels"], d["all_pct"], d["ua_pct"]), use_container_width=True)
    _dta_btn(
        pd.DataFrame({"option": d["labels"], "pct_all_jobs": d["all_pct"], "pct_ua_targeted": d["ua_pct"]}),
        "ua_contract_type.dta", "dta_contract",
    )

    st.markdown('<hr style="border:none;border-top:1px solid #eee;margin:0.5rem 0 1rem">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        d = _UA_DATA["Work Dimension"]
        st.markdown(
            '<div style="font-size:1.02rem;font-weight:600;color:#1a1a2e;margin:0.15rem 0 0.45rem 0;">Work Dimension</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(_ua_grouped_bar_v("Work Dimension", d["labels"], d["all_pct"], d["ua_pct"]), use_container_width=True)
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
        st.plotly_chart(_ua_grouped_bar_v("Work Mode", d["labels"], d["all_pct"], d["ua_pct"]), use_container_width=True)
        _dta_btn(
            pd.DataFrame({"option": d["labels"], "pct_all_jobs": d["all_pct"], "pct_ua_targeted": d["ua_pct"]}),
            "ua_work_mode.dta", "dta_mode",
        )

    st.markdown('<hr style="border:none;border-top:1px solid #eee;margin:0.5rem 0 1rem">', unsafe_allow_html=True)

    d = _UA_DATA["Salary Transparency"]
    st.markdown(
        '<div style="font-size:1.02rem;font-weight:600;color:#1a1a2e;margin:0.15rem 0 0.45rem 0;">Salary Transparency</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(_ua_grouped_bar_v("Salary Transparency", d["labels"], d["all_pct"], d["ua_pct"]), use_container_width=True)
    _dta_btn(
        pd.DataFrame({"option": d["labels"], "pct_all_jobs": d["all_pct"], "pct_ua_targeted": d["ua_pct"]}),
        "ua_salary_transparency.dta", "dta_salary",
    )


if __name__ == "__main__":
    main()
