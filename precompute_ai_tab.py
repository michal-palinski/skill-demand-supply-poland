"""
Pre-compute AI demand data for the Streamlit AI Skills tab.
Outputs: app_deploy/ai_tab_cache.json
"""

import sqlite3
import json
import re
import pandas as pd
import numpy as np
from collections import Counter

print("Loading data...")

ai_ref = pd.read_csv("esco_ai_skills.csv", sep=";")
pl_skills = pd.read_csv(
    "ESCO dataset - v1.2.1 - classification - pl - csv/skills_pl.csv"
)
ai_ref = ai_ref.merge(pl_skills[["conceptUri", "preferredLabel"]], on="conceptUri", how="left")

ai_skills_pl = {}
for _, row in ai_ref.iterrows():
    label = row["preferredLabel"].strip().lower()
    ai_skills_pl[label] = {
        "name_en": row["ESCO Skill/Knowledge Name"],
        "category": row["Revised Category"],
        "scope": row["Portfolio Scope"],
    }

en_skills = pd.read_csv("ESCO dataset - v1.2.1 - classification - en - csv/skills_en.csv")
skill_uri_to_en = dict(zip(en_skills["conceptUri"], en_skills["preferredLabel"]))
skill_pl_to_en = {}
for _, r in pl_skills.iterrows():
    pl_label = str(r["preferredLabel"]).strip().lower()
    uri = r["conceptUri"]
    if uri in skill_uri_to_en:
        skill_pl_to_en[pl_label] = skill_uri_to_en[uri]

transversal = pd.read_csv(
    "ESCO dataset - v1.2.1 - classification - pl - csv/transversalSkillsCollection_pl.csv"
)
transversal_set = set(transversal["preferredLabel"].str.strip().str.lower())

occ_pl = pd.read_csv("ESCO dataset - v1.2.1 - classification - pl - csv/occupations_pl.csv")
occ_en = pd.read_csv("ESCO dataset - v1.2.1 - classification - en - csv/occupations_en.csv")
occ_uri_to_en = dict(zip(occ_en["conceptUri"], occ_en["preferredLabel"]))
occ_pl_to_en = {}
for _, r in occ_pl.iterrows():
    pl_label = str(r["preferredLabel"]).strip().lower()
    uri = r["conceptUri"]
    if uri in occ_uri_to_en:
        occ_pl_to_en[pl_label] = occ_uri_to_en[uri]

conn = sqlite3.connect("jobs_database.db")
df = pd.read_sql_query(
    """SELECT id, title, requirements, responsibilities,
              contract_types, seniority_level, location, technologies,
              skills_esco_contextual, title_clean
       FROM job_ads""",
    conn,
)
conn.close()
print(f"Loaded {len(df)} job ads")

conn_nace = sqlite3.connect("job_nace.db")
nace_df = pd.read_sql_query(
    "SELECT job_id, esco_label, nace_code, nace_title FROM job_nace", conn_nace
)
conn_nace.close()

# ── Parse ESCO skills & flag AI ──────────────────────────────────────────────
print("Parsing ESCO skills...")


def parse_skills(raw):
    if not raw or pd.isna(raw):
        return []
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []


df["skills_parsed"] = df["skills_esco_contextual"].apply(parse_skills)


def check_ai_esco(skills_list):
    found = []
    for s in skills_list:
        esco_label = s.get("esco", "").strip().lower()
        if esco_label in ai_skills_pl:
            info = ai_skills_pl[esco_label]
            found.append({
                "esco_pl": esco_label,
                "esco_en": info["name_en"],
                "category": info["category"],
                "scope": info["scope"],
            })
    return found


df["ai_skills_esco"] = df["skills_parsed"].apply(check_ai_esco)
df["has_ai_esco"] = df["ai_skills_esco"].apply(lambda x: len(x) > 0)
df["has_ai_core_esco"] = df["ai_skills_esco"].apply(
    lambda x: any(s["scope"] == "Strict AI" for s in x)
)

# ── Keyword matching ─────────────────────────────────────────────────────────
print("Keyword matching...")

AI_KEYWORDS_STRICT = [
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

strict_pattern = re.compile("|".join(AI_KEYWORDS_STRICT), re.IGNORECASE)


def build_search_text(row):
    parts = []
    for col in ["title", "requirements", "responsibilities"]:
        v = row[col]
        if v and not pd.isna(v):
            parts.append(str(v))
    return " ".join(parts)


df["search_text"] = df.apply(build_search_text, axis=1)
df["has_ai_kw_strict"] = df["search_text"].apply(lambda t: bool(strict_pattern.search(t)))


def find_matched_keywords(text):
    if not text:
        return []
    return list(set(m.group() for m in strict_pattern.finditer(text)))


df["ai_keywords_found"] = df["search_text"].apply(find_matched_keywords)

df["is_ai_offer"] = df["has_ai_esco"] | df["has_ai_kw_strict"]
df["is_ai_strict"] = df["has_ai_core_esco"] | df["has_ai_kw_strict"]
df["is_ai_extended_only"] = df["is_ai_offer"] & ~df["is_ai_strict"]
df["is_ict"] = df["technologies"].notna() & (df["technologies"].str.strip() != "")
df["is_ai_and_ict"] = df["is_ai_offer"] & df["is_ict"]

total = len(df)
ai_offers = df[df["is_ai_offer"]]
non_ai = df[~df["is_ai_offer"]]
n_ai = len(ai_offers)
n_ict = int(df["is_ict"].sum())
n_ai_ict = int(df["is_ai_and_ict"].sum())

print(f"AI offers: {n_ai}, ICT offers: {n_ict}")

# ── Overview ──────────────────────────────────────────────────────────────────
overview = {
    "total_offers": total,
    "ai_esco": int(df["has_ai_esco"].sum()),
    "ai_keyword": int(df["has_ai_kw_strict"].sum()),
    "ai_combined": n_ai,
    "ai_both": int((df["has_ai_esco"] & df["has_ai_kw_strict"]).sum()),
    "ai_strict": int(df["is_ai_strict"].sum()),
    "ai_extended_only": int(df["is_ai_extended_only"].sum()),
    "ict_total": n_ict,
    "ai_in_ict": n_ai_ict,
    "pct_ai_all": round(n_ai / total * 100, 2),
    "pct_ai_ict": round(n_ai_ict / n_ict * 100, 2) if n_ict else 0,
}

# ── AI skills frequency ──────────────────────────────────────────────────────
skill_counts = Counter()
for skills in ai_offers["ai_skills_esco"]:
    for s in skills:
        key = (s["esco_en"], s["esco_pl"], s["category"], s["scope"])
        skill_counts[key] += 1

skills_freq = [
    {
        "name_en": k[0], "name_pl": k[1], "category": k[2], "scope": k[3],
        "n": v, "pct_ai": round(v / n_ai * 100, 2),
        "pct_all": round(v / total * 100, 3),
        "pct_ict": round(v / n_ict * 100, 3) if n_ict else 0,
    }
    for k, v in skill_counts.most_common()
]

# ── Keyword frequency ────────────────────────────────────────────────────────
kw_counts = Counter()
for kws in ai_offers["ai_keywords_found"]:
    for kw in kws:
        kw_counts[kw.lower()] += 1

kw_freq = [
    {"keyword": k, "n": v, "pct_ai": round(v / n_ai * 100, 2)}
    for k, v in kw_counts.most_common()
]

# ── Co-occurring skills ──────────────────────────────────────────────────────
print("Co-occurring skills...")
ai_all_skills = Counter()
non_ai_all_skills = Counter()

for _, row in ai_offers.iterrows():
    seen = set()
    for s in row["skills_parsed"]:
        label = s.get("esco", "").strip().lower()
        if label and label not in seen:
            seen.add(label)
            ai_all_skills[label] += 1

sample_non_ai = non_ai.sample(n=min(100_000, len(non_ai)), random_state=42)
for _, row in sample_non_ai.iterrows():
    seen = set()
    for s in row["skills_parsed"]:
        label = s.get("esco", "").strip().lower()
        if label and label not in seen:
            seen.add(label)
            non_ai_all_skills[label] += 1

n_non_ai_sample = len(sample_non_ai)

cooccur = []
for skill, ai_count in ai_all_skills.items():
    if skill in ai_skills_pl:
        continue
    non_ai_count = non_ai_all_skills.get(skill, 0)
    ai_rate = ai_count / n_ai
    non_ai_rate = non_ai_count / n_non_ai_sample if n_non_ai_sample else 0
    ratio = round(ai_rate / non_ai_rate, 2) if non_ai_rate > 0 else 999
    if ai_count < 50:
        continue
    en_label = skill_pl_to_en.get(skill, "")
    cooccur.append({
        "name_pl": skill, "name_en": en_label,
        "transversal": skill in transversal_set,
        "n_ai": ai_count,
        "pct_ai": round(ai_rate * 100, 2),
        "pct_non_ai": round(non_ai_rate * 100, 2),
        "ratio": ratio,
    })

cooccur.sort(key=lambda x: x["ratio"], reverse=True)

# ── Technologies ──────────────────────────────────────────────────────────────
print("Technologies...")


def parse_tech(val):
    if not val or pd.isna(val):
        return []
    return [t.strip() for t in str(val).split(",") if t.strip()]


ai_tech = Counter()
non_ai_tech = Counter()
for techs in ai_offers["technologies"].apply(parse_tech):
    for t in techs:
        ai_tech[t.lower()] += 1
for techs in sample_non_ai["technologies"].apply(parse_tech):
    for t in techs:
        non_ai_tech[t.lower()] += 1

tech_data = []
for tech, ai_count in ai_tech.items():
    if ai_count < 20:
        continue
    non_ai_count = non_ai_tech.get(tech, 0)
    ai_rate = ai_count / n_ai
    non_ai_rate = non_ai_count / n_non_ai_sample if n_non_ai_sample else 0
    ratio = round(ai_rate / non_ai_rate, 2) if non_ai_rate > 0 else 999
    tech_data.append({
        "tech": tech, "n_ai": ai_count,
        "pct_ai": round(ai_rate * 100, 2),
        "pct_non_ai": round(non_ai_rate * 100, 2),
        "ratio": ratio,
    })
tech_data.sort(key=lambda x: x["ratio"], reverse=True)

# ── Contract types ────────────────────────────────────────────────────────────
print("Contracts...")

CONTRACT_MAP = {
    "umowa o pracę": "Employment contract",
    "contract of employment": "Employment contract",
    "umowa o pracę tymczasową": "Temporary employment",
    "umowa na zastępstwo": "Temporary employment",
    "temporary staffing agreement": "Temporary employment",
    "substitution agreement": "Temporary employment",
    "kontrakt B2B": "B2B contract",
    "b2b contract": "B2B contract",
    "umowa zlecenie": "Contract of mandate",
    "contract of mandate": "Contract of mandate",
    "umowa o dzieło": "Contract for specific work",
    "contract for specific work": "Contract for specific work",
    "umowa agencyjna": "Agency contract",
    "agency agreement": "Agency contract",
    "umowa o staż / praktyki": "Internship / traineeship",
    "internship / apprenticeship contract": "Internship / traineeship",
}


def map_contracts(val):
    if not val or pd.isna(val):
        return ["No data"]
    raw = [c.strip() for c in str(val).split(",")]
    return list(set(CONTRACT_MAP.get(c.strip(), c.strip()) for c in raw))


ai_contracts = Counter()
all_contracts = Counter()
for contracts in ai_offers["contract_types"].apply(map_contracts):
    for c in contracts:
        ai_contracts[c] += 1
for contracts in df["contract_types"].apply(map_contracts):
    for c in contracts:
        all_contracts[c] += 1

contract_data = []
for c, ai_cnt in ai_contracts.most_common():
    all_cnt = all_contracts.get(c, 0)
    contract_data.append({
        "type": c, "n_ai": ai_cnt,
        "pct_ai": round(ai_cnt / n_ai * 100, 2),
        "pct_all": round(all_cnt / total * 100, 2),
    })

# ── Seniority ─────────────────────────────────────────────────────────────────
print("Seniority...")

SENIORITY_MAP = {
    "junior": ["młodszy specjalista (junior)", "junior specialist (junior)",
               "praktykant / stażysta", "trainee", "asystent", "assistant",
               "entry level & blue collar"],
    "mid": ["specjalista (mid / regular)", "specialist (mid / regular)"],
    "senior": ["starszy specjalista (senior)", "senior specialist (senior)",
               "ekspert", "expert"],
    "management": ["kierownik / koordynator", "manager / supervisor",
                   "menedżer", "team manager", "dyrektor", "director",
                   "prezes", "executive"],
    "blue collar": ["pracownik fizyczny"],
}
seniority_lookup = {}
for group, labels in SENIORITY_MAP.items():
    for l in labels:
        seniority_lookup[l] = group


def map_seniority(val):
    if not val or pd.isna(val):
        return ["No data"]
    parts = [p.strip().lower() for p in str(val).split(",")]
    return list(set(seniority_lookup.get(p, "Other") for p in parts))


ai_seniority = Counter()
all_seniority = Counter()
for levels in ai_offers["seniority_level"].apply(map_seniority):
    for l in levels:
        ai_seniority[l] += 1
for levels in df["seniority_level"].apply(map_seniority):
    for l in levels:
        all_seniority[l] += 1

SENIORITY_ORDER = ["junior", "mid", "senior", "management", "blue collar", "No data"]
seniority_data = []
for level in SENIORITY_ORDER:
    ai_cnt = ai_seniority.get(level, 0)
    all_cnt = all_seniority.get(level, 0)
    if ai_cnt == 0 and all_cnt == 0:
        continue
    seniority_data.append({
        "level": level.title(), "n_ai": ai_cnt,
        "pct_ai": round(ai_cnt / n_ai * 100, 2),
        "pct_all": round(all_cnt / total * 100, 2),
    })

# ── Location ──────────────────────────────────────────────────────────────────
print("Location...")

CITY_PATTERNS = {
    "Warszawa": r"warszaw", "Kraków": r"krak[oó]w", "Wrocław": r"wroc[lł]aw",
    "Poznań": r"pozna[nń]", "Gdańsk": r"gda[nń]sk", "Łódź": r"[lł][oó]d[zź]",
    "Katowice": r"katowic", "Lublin": r"lublin", "Szczecin": r"szczecin",
    "Bydgoszcz": r"bydgoszcz", "Białystok": r"bia[lł]ystok", "Gdynia": r"gdyni",
    "Rzeszów": r"rzesz[oó]w", "Toruń": r"toru[nń]", "Kielce": r"kielc",
    "Opole": r"opol", "Olsztyn": r"olsztyn", "Zielona Góra": r"zielona g[oó]r",
    "Gliwice": r"gliwic", "Częstochowa": r"cz[eę]stochow", "Radom": r"\bradom",
    "Sosnowiec": r"sosnowiec", "Bielsko-Biała": r"bielsko",
    "Remote": r"(zdaln|remote|home office)", "Abroad": r"zagrani",
}


def extract_city(location):
    if not location or pd.isna(location):
        return "No data"
    loc_lower = str(location).lower()
    cities = []
    for city, pat in CITY_PATTERNS.items():
        if re.search(pat, loc_lower):
            cities.append(city)
    return "; ".join(cities) if cities else "Other"


ai_city = Counter()
all_city = Counter()
for loc in ai_offers["location"].apply(extract_city):
    for c in loc.split("; "):
        ai_city[c.strip()] += 1
for loc in df["location"].apply(extract_city):
    for c in loc.split("; "):
        all_city[c.strip()] += 1

location_data = []
for city, cnt in ai_city.most_common():
    all_cnt = all_city.get(city, 0)
    location_data.append({
        "city": city, "n_ai": cnt,
        "pct_ai": round(cnt / n_ai * 100, 2),
        "pct_all": round(all_cnt / total * 100, 2),
    })

# ── ESCO job titles & NACE ───────────────────────────────────────────────────
print("Job titles & NACE...")
ai_ids = set(ai_offers["id"].values)
nace_ai = nace_df[nace_df["job_id"].isin(ai_ids)]

esco_titles = nace_ai["esco_label"].value_counts().head(30)
esco_title_data = []
for label, cnt in esco_titles.items():
    en = occ_pl_to_en.get(str(label).strip().lower(), "")
    esco_title_data.append({
        "title_pl": label, "title_en": en, "n": int(cnt),
        "pct_ai": round(cnt / n_ai * 100, 2),
    })

nace_section = nace_ai.copy()
nace_section["sec"] = nace_section["nace_code"].str[0]
NACE_SEC = {
    "A": "Agriculture", "B": "Mining", "C": "Manufacturing",
    "D": "Electricity, gas", "E": "Water supply", "F": "Construction",
    "G": "Wholesale & retail", "H": "Transportation", "I": "Accommodation & food",
    "J": "ICT", "K": "Finance & insurance", "L": "Real estate",
    "M": "Professional & scientific", "N": "Admin & support",
    "O": "Public admin", "P": "Education", "Q": "Health & social work",
    "R": "Arts & entertainment", "S": "Other services",
    "T": "Activities of households as employers",
}
nace_sec_counts = nace_section["sec"].value_counts()
nace_section_data = []
for sec, cnt in nace_sec_counts.items():
    nace_section_data.append({
        "section": sec, "name": NACE_SEC.get(sec, sec),
        "n": int(cnt), "pct_ai": round(cnt / n_ai * 100, 2),
    })

# ── Assemble and save ────────────────────────────────────────────────────────
cache = {
    "overview": overview,
    "skills_freq": skills_freq,
    "kw_freq": kw_freq,
    "cooccur": cooccur,
    "technologies": tech_data,
    "contracts": contract_data,
    "seniority": seniority_data,
    "locations": location_data,
    "esco_titles": esco_title_data,
    "nace_sections": nace_section_data,
}

out_path = "app_deploy/ai_tab_cache.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(cache, f, ensure_ascii=False, indent=2)

print(f"\nSaved to {out_path}")
print(f"AI offers: {n_ai:,} / {total:,} ({overview['pct_ai_all']}%)")
