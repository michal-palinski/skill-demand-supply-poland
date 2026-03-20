#!/usr/bin/env python3
"""
Create / update title_clean column in jobs_database.db.

Cleaning steps (in order):
  1. Zero-width / soft-hyphen chars
  2. Salary & price info (numbers + currency symbols)
  3. Market info  (rynek + country adjective / geo code)
  4. Language info (z j. X, z językiem X, with German, …)
  5. Gender markers  (m/f), (k/m), M/F, (She/He/They), …
  6. Feminatywy
       a. /-suffix inline  →  Specjalista/-tka  → Specjalista
       b. Word/suffix      →  Specjalista/ka    → Specjalista
       c. (FemAdj) FemWord / (MascAdj) MascWord  →  MascAdj MascWord
       d. Word / FemWord pairs  (shared root heuristic)
       e. Full feminine clause after  |  or long  /  split
  7. Final cleanup (dangling separators, whitespace)
"""

import re
import sqlite3
from tqdm import tqdm

DB_PATH = "jobs_database.db"
BATCH = 10_000

# ── helpers ───────────────────────────────────────────────────────────────────

LANG_EN = (
    r"German|English|French|Italian|Spanish|Dutch|Czech|Slovak|Romanian|"
    r"Portuguese|Polish|Ukrainian|Russian|Hungarian|Danish|Swedish|Norwegian|"
    r"Finnish|Turkish|Greek|Bulgarian|Croatian|Hebrew|Japanese|Chinese|Korean|"
    r"Arabic|Latvian|Lithuanian|Estonian|Serbian|Bosnian|Macedonian|Slovenian|"
    r"Albanian|Nordic[s]?|Scandinavian|Flemish|Catalan|Georgian|Armenian|"
    r"Azerbaijani|Montenegrin|Maltese"
)

LANG_PL = (
    r"niemiec\w+|angiel\w+|francusk\w+|włosk\w+|hiszpa[nń]\w+|"
    r"holenders\w+|niderlandz\w+|czeski\w+|słowack\w+|rumuńsk\w+|"
    r"portug\w+|polsk\w+|ukraiń\w+|rosyjs\w+|węgiers\w+|duńsk\w+|"
    r"szwedz\w+|norwesk\w+|fińsk\w+|tureck\w+|greck\w+|bułgars\w+|"
    r"chorwack\w+|serbsk\w+|słoweńsk\w+|macedońsk\w+|nordyck\w+|"
    r"skandynaw\w+|hebrajsk\w+|japońsk\w+|chiński\w+|arabsk\w+|"
    r"łotews\w+|litews\w+|estoń\w+"
)

# country/geo adjectives used in "rynek X" context
COUNTRY_ADJ = (
    r"niemiec\w+|ukra\w+|pols\w+|czeski\w+|słowack\w+|holenders\w+|"
    r"niderlandz\w+|francusk\w+|włosk\w+|hiszpa[nń]\w+|portug\w+|"
    r"bułgars\w+|chorwack\w+|serbsk\w+|rosyjs\w+|węgiers\w+|duńsk\w+|"
    r"szwedz\w+|norwesk\w+|nordyck\w+|skandynaw\w+|łotews\w+|litews\w+|"
    r"estoń\w+|arabsk\w+|tureck\w+|greck\w+|rumuńsk\w+|"
    r"DACH|EMEA|CEE|APAC|UE|EU|UK|US|[A-Z]{2,4}"
)

# feminine adjective forms (seniority) → their masculine counterparts
FEM_TO_MASC_ADJ = {
    "starsza": "starszy",
    "młodsza": "młodszy",
    "doświadczona": "doświadczony",
}
MASC_TO_FEM_ADJ = {v: k for k, v in FEM_TO_MASC_ADJ.items()}

# known feminatyw inline suffixes (after slash inside a single token)
FEM_INLINE_SUFFIXES = r"(?:tka|czka|sza|ini|nia|rka|lka|nka|ka|a)"


def _common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def _is_fem_pair(word1: str, word2: str) -> bool:
    """Return True if word2 looks like a feminatyw of word1 (or vice-versa)."""
    w1, w2 = word1.lower(), word2.lower()
    # must share at least 5 chars prefix (or 70 % of shorter word)
    threshold = max(5, int(min(len(w1), len(w2)) * 0.65))
    if _common_prefix_len(w1, w2) >= threshold:
        return True
    # explicit suffix check: word2 = word1 + suffix
    for suf in ("czka", "tka", "anka", "ini", "ka", "a"):
        if w2 == w1 + suf or w1 == w2 + suf:
            return True
    return False


def _replace_fem_pairs(text: str) -> str:
    """Replace  'MascWord / FemWord'  or  'FemWord / MascWord'  → keep shorter."""
    parts = re.split(r"(\s*/\s*)", text)
    if len(parts) < 3:
        return text

    result = [parts[0]]
    i = 1
    while i < len(parts) - 1:
        sep = parts[i]
        next_tok = parts[i + 1] if i + 1 < len(parts) else ""

        # both tokens must be single Polish-capitalised words
        prev_word = re.search(r"([A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]{3,})$", result[-1])
        next_word = re.match(r"([A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]{3,})", next_tok)

        if prev_word and next_word and _is_fem_pair(prev_word.group(1), next_word.group(1)):
            # drop the separator and the feminatyw word; keep the shorter (masculine)
            shorter = (
                prev_word.group(1)
                if len(prev_word.group(1)) <= len(next_word.group(1))
                else next_word.group(1)
            )
            # replace tail of result[-1] with the shorter form
            result[-1] = result[-1][: prev_word.start(1)] + shorter
            # skip the feminine token (but keep anything after it in next_tok)
            tail = next_tok[next_word.end(1):]
            result.append(tail)
            i += 2
        else:
            result.append(sep)
            i += 1

    if i < len(parts):
        result.append(parts[i])

    return "".join(result)


def clean_title(title: str) -> str:
    if not title:
        return title

    t = title.strip()

    # ── 0. zero-width / soft-hyphen chars ────────────────────────────────────
    t = re.sub(r"[\u00ad\u200b\u200c\u200d\u200e\u200f\ufeff]", "", t)

    # ── 1. salary / price info ────────────────────────────────────────────────
    CURR = r"(?:€|\$|zł|PLN|USD|EUR|GBP|CHF|SEK|NOK|DKK)"
    # parenthetical salary: (3800-4000 € netto), (540 - 620 zł)
    t = re.sub(
        rf"\s*\(\s*[\d][\d\s.,–\-]*\s*{CURR}[^)]*\)",
        "", t,
    )
    # salary with unit rate: 16,74 € b/h, 540 - 600€/ 40h  (after a space)
    t = re.sub(
        rf"\s+[\d][\d\s.,–\-/]*\s*{CURR}[\s\w/.,–\-]*",
        "", t,
    )
    # currency first: $500
    t = re.sub(rf"\s*{CURR}\s*[\d][\d\s.,–\-]*[\s\w/.,–\-]*", "", t)

    # ── 2. market / rynek info ────────────────────────────────────────────────
    # "- rynek ukraiński", "– rynek DACH", "(rynek UK)"
    t = re.sub(
        rf"\s*[\(\[]?\s*[-–—,]?\s*(?:na\s+)?rynek[u]?\s+(?:{COUNTRY_ADJ})\b[^\|\-–—,\)]*[\)\]]?",
        "", t, flags=re.IGNORECASE,
    )
    # "na rynek X" without dash if at end-ish
    t = re.sub(
        rf"\s+na\s+rynek[u]?\s+(?:{COUNTRY_ADJ})\b[^,\-–—\|]*",
        "", t, flags=re.IGNORECASE,
    )

    # ── 3. language info ──────────────────────────────────────────────────────
    # Polish: (z j. niemieckim), z językiem X, z j. X i Y
    t = re.sub(
        rf"\s*\(\s*z\s+j\.\s+(?:{LANG_PL})[^)]*\)",
        "", t, flags=re.IGNORECASE,
    )
    t = re.sub(
        rf"\s*\(\s*z\s+językiem\s+(?:{LANG_PL})[^)]*\)",
        "", t, flags=re.IGNORECASE,
    )
    t = re.sub(
        rf"\s+z\s+j\.\s+(?:{LANG_PL})(?:\s+(?:i|lub|or)\s+(?:{LANG_PL}))*",
        "", t, flags=re.IGNORECASE,
    )
    t = re.sub(
        rf"\s+z\s+językiem\s+(?:{LANG_PL})(?:\s+(?:i|lub|or)\s+(?:{LANG_PL}))*",
        "", t, flags=re.IGNORECASE,
    )
    t = re.sub(
        rf"\s+z\s+językami?\s+(?:{LANG_PL})(?:\s+(?:i|lub|or)\s+(?:{LANG_PL}))*",
        "", t, flags=re.IGNORECASE,
    )
    t = re.sub(
        rf"\s+z\s+znajomości\w*\s+języka\s+(?:{LANG_PL})",
        "", t, flags=re.IGNORECASE,
    )
    # English: "with German", "with English or French", "with German Language"
    # careful: only when followed by a known language name (not "with DevOps")
    t = re.sub(
        rf"\s+with\s+(?:the\s+)?(?:{LANG_EN})(?:\s+(?:or|and)\s+(?:{LANG_EN}))*"
        rf"(?:\s+(?:Languages?|Lang))?\b",
        "", t, flags=re.IGNORECASE,
    )
    # parenthetical: (with German)
    t = re.sub(
        rf"\s*\(\s*with\s+(?:{LANG_EN})\s*\)",
        "", t, flags=re.IGNORECASE,
    )

    # ── 4. gender markers ─────────────────────────────────────────────────────
    # (m/f), (m/f/d), (k/m), (k/m/n), (f/m/d), (m/w/x*), (M/K), etc.
    t = re.sub(r"\s*\(\s*[mfkwxdnMFKWXDN]+(?:/[mfkwxdnMFKWXDN]+)+\s*\*?\s*\)", "", t)
    # (She/He/They)
    t = re.sub(r"\s*\(\s*(?:She|He|They)(?:/(?:She|He|They))+\s*\)", "", t, flags=re.IGNORECASE)
    # standalone at end: M/F, K/M, M/K
    t = re.sub(r"\s+[MKFmkf]/[MKFmkf](?:/[a-zA-Z])?\s*$", "", t)
    # .NET FullStack Developer M/F at end
    t = re.sub(r"\s+[MF]/[MF]\b", "", t)

    # ── 5. feminatywy ─────────────────────────────────────────────────────────

    # 5a. /-suffix inline  →  "Specjalista /-tka"  →  "Specjalista"
    #     also  "(Starszy /-sza)"  →  "(Starszy)"
    t = re.sub(rf"\s*/\s*-\s*{FEM_INLINE_SUFFIXES}\b", "", t)

    # 5b. Word/suffix (no space)  →  "Specjalista/ka"  →  "Specjalista"
    #     guard: word must be ≥4 chars so we don't mangle .NET/C#
    t = re.sub(rf"\b([a-ząćęłńóśźżA-ZĄĆĘŁŃÓŚŹŻ]{{4,}})/(?:{FEM_INLINE_SUFFIXES})\b", r"\1", t)

    # 5c. (FemAdj) FemWord / (MascAdj) MascWord  →  MascAdj MascWord
    #     and reversed: (MascAdj) MascWord / (FemAdj) FemWord
    fem_adj_pat = r"(?:" + "|".join(re.escape(k) for k in FEM_TO_MASC_ADJ) + r")"
    masc_adj_pat = r"(?:" + "|".join(re.escape(k) for k in MASC_TO_FEM_ADJ) + r")"

    def _keep_masc_bracket(m: re.Match) -> str:
        adj1, word1, adj2, word2 = m.group(1), m.group(2), m.group(3), m.group(4)
        a1_lower = adj1.lower()
        a2_lower = adj2.lower()
        if a1_lower in FEM_TO_MASC_ADJ:
            # first bracket is feminine → keep second (masculine)
            masc_adj = adj2[0].upper() + adj2[1:]
            return f"({masc_adj}) {word2}"
        else:
            # first bracket is masculine → keep first
            masc_adj = adj1[0].upper() + adj1[1:]
            return f"({masc_adj}) {word1}"

    bracket_pair = (
        rf"\(({fem_adj_pat}|{masc_adj_pat})\)\s+(\w+)\s*/\s*"
        rf"\(({fem_adj_pat}|{masc_adj_pat})\)\s+(\w+)"
    )
    t = re.sub(bracket_pair, _keep_masc_bracket, t, flags=re.IGNORECASE)

    # 5d. Inline "MascWord / FemWord" or "FemWord / MascWord" pairs
    t = _replace_fem_pairs(t)

    # 5e. Full feminine clause after  |  or after long  /  split
    #     e.g. "Analityk danych – spec. | Analityczka danych – spec."
    #     heuristic: if second clause starts with a word that shares ≥5 chars
    #     with the first word of the first clause → drop the second clause
    def _drop_fem_clause(sep_char: str, text: str) -> str:
        if sep_char not in text:
            return text
        parts = text.split(sep_char, 1)
        if len(parts) != 2:
            return text
        first_word = re.search(r"\b([A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]{4,})", parts[0])
        second_word = re.search(r"^\s*([A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]{4,})", parts[1])
        if first_word and second_word:
            if _is_fem_pair(first_word.group(1), second_word.group(1)):
                return parts[0].rstrip(" –—-")
        return text

    t = _drop_fem_clause(" | ", t)
    # long slash split (both sides ≥ 20 chars → not a simple tech-stack slash)
    if " / " in t:
        left, _, right = t.partition(" / ")
        if len(left.strip()) >= 20 and len(right.strip()) >= 20:
            t = _drop_fem_clause(" / ", t)

    # ── 6. final cleanup ──────────────────────────────────────────────────────
    # Remove leading junk chars: #, *, quotes
    t = re.sub(r'^[\s#*"\'„""]+', "", t)
    # Dangling separators at start/end
    t = re.sub(r"^[\s\-–—,|/]+|[\s\-–—,|/]+$", "", t)
    # Collapse multiple spaces / separators
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\s*[-–—]\s*[-–—]+", " –", t)
    t = t.strip()

    return t


# ── DB update ─────────────────────────────────────────────────────────────────

def main() -> None:
    import time
    conn = sqlite3.connect(DB_PATH, isolation_level=None)   # autocommit off via manual BEGIN
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")     # safe here: worst case = redo on crash
    conn.execute("PRAGMA cache_size=-128000")  # 128 MB

    # ── ensure title_clean column exists ─────────────────────────────────────
    cols = [r[1] for r in conn.execute("PRAGMA table_info(job_ads)")]
    if "title_clean" not in cols:
        conn.execute("ALTER TABLE job_ads ADD COLUMN title_clean TEXT")
        print("Added column title_clean")

    # ── step 1: read all ids + titles into Python ─────────────────────────────
    t0 = time.time()
    print("Reading all rows…")
    rows: list[tuple[int, str]] = conn.execute(
        "SELECT id, title FROM job_ads WHERE title IS NOT NULL"
    ).fetchall()
    print(f"  {len(rows):,} rows in {time.time()-t0:.1f}s")

    # ── step 2: clean titles in Python ───────────────────────────────────────
    t1 = time.time()
    print("Cleaning titles…")
    cleaned: list[tuple[int, str]] = [
        (row_id, clean_title(title))
        for row_id, title in tqdm(rows, unit="row", mininterval=1)
    ]
    print(f"  done in {time.time()-t1:.1f}s")

    # ── step 3: bulk-load into temp table, then one UPDATE ────────────────────
    # Inserting into a fresh table is ~10× faster than 749K individual UPDATEs.
    t2 = time.time()
    print("Writing via temp table…")
    conn.execute("DROP TABLE IF EXISTS _tc_tmp")
    conn.execute("CREATE TABLE _tc_tmp (id INTEGER PRIMARY KEY, tc TEXT)")

    conn.execute("BEGIN")
    conn.executemany("INSERT INTO _tc_tmp VALUES (?,?)", cleaned)
    conn.execute(
        "UPDATE job_ads SET title_clean = "
        "(SELECT tc FROM _tc_tmp WHERE _tc_tmp.id = job_ads.id)"
    )
    conn.execute("DROP TABLE _tc_tmp")
    conn.execute("COMMIT")

    print(f"  written in {time.time()-t2:.1f}s")
    conn.close()
    print(f"\nDone. {len(cleaned):,} rows → title_clean  (total {time.time()-t0:.0f}s)")


# ── quick smoke-test ──────────────────────────────────────────────────────────

def _test() -> None:
    cases = [
        # salary
        ("Hydraulik - Monter instalacji grzewczych, sanitarnych, wentylacji z j. niemieckim 3950€",
         "Hydraulik - Monter instalacji grzewczych, sanitarnych, wentylacji"),
        ("Elektryk - 540 - 600€/ 40h",        "Elektryk"),
        ("Lakiernik Meblowy 4400-4840 €​ netto/mies", "Lakiernik Meblowy"),
        # language
        ("Specjalista ds. IT with German",     "Specjalista ds. IT"),
        ("Account Manager z j. włoskim",       "Account Manager"),
        ("Bookkeeper z językiem niemieckim i angielskim", "Bookkeeper"),
        ("(Junior) Process Officer with Danish Language", "(Junior) Process Officer"),
        # gender markers
        ("Operator wózka widłowego (M/K/X)",   "Operator wózka widłowego"),
        (".NET FullStack Developer M/F",        ".NET FullStack Developer"),
        ("AI Content Specialist (m/k)",         "AI Content Specialist"),
        # rynek
        ("Kierownik kontraktu - rynek ukraiński", "Kierownik kontraktu"),
        ("Account Manager na rynek niemiecki",  "Account Manager"),
        ("Business Development Manager – rynek DACH", "Business Development Manager"),
        # feminatywy inline
        ("Specjalista/ka ds. HR",               "Specjalista ds. HR"),
        ("Specjalista/-tka ds. PR",             "Specjalista ds. PR"),
        ("(Starszy /-sza) Specjalista /-tka ds. X", "(Starszy) Specjalista ds. X"),
        ("Starszy/a Księgowy/a",                "Starszy Księgowy"),
        # feminatywy pairs
        ("Recepcjonista / Recepcjonistka w hotelu", "Recepcjonista w hotelu"),
        ("Specjalista / Specjalistka ds. HR",   "Specjalista ds. HR"),
        ("Konsultant / Konsultantka ds. sprzedaży", "Konsultant ds. sprzedaży"),
        # complex
        ("(Starsza) Specjalistka/ (Starszy) Specjalista ds. Kadr i Płac",
         "(Starszy) Specjalista ds. Kadr i Płac"),
        ("Analityk / Analityczka - Specjalista / Specjalistka ds. Analiz",
         "Analityk - Specjalista ds. Analiz"),
    ]

    ok = fail = 0
    for raw, expected in cases:
        got = clean_title(raw)
        if got == expected:
            ok += 1
        else:
            fail += 1
            print(f"FAIL\n  in : {raw!r}\n  exp: {expected!r}\n  got: {got!r}")

    print(f"\nTests: {ok} passed, {fail} failed")


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        _test()
    else:
        main()
