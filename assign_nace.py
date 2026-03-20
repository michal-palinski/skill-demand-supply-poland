#!/usr/bin/env python3
"""
Assign NACE codes per job offer (job_ads.id):
  1. title_clean → ESCO label  (contextual, from title_esco_3approaches.db)
  2. ESCO code   → NACE candidates  (crosswalk)
  3. 1 candidate → direct
  4. multiple    → disambiguate using THIS offer's title+resp embedding
                   vs NACE PL title embeddings (no averaging across offers)

Output: job_nace.db  table job_nace
  job_id, title_clean, esco_label, nace_code, nace_title
"""

import os, pickle, sqlite3
import numpy as np
import pandas as pd
import faiss
import voyageai
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

IDX_DIR   = "faiss_indexes"
CROSSWALK = "esco_nace_crosswalk.xlsx"
MATCH_DB  = "title_esco_3approaches.db"
OUT_DB    = "job_nace.db"

NACE_INDEX_PATH = os.path.join(IDX_DIR, "nace_pl_titles.index")
NACE_META_PATH  = os.path.join(IDX_DIR, "nace_pl_titles_metadata.pkl")

MODEL = "voyage-4"
DIMS  = 2048
BATCH = 128
CHUNK = 50_000   # process resp index in chunks to avoid OOM


def embed_texts(vo, texts):
    all_emb = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i : i + BATCH]
        r = vo.embed(batch, model=MODEL, input_type="document", output_dimension=DIMS)
        all_emb.extend(r.embeddings)
    arr = np.array(all_emb, dtype=np.float32)
    faiss.normalize_L2(arr)
    return arr


def main():
    vo = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

    # ── 1. ESCO label → code ────────────────────────────────────
    print("[1] ESCO metadata …")
    with open(os.path.join(IDX_DIR, "esco_occupations_contextual_2048_metadata.pkl"), "rb") as f:
        em = pickle.load(f)
    label2code = dict(zip(em["labels"], em["codes"]))

    # ── 2. Crosswalk ────────────────────────────────────────────
    print("[2] NACE crosswalk …")
    xw = pd.read_excel(CROSSWALK)
    nace_uniq = xw.drop_duplicates("NACE code")[["NACE code", "PL title"]].dropna()
    nace_codes_list  = nace_uniq["NACE code"].tolist()
    nace_titles_list = nace_uniq["PL title"].tolist()
    nace_code2title  = dict(zip(nace_codes_list, nace_titles_list))
    nace_code2pos    = {c: i for i, c in enumerate(nace_codes_list)}

    ec2nc = {}
    for _, r in xw.iterrows():
        ec, nc = r["ESCO Code"], r["NACE code"]
        if pd.notna(ec) and pd.notna(nc):
            ec2nc.setdefault(ec, set()).add(nc)
    ec2nc = {k: sorted(v) for k, v in ec2nc.items()}

    print(f"  {len(nace_codes_list)} unique NACE, {len(ec2nc)} ESCO→NACE mappings")

    # ── 3. NACE embeddings ──────────────────────────────────────
    print("[3] NACE embeddings …")
    if os.path.exists(NACE_INDEX_PATH):
        nace_idx = faiss.read_index(NACE_INDEX_PATH)
        with open(NACE_META_PATH, "rb") as f:
            pickle.load(f)
        print(f"  Loaded ({nace_idx.ntotal} vectors)")
    else:
        print(f"  Embedding {len(nace_titles_list)} NACE PL titles …")
        arr = embed_texts(vo, nace_titles_list)
        nace_idx = faiss.IndexFlatIP(DIMS)
        nace_idx.add(arr)
        faiss.write_index(nace_idx, NACE_INDEX_PATH)
        with open(NACE_META_PATH, "wb") as f:
            pickle.dump({"codes": nace_codes_list, "titles": nace_titles_list}, f)
        print(f"  Created ({nace_idx.ntotal} vectors)")

    # pre-build full NACE matrix (706×2048) for fast dot-product
    nace_mat = np.vstack([nace_idx.reconstruct(i) for i in range(nace_idx.ntotal)])

    # ── 4. title_clean → ESCO contextual ───────────────────────
    print("[4] Loading title→ESCO matches …")
    conn = sqlite3.connect(MATCH_DB)
    # prefer contextual; fall back to label when contextual is NULL
    rows_m = conn.execute(
        "SELECT title_clean, "
        "COALESCE(esco_contextual, esco_label) AS esco, "
        "esco_contextual "
        "FROM matches"
    ).fetchall()
    conn.close()

    t2e = {r[0]: r[1] for r in rows_m}
    n_fallback = sum(1 for r in rows_m if r[2] is None and r[1] is not None)
    print(f"  {len(t2e):,} titles  ({n_fallback} used label fallback)")

    # title_clean → NACE candidates
    t2nc = {}
    for title, esco_lbl in t2e.items():
        ec    = label2code.get(esco_lbl)
        cands = ec2nc.get(ec, []) if ec else []
        t2nc[title] = cands

    # ── 5. Load job resp index ──────────────────────────────────
    print("[5] Loading title+resp index …")
    resp_idx = faiss.read_index(os.path.join(IDX_DIR, "job_title_clean_resp.index"))
    with open(os.path.join(IDX_DIR, "job_title_clean_resp_metadata.pkl"), "rb") as f:
        rm = pickle.load(f)

    job_ids    = rm["ids"]     # list of job_ads.id, position = faiss row
    job_labels = rm["labels"]  # list of title_clean, same positions
    n_total    = len(job_ids)

    # ── 6. Process in chunks ────────────────────────────────────
    print(f"[6] Assigning NACE for {n_total:,} offers …")

    n_direct = n_disambig = n_none = 0
    rows = []

    for start in tqdm(range(0, n_total, CHUNK), desc="chunks", unit="chunk"):
        end = min(start + CHUNK, n_total)

        vecs   = resp_idx.reconstruct_n(start, end - start)  # (chunk, 2048)
        chunk_ids    = job_ids[start:end]
        chunk_labels = job_labels[start:end]

        for i in range(end - start):
            jid    = chunk_ids[i]
            title  = chunk_labels[i]
            cands  = t2nc.get(title, [])
            esco_lbl = t2e.get(title)

            if not cands:
                nc = None
                n_none += 1
            elif len(cands) == 1:
                nc = cands[0]
                n_direct += 1
            else:
                # disambiguate with this specific offer's embedding
                cand_pos  = [nace_code2pos[nc] for nc in cands if nc in nace_code2pos]
                if not cand_pos:
                    nc = cands[0]
                else:
                    job_vec   = vecs[i]                          # (2048,)
                    cand_vecs = nace_mat[cand_pos]               # (k, 2048)
                    best_idx  = int((job_vec @ cand_vecs.T).argmax())
                    nc = cands[best_idx]
                n_disambig += 1

            nt = nace_code2title.get(nc) if nc else None
            rows.append((jid, title, esco_lbl, nc, nt))

    # ── 7. Save ─────────────────────────────────────────────────
    print("[7] Saving …")
    out = sqlite3.connect(OUT_DB, isolation_level=None)
    out.execute("PRAGMA journal_mode=WAL")
    out.execute("PRAGMA synchronous=OFF")
    out.execute("DROP TABLE IF EXISTS job_nace")
    out.execute("""
        CREATE TABLE job_nace (
            job_id       INTEGER PRIMARY KEY,
            title_clean  TEXT,
            esco_label   TEXT,
            nace_code    TEXT,
            nace_title   TEXT
        )
    """)
    out.execute("CREATE INDEX IF NOT EXISTS idx_jn_title ON job_nace(title_clean)")
    out.execute("CREATE INDEX IF NOT EXISTS idx_jn_nace  ON job_nace(nace_code)")
    out.execute("BEGIN")
    out.executemany("INSERT INTO job_nace VALUES (?,?,?,?,?)", rows)
    out.execute("COMMIT")
    out.execute("VACUUM")
    out.close()

    sz = os.path.getsize(OUT_DB) / 1_048_576
    print(f"\nSaved: {OUT_DB} ({sz:.1f} MB)")
    print(f"  Total offers:  {n_total:,}")
    print(f"  Direct:        {n_direct:,}  ({n_direct/n_total*100:.1f}%)")
    print(f"  Disambiguated: {n_disambig:,}  ({n_disambig/n_total*100:.1f}%)")
    print(f"  No NACE:       {n_none:,}")

    # sample disambiguated
    out = sqlite3.connect(OUT_DB)
    print("\nSample (disambiguated offers):")
    rows_s = out.execute("""
        SELECT job_id, title_clean, esco_label, nace_code, nace_title
        FROM job_nace
        WHERE nace_code IS NOT NULL
        ORDER BY RANDOM() LIMIT 5
    """).fetchall()
    for r in rows_s:
        print(f"  [{r[0]}] {r[1]}")
        print(f"    ESCO: {r[2]}")
        print(f"    NACE: {r[3]} — {r[4]}")
    out.close()


if __name__ == "__main__":
    main()
