#!/usr/bin/env python3
"""
Build Voyage embeddings for Lightcast AI/ML skills and an ESCO subset,
then create a Lightcast -> ESCO bridge based on cosine similarity.

ESCO subset:
- all descendants of S5 ("working with computers")
- knowledge concepts that are ICT / digital related

Embeddings are created from weighted text built from label + description.
"""

from __future__ import annotations

import csv
import json
import os
import sqlite3
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import voyageai
from dotenv import load_dotenv

load_dotenv()


MODEL = "voyage-4-large"
DIMENSIONS = 1024
BATCH_SIZE = 128
TOP_K = 10
LABEL_WEIGHT = 3
DESCRIPTION_WEIGHT = 1
HADOOP_CODE = "0613.79.4"
BIG_DATA_CODE = "S5.5.2.5.1"
AI_PRIORITY_CODES = [
    "0619.12",       # principles of artificial intelligence
    "0619.12.1",     # machine learning
    "S5.1.0.29",     # utilise machine learning
    "0619.12.2",     # deep learning
    "0619.12.3",     # artificial neural networks
    "0619.14",       # computer vision
    "0619.5",        # image recognition
    "0231.4",        # natural language processing
    "0232.24",       # speech recognition
    "0232.6.1",      # computational linguistics
    "0613.12",       # machine translation
    "0612.94",       # information extraction
    "0688.2",        # data science
    "0688.4",        # cognitive computing
    "S5.5.2.5.1",    # analyse big data
    "S5.1.0.9",      # perform dimensionality reduction
    "S5.1.0.33",     # build recommender systems
    "0613.102",      # optical character recognition software
    "0612.47.0",     # data mining methods
    "0611.46",       # decision support systems
    "0714.62",       # robotics
    "T6.6.2.0.3",    # data ethics
]

REPO_ROOT = Path(__file__).resolve().parents[1]
LIGHTCAST_JSON = REPO_ROOT / "lightcast_aiml" / "lightcast_aiml_skills.json"
ESCO_DB = REPO_ROOT / "comprehensive_esco.db"
OUTPUT_DIR = REPO_ROOT / "lightcast_aiml" / "bridge_esco"


@dataclass
class SkillRecord:
    source: str
    group: str
    identifier: str
    label: str
    description: str
    level: int | None = None
    parent_code: str | None = None
    is_digital: int | None = None
    uri: str | None = None

    def weighted_text(self, label_weight: int, description_weight: int) -> str:
        label = normalize_text(self.label)
        description = normalize_text(self.description)

        parts: list[str] = []
        for _ in range(label_weight):
            parts.append(f"Label: {label}.")

        if description:
            for _ in range(description_weight):
                parts.append(f"Description: {description}.")

        return " ".join(parts)


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    value = " ".join(str(value).split())
    if value.lower() == "no description available.":
        return ""
    return value


def load_lightcast_records(path: Path) -> list[SkillRecord]:
    raw = json.loads(path.read_text())
    records: list[SkillRecord] = []

    for row in raw:
        records.append(
            SkillRecord(
                source="lightcast",
                group="aiml",
                identifier=row["id"],
                label=normalize_text(row.get("name", "")),
                description=normalize_text(row.get("description", "")),
            )
        )

    return records


def load_esco_records(db_path: Path) -> list[SkillRecord]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    s5_rows = cur.execute(
        """
        WITH RECURSIVE s5 AS (
            SELECT code, uri, title, description, level, parent_code, skill_type, is_digital
            FROM esco_concepts
            WHERE code = 'S5'

            UNION ALL

            SELECT e.code, e.uri, e.title, e.description, e.level, e.parent_code, e.skill_type, e.is_digital
            FROM esco_concepts e
            JOIN s5 ON e.parent_code = s5.code
        )
        SELECT DISTINCT
            code,
            uri,
            title,
            description,
            level,
            parent_code,
            is_digital
        FROM s5
        ORDER BY level, code
        """
    ).fetchall()

    knowledge_rows = cur.execute(
        """
        WITH RECURSIVE ict AS (
            SELECT code
            FROM esco_concepts
            WHERE code = '06'

            UNION ALL

            SELECT e.code
            FROM esco_concepts e
            JOIN ict ON e.parent_code = ict.code
        )
        SELECT DISTINCT
            code,
            uri,
            title,
            description,
            level,
            parent_code,
            is_digital
        FROM esco_concepts
        WHERE skill_type = 'knowledge'
          AND (
            code IN (SELECT code FROM ict)
            OR is_digital = 1
            OR lower(coalesce(title, '')) LIKE '%ict%'
            OR lower(coalesce(title, '')) LIKE '%digital%'
            OR lower(coalesce(title, '')) LIKE '%computer%'
            OR lower(coalesce(title, '')) LIKE '%information technology%'
            OR lower(coalesce(title, '')) LIKE '%information and communication%'
            OR lower(coalesce(description, '')) LIKE '%ict%'
            OR lower(coalesce(description, '')) LIKE '%digital%'
            OR lower(coalesce(description, '')) LIKE '%computer%'
            OR lower(coalesce(description, '')) LIKE '%information technology%'
            OR lower(coalesce(description, '')) LIKE '%information and communication%'
          )
        ORDER BY level, code
        """
    ).fetchall()

    extra_ai_rows = cur.execute(
        f"""
        SELECT DISTINCT
            code,
            uri,
            title,
            description,
            level,
            parent_code,
            is_digital
        FROM esco_concepts
        WHERE code IN ({",".join("?" for _ in AI_PRIORITY_CODES)})
        ORDER BY level, code
        """,
        AI_PRIORITY_CODES,
    ).fetchall()

    conn.close()

    records_by_key: dict[tuple[str, str], SkillRecord] = {}

    for row in s5_rows:
        key = ("esco", row["code"])
        records_by_key[key] = SkillRecord(
            source="esco",
            group="s5_skill",
            identifier=row["code"],
            label=normalize_text(row["title"]),
            description=normalize_text(row["description"]),
            level=row["level"],
            parent_code=row["parent_code"],
            is_digital=row["is_digital"],
            uri=row["uri"],
        )

    for row in knowledge_rows:
        key = ("esco", row["code"])
        if key in records_by_key:
            continue
        records_by_key[key] = SkillRecord(
            source="esco",
            group="digital_knowledge",
            identifier=row["code"],
            label=normalize_text(row["title"]),
            description=normalize_text(row["description"]),
            level=row["level"],
            parent_code=row["parent_code"],
            is_digital=row["is_digital"],
            uri=row["uri"],
        )

    for row in extra_ai_rows:
        key = ("esco", row["code"])
        if key in records_by_key:
            continue
        records_by_key[key] = SkillRecord(
            source="esco",
            group="digital_knowledge",
            identifier=row["code"],
            label=normalize_text(row["title"]),
            description=normalize_text(row["description"]),
            level=row["level"],
            parent_code=row["parent_code"],
            is_digital=row["is_digital"],
            uri=row["uri"],
        )

    return list(records_by_key.values())


def embed_records(
    client: voyageai.Client,
    records: list[SkillRecord],
    label_weight: int,
    description_weight: int,
) -> np.ndarray:
    weighted_texts = [record.weighted_text(label_weight, description_weight) for record in records]
    all_embeddings: list[list[float]] = []

    for start in range(0, len(weighted_texts), BATCH_SIZE):
        batch = weighted_texts[start : start + BATCH_SIZE]
        result = client.embed(
            texts=batch,
            model=MODEL,
            input_type="document",
            output_dimension=DIMENSIONS,
        )
        all_embeddings.extend(result.embeddings)

        if start + BATCH_SIZE < len(weighted_texts):
            time.sleep(0.1)

    embeddings = np.array(all_embeddings, dtype=np.float32)
    normalize_rows(embeddings)
    return embeddings


def normalize_rows(matrix: np.ndarray) -> None:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix /= norms


def write_records_jsonl(
    path: Path,
    records: Iterable[SkillRecord],
    label_weight: int,
    description_weight: int,
) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            payload = asdict(record)
            payload["weighted_text"] = record.weighted_text(label_weight, description_weight)
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_embeddings_npz(path: Path, records: list[SkillRecord], embeddings: np.ndarray) -> None:
    np.savez_compressed(
        path,
        embeddings=embeddings,
        identifiers=np.array([record.identifier for record in records], dtype=object),
        labels=np.array([record.label for record in records], dtype=object),
        groups=np.array([record.group for record in records], dtype=object),
    )


def compute_topk_matches(
    source_records: list[SkillRecord],
    source_embeddings: np.ndarray,
    target_records: list[SkillRecord],
    target_embeddings: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    k = min(top_k, len(target_records))
    sims = source_embeddings @ target_embeddings.T
    top_idx = np.argpartition(sims, -k, axis=1)[:, -k:]
    top_scores = np.take_along_axis(sims, top_idx, axis=1)

    order = np.argsort(top_scores, axis=1)[:, ::-1]
    top_idx = np.take_along_axis(top_idx, order, axis=1)
    top_scores = np.take_along_axis(top_scores, order, axis=1)
    return top_idx, top_scores


def apply_big_data_override(
    source_embeddings: np.ndarray,
    target_records: list[SkillRecord],
    target_embeddings: np.ndarray,
    top_idx: np.ndarray,
    top_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    target_index_by_code = {
        record.identifier: idx for idx, record in enumerate(target_records)
    }
    hadoop_idx = target_index_by_code.get(HADOOP_CODE)
    big_data_idx = target_index_by_code.get(BIG_DATA_CODE)

    if hadoop_idx is None or big_data_idx is None:
        return top_idx, top_scores

    all_scores = source_embeddings @ target_embeddings.T
    big_data_scores = source_embeddings @ target_embeddings[big_data_idx]

    for i in range(len(top_idx)):
        if top_idx[i, 0] != hadoop_idx:
            continue

        forced = [(big_data_idx, float(big_data_scores[i]))]
        ranked_rest = []

        for idx, score in zip(top_idx[i], top_scores[i]):
            if idx in (hadoop_idx, big_data_idx):
                continue
            ranked_rest.append((int(idx), float(score)))

        ranked_rest = sorted(ranked_rest, key=lambda item: item[1], reverse=True)

        seen = {big_data_idx}
        for idx, _ in ranked_rest:
            seen.add(idx)

        if 1 + len(ranked_rest) < top_idx.shape[1]:
            extra_order = np.argsort(all_scores[i])[::-1]
            for idx in extra_order:
                idx = int(idx)
                if idx in seen or idx == hadoop_idx:
                    continue
                ranked_rest.append((idx, float(all_scores[i, idx])))
                seen.add(idx)
                if 1 + len(ranked_rest) == top_idx.shape[1]:
                    break

        ranked = forced + ranked_rest
        ranked = ranked[: top_idx.shape[1]]

        top_idx[i] = np.array([idx for idx, _ in ranked], dtype=np.int32)
        top_scores[i] = np.array([score for _, score in ranked], dtype=np.float32)

    return top_idx, top_scores


def apply_ai_subset_override(
    source_embeddings: np.ndarray,
    target_records: list[SkillRecord],
    target_embeddings: np.ndarray,
    top_idx: np.ndarray,
    top_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    target_index_by_code = {
        record.identifier: idx for idx, record in enumerate(target_records)
    }
    ai_target_indices = [
        target_index_by_code[code]
        for code in AI_PRIORITY_CODES
        if code in target_index_by_code
    ]
    ai_target_index_set = set(ai_target_indices)

    if not ai_target_indices:
        return top_idx, top_scores

    ai_embeddings = target_embeddings[ai_target_indices]
    ai_scores = source_embeddings @ ai_embeddings.T
    ai_best_local = np.argmax(ai_scores, axis=1)
    ai_best_scores = ai_scores[np.arange(len(source_embeddings)), ai_best_local]
    ai_best_indices = np.array(
        [ai_target_indices[i] for i in ai_best_local],
        dtype=np.int32,
    )
    all_scores = source_embeddings @ target_embeddings.T

    for i in range(len(top_idx)):
        if top_idx[i, 0] in ai_target_index_set:
            continue

        forced = [(int(ai_best_indices[i]), float(ai_best_scores[i]))]
        ranked_rest = []

        for idx, score in zip(top_idx[i], top_scores[i]):
            idx = int(idx)
            if idx == forced[0][0]:
                continue
            ranked_rest.append((idx, float(score)))

        ranked_rest = sorted(ranked_rest, key=lambda item: item[1], reverse=True)

        seen = {forced[0][0]}
        for idx, _ in ranked_rest:
            seen.add(idx)

        if 1 + len(ranked_rest) < top_idx.shape[1]:
            extra_order = np.argsort(all_scores[i])[::-1]
            for idx in extra_order:
                idx = int(idx)
                if idx in seen:
                    continue
                ranked_rest.append((idx, float(all_scores[i, idx])))
                seen.add(idx)
                if 1 + len(ranked_rest) == top_idx.shape[1]:
                    break

        ranked = (forced + ranked_rest)[: top_idx.shape[1]]
        top_idx[i] = np.array([idx for idx, _ in ranked], dtype=np.int32)
        top_scores[i] = np.array([score for _, score in ranked], dtype=np.float32)

    return top_idx, top_scores


def write_bridge_csv(
    path: Path,
    source_records: list[SkillRecord],
    target_records: list[SkillRecord],
    top_idx: np.ndarray,
    top_scores: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "lightcast_id",
                "lightcast_label",
                "lightcast_description",
                "rank",
                "similarity",
                "esco_code",
                "esco_label",
                "esco_group",
                "esco_level",
                "esco_parent_code",
                "esco_is_digital",
                "esco_description",
                "esco_uri",
            ]
        )

        for i, source in enumerate(source_records):
            for rank in range(top_idx.shape[1]):
                target = target_records[top_idx[i, rank]]
                writer.writerow(
                    [
                        source.identifier,
                        source.label,
                        source.description,
                        rank + 1,
                        round(float(top_scores[i, rank]), 6),
                        target.identifier,
                        target.label,
                        target.group,
                        target.level,
                        target.parent_code,
                        target.is_digital,
                        target.description,
                        target.uri,
                    ]
                )


def write_best_match_csv(
    path: Path,
    source_records: list[SkillRecord],
    target_records: list[SkillRecord],
    top_idx: np.ndarray,
    top_scores: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "lightcast_id",
                "lightcast_label",
                "similarity",
                "esco_code",
                "esco_label",
                "esco_group",
                "esco_description",
            ]
        )

        for i, source in enumerate(source_records):
            target = target_records[top_idx[i, 0]]
            writer.writerow(
                [
                    source.identifier,
                    source.label,
                    round(float(top_scores[i, 0]), 6),
                    target.identifier,
                    target.label,
                    target.group,
                    target.description,
                ]
            )


def write_summary(
    path: Path,
    lightcast_records: list[SkillRecord],
    esco_records: list[SkillRecord],
    top_scores: np.ndarray,
) -> None:
    counts: dict[str, int] = {}
    for record in esco_records:
        counts[record.group] = counts.get(record.group, 0) + 1

    summary = {
        "model": MODEL,
        "dimensions": DIMENSIONS,
        "batch_size": BATCH_SIZE,
        "label_weight": LABEL_WEIGHT,
        "description_weight": DESCRIPTION_WEIGHT,
        "lightcast_count": len(lightcast_records),
        "esco_count": len(esco_records),
        "esco_group_counts": counts,
        "top1_similarity": {
            "min": round(float(np.min(top_scores[:, 0])), 6),
            "mean": round(float(np.mean(top_scores[:, 0])), 6),
            "median": round(float(np.median(top_scores[:, 0])), 6),
            "max": round(float(np.max(top_scores[:, 0])), 6),
        },
    }
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        raise RuntimeError("VOYAGE_API_KEY not set in environment or .env")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading records...")
    lightcast_records = load_lightcast_records(LIGHTCAST_JSON)
    esco_records = load_esco_records(ESCO_DB)
    print(f"  Lightcast AI/ML: {len(lightcast_records):,}")
    print(f"  ESCO target subset: {len(esco_records):,}")

    client = voyageai.Client(api_key=api_key)

    print("\nEmbedding Lightcast records...")
    lightcast_embeddings = embed_records(
        client,
        lightcast_records,
        LABEL_WEIGHT,
        DESCRIPTION_WEIGHT,
    )
    print(f"  shape: {lightcast_embeddings.shape}")

    print("\nEmbedding ESCO records...")
    esco_embeddings = embed_records(
        client,
        esco_records,
        LABEL_WEIGHT,
        DESCRIPTION_WEIGHT,
    )
    print(f"  shape: {esco_embeddings.shape}")

    print("\nComputing Lightcast -> ESCO bridge...")
    top_idx, top_scores = compute_topk_matches(
        lightcast_records,
        lightcast_embeddings,
        esco_records,
        esco_embeddings,
        TOP_K,
    )
    top_idx, top_scores = apply_big_data_override(
        lightcast_embeddings,
        esco_records,
        esco_embeddings,
        top_idx,
        top_scores,
    )
    top_idx, top_scores = apply_ai_subset_override(
        lightcast_embeddings,
        esco_records,
        esco_embeddings,
        top_idx,
        top_scores,
    )

    print("\nWriting outputs...")
    write_records_jsonl(
        OUTPUT_DIR / "lightcast_aiml_records.jsonl",
        lightcast_records,
        LABEL_WEIGHT,
        DESCRIPTION_WEIGHT,
    )
    write_records_jsonl(
        OUTPUT_DIR / "esco_s5_ict_records.jsonl",
        esco_records,
        LABEL_WEIGHT,
        DESCRIPTION_WEIGHT,
    )
    write_embeddings_npz(OUTPUT_DIR / "lightcast_aiml_embeddings.npz", lightcast_records, lightcast_embeddings)
    write_embeddings_npz(OUTPUT_DIR / "esco_s5_ict_embeddings.npz", esco_records, esco_embeddings)
    write_bridge_csv(
        OUTPUT_DIR / "lightcast_to_esco_bridge_top10.csv",
        lightcast_records,
        esco_records,
        top_idx,
        top_scores,
    )
    write_best_match_csv(
        OUTPUT_DIR / "lightcast_to_esco_bridge_top1.csv",
        lightcast_records,
        esco_records,
        top_idx,
        top_scores,
    )
    write_summary(OUTPUT_DIR / "bridge_summary.json", lightcast_records, esco_records, top_scores)

    print("\nTop 10 example matches:")
    for i in range(min(10, len(lightcast_records))):
        best = esco_records[top_idx[i, 0]]
        score = float(top_scores[i, 0])
        print(f"  {lightcast_records[i].label} -> {best.label} [{best.group}] ({score:.4f})")

    print(f"\nDone. Outputs saved in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
