#!/usr/bin/env python3
"""Fix similarity scores in job_kzis_matches table."""

import sqlite3
import numpy as np
from tqdm import tqdm

conn = sqlite3.connect("jobs_database.db")
cursor = conn.cursor()

print("Fixing similarity scores...")

# Get total count
cursor.execute("SELECT COUNT(*) FROM job_kzis_matches")
total = cursor.fetchone()[0]
print(f"Total rows: {total:,}")

# Check if already fixed
cursor.execute("SELECT similarity_score FROM job_kzis_matches LIMIT 1")
test_val = cursor.fetchone()[0]

if isinstance(test_val, (int, float)):
    print(f"Already fixed! Sample value: {test_val}")
    conn.close()
    exit(0)

print("Converting BLOB to REAL...")

# Create new table with correct schema
cursor.execute("""
    CREATE TABLE job_kzis_matches_new (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_title TEXT NOT NULL,
        rank INTEGER NOT NULL,
        kzis_occupation_id INTEGER NOT NULL,
        kzis_occupation_name TEXT NOT NULL,
        similarity_score REAL NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT unique_match UNIQUE (job_title, rank)
    )
""")

# Process in batches
batch_size = 10000
offset = 0

with tqdm(total=total) as pbar:
    while offset < total:
        # Fetch batch
        cursor.execute(f"""
            SELECT id, job_title, rank, kzis_occupation_id, kzis_occupation_name, similarity_score
            FROM job_kzis_matches
            LIMIT {batch_size} OFFSET {offset}
        """)
        
        rows = cursor.fetchall()
        if not rows:
            break
        
        # Convert BLOBs to float
        new_rows = []
        for row_id, title, rank, kzis_id, kzis_name, sim_blob in rows:
            try:
                # Try to decode as float32
                sim_value = np.frombuffer(sim_blob, dtype=np.float32)[0]
                new_rows.append((row_id, title, rank, kzis_id, kzis_name, float(sim_value)))
            except:
                # If that fails, try float64
                try:
                    sim_value = np.frombuffer(sim_blob, dtype=np.float64)[0]
                    new_rows.append((row_id, title, rank, kzis_id, kzis_name, float(sim_value)))
                except:
                    print(f"Failed to convert row {row_id}")
                    continue
        
        # Insert into new table
        cursor.executemany("""
            INSERT INTO job_kzis_matches_new (id, job_title, rank, kzis_occupation_id, kzis_occupation_name, similarity_score)
            VALUES (?, ?, ?, ?, ?, ?)
        """, new_rows)
        
        conn.commit()
        pbar.update(len(rows))
        offset += batch_size

print("\nReplacing table...")

# Drop old table and rename
cursor.execute("DROP TABLE job_kzis_matches")
cursor.execute("ALTER TABLE job_kzis_matches_new RENAME TO job_kzis_matches")

# Recreate indexes
print("Creating indexes...")
cursor.execute("CREATE INDEX idx_job_title_match ON job_kzis_matches(job_title)")
cursor.execute("CREATE INDEX idx_kzis_occ_id ON job_kzis_matches(kzis_occupation_id)")
cursor.execute("CREATE INDEX idx_similarity ON job_kzis_matches(similarity_score DESC)")

conn.commit()

# Verify
cursor.execute("SELECT job_title, similarity_score FROM job_kzis_matches LIMIT 5")
print("\nVerification:")
for title, score in cursor.fetchall():
    print(f"  {title}: {score:.4f} (type: {type(score).__name__})")

conn.close()
print("\n✓ Fixed!")
