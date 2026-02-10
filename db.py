"""
db_setup.py
-----------
Seeds the local DuckDB database with DS-1 and DS-2.

DS-1 rules enforced while seeding:
    - Training windows define when each salesperson is licensed.
    - Orders placed OUTSIDE a training window are deliberately invalid
      so the agent can find them.
    - Bonus-Pay tiers exist for 2012 and 2013 (matching the doc example).

DS-2 rules enforced while seeding:
    - Same agent appears under multiple Upline Managers / Agencies
      so the agent must learn to disambiguate.
    - Monthly columns Jan-2022 … Dec-2024 (36 months).
    - Values are plain floats (we store them normalised; the UI layer
      can format as currency later).
"""

import duckdb
import pandas as pd
from datetime import date

DB_PATH = "salesdata.db"


# ──────────────────────────────────────────────
# DS-1  —  four tables
# ──────────────────────────────────────────────

SALESPERSON = pd.DataFrame({
    "id":   [1,     2,     3,     4,     5],
    "name": ["Joe", "Bob", "Josh","Alice","Tom"],
    "age":  [61,    34,    45,    29,    52],
})

# Training windows – every salesperson gets at least one window.
# Salesperson 4 (Alice) has NO 2012 window on purpose → her 2012 order is invalid.
TRAINING = pd.DataFrame({
    "id":              [1, 2, 3, 4, 5, 6, 7, 8],
    "salesperson_id":  [1, 1, 2, 2, 3, 3, 4, 5],
    "start_date":      [
        date(2012, 1, 1),   # Joe  – 2012
        date(2013, 1, 1),   # Joe  – 2013-2015
        date(2012, 1, 1),   # Bob  – 2012
        date(2013, 3, 1),   # Bob  – 2013 (starts March → Jan/Feb orders invalid)
        date(2012, 6, 1),   # Josh – 2012 (starts June)
        date(2013, 1, 1),   # Josh – 2013
        date(2013, 1, 1),   # Alice – 2013 only (NO 2012 window)
        date(2012, 1, 1),   # Tom  – 2012-2014
    ],
    "end_date": [
        date(2012, 12, 31),
        date(2015, 12, 31),
        date(2012, 12, 31),
        date(2013, 12, 31),
        date(2012, 12, 31),
        date(2013, 12, 31),
        date(2013, 12, 31),
        date(2014, 12, 31),
    ],
})

# Orders – mix of valid and deliberately invalid ones.
#   Invalid orders:
#     id=5  – Alice (id=4) on 2012-03-15, but Alice has no 2012 training window
#     id=8  – Bob   (id=2) on 2013-01-15, but Bob's 2013 window starts 2013-03-01
ORDERS = pd.DataFrame({
    "id":               [1,  2,  3,  4,  5,    6,    7,    8,    9,   10],
    "order_date":       [
        date(2012, 8, 2),    # 1 – Bob   VALID
        date(2012, 1, 30),   # 2 – Josh  INVALID (Josh 2012 starts June)
        date(2012, 7, 10),   # 3 – Josh  VALID
        date(2013, 5, 22),   # 4 – Bob   VALID
        date(2012, 3, 15),   # 5 – Alice INVALID (no 2012 window)
        date(2013, 4, 10),   # 6 – Alice VALID
        date(2012, 9, 5),    # 7 – Joe   VALID
        date(2013, 1, 15),   # 8 – Bob   INVALID (2013 window starts March)
        date(2013, 6, 20),   # 9 – Josh  VALID
        date(2012, 11, 1),   # 10– Tom   VALID
    ],
    "salesperson_id":   [2,  3,  3,  2,  4,  4,  1,  2,  3,  5],
    "amount":           [540, 800, 1200, 1100, 900, 1600, 700, 950, 1500, 2200],
})

# Bonus-Pay tiers – the lookup table.
# Tiers are layered: you earn the HIGHEST tier you exceed.
#   2012: ≥1500 → $500,  ≥3000 → $1000
#   2013: ≥1500 → $1000, ≥3000 → $2000   (doc example: Josh 1650 in 2013 → 1000)
BONUS_PAY = pd.DataFrame({
    "id":    [1, 2, 3, 4],
    "year":  [2012, 2012, 2013, 2013],
    "tier":  [1500, 3000, 1500, 3000],
    "bonus": [500,  1000, 1000, 2000],
})


# ──────────────────────────────────────────────
# DS-2  —  wide-format agent commissions
# ──────────────────────────────────────────────
# 36 monthly columns: Jan-2022 … Dec-2024
# Avery Rodriguez appears TWICE (different uplines) so the agent
# must ask the user which one they mean.

import random
random.seed(42)

MONTHS = []
for y in range(2022, 2025):          # 2022, 2023, 2024
    for m in range(1, 13):
        MONTHS.append(f"{date(y, m, 1).strftime('%b')}-{y}")   # e.g. "Jan-2022"

AGENTS_RAW = [
    # (name,            agent_id,       upline_manager,  upline_id,        agency)
    ("Alex Garcia",     "Agent-ID-659", "Skyler Miller", "Upline-ID-482",  "Summit Group"),
    ("Avery Rodriguez","Agent-ID-712", "Dana Cole",     "Upline-ID-391",  "Summit Group"),
    ("Avery Rodriguez","Agent-ID-712", "Skyler Miller", "Upline-ID-482",  "Pinnacle Agency"),  # DUPLICATE agent, different upline+agency
    ("Blake Turner",    "Agent-ID-803", "Dana Cole",     "Upline-ID-391",  "Pinnacle Agency"),
    ("Casey Morgan",    "Agent-ID-447", "Skyler Miller", "Upline-ID-482",  "Apex Financial"),
    ("Jordan Lee",      "Agent-ID-521", "Dana Cole",     "Upline-ID-391",  "Apex Financial"),
    ("Morgan Hayes",    "Agent-ID-638", "Riley Stone",   "Upline-ID-507",  "Summit Group"),
    ("Riley Stone",     "Agent-ID-290", "Skyler Miller", "Upline-ID-482",  "Pinnacle Agency"),
    ("Sam Rivera",      "Agent-ID-174", "Riley Stone",   "Upline-ID-507",  "Apex Financial"),
    ("Taylor Brooks",   "Agent-ID-955", "Dana Cole",     "Upline-ID-391",  "Summit Group"),
]

rows = []
for name, aid, upline, uid, agency in AGENTS_RAW:
    row = {
        "agent_name":     name,
        "agent_id":       aid,
        "upline_manager": upline,
        "upline_id":      uid,
        "agency_name":    agency,
    }
    for m in MONTHS:
        row[m] = round(random.uniform(100, 10000), 2)   # positive commission amounts
    rows.append(row)

DS2 = pd.DataFrame(rows)


# ──────────────────────────────────────────────
# Write everything into DuckDB
# ──────────────────────────────────────────────

def seed():
    conn = duckdb.connect(DB_PATH)

    # DS-1
    conn.execute("DROP TABLE IF EXISTS salesperson")
    conn.execute("DROP TABLE IF EXISTS orders")
    conn.execute("DROP TABLE IF EXISTS training")
    conn.execute("DROP TABLE IF EXISTS bonus_pay")

    conn.execute("""
        CREATE TABLE salesperson (
            id   INTEGER PRIMARY KEY,
            name VARCHAR,
            age  INTEGER
        )
    """)
    conn.executemany(
        "INSERT INTO salesperson VALUES (?, ?, ?)",
        SALESPERSON.values.tolist()
    )

    conn.execute("""
        CREATE TABLE orders (
            id              INTEGER PRIMARY KEY,
            order_date      DATE,
            salesperson_id  INTEGER,
            amount          DOUBLE
        )
    """)
    conn.executemany(
        "INSERT INTO orders VALUES (?, ?, ?, ?)",
        ORDERS.values.tolist()
    )

    conn.execute("""
        CREATE TABLE training (
            id              INTEGER PRIMARY KEY,
            salesperson_id  INTEGER,
            start_date      DATE,
            end_date        DATE
        )
    """)
    conn.executemany(
        "INSERT INTO training VALUES (?, ?, ?, ?)",
        TRAINING.values.tolist()
    )

    conn.execute("""
        CREATE TABLE bonus_pay (
            id     INTEGER PRIMARY KEY,
            year   INTEGER,
            tier   INTEGER,
            bonus  INTEGER
        )
    """)
    conn.executemany(
        "INSERT INTO bonus_pay VALUES (?, ?, ?, ?)",
        BONUS_PAY.values.tolist()
    )

    # DS-2  – store in a single wide table exactly as the source sheet
    conn.execute("DROP TABLE IF EXISTS agent_commissions")

    month_cols = ", ".join([f'"{m}" DOUBLE' for m in MONTHS])
    conn.execute(f"""
        CREATE TABLE agent_commissions (
            agent_name     VARCHAR,
            agent_id       VARCHAR,
            upline_manager VARCHAR,
            upline_id      VARCHAR,
            agency_name    VARCHAR,
            {month_cols}
        )
    """)

    placeholders = ", ".join(["?"] * (5 + len(MONTHS)))
    conn.executemany(
        f"INSERT INTO agent_commissions VALUES ({placeholders})",
        DS2.values.tolist()
    )

    conn.close()
    print("[db_setup] ✓ salesdata.db seeded successfully.")
    print(f"           DS-1 tables : salesperson({len(SALESPERSON)} rows), "
          f"orders({len(ORDERS)} rows), training({len(TRAINING)} rows), "
          f"bonus_pay({len(BONUS_PAY)} rows)")
    print(f"           DS-2 table  : agent_commissions({len(DS2)} rows, "
          f"{len(MONTHS)} monthly columns)")


if __name__ == "__main__":
    seed()