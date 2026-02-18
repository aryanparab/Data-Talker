"""
db_setup.py
-----------
Seeds the local DuckDB database with DS-1 and DS-2 from the Excel file.

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
from datetime import datetime

DB_PATH = "salesdata.db"
EXCEL_FILE = "ENXT - AI Dev Case Study_Excel - January 2026 (1).xlsx"


# ──────────────────────────────────────────────
# DS-1  —  four tables from Excel
# ──────────────────────────────────────────────

def load_ds1_from_excel(filepath):
    """
    Load the four DS-1 tables from the Excel file.
    The tables are arranged side-by-side in the sheet.
    """
    # Read the entire sheet without headers
    df = pd.read_excel(filepath, sheet_name="DS-1", header=None)
    
    # SALESPERSON table: columns 0-2 (ID, Name, Age)
    # Row 0 is "Salesperson", Row 1 has headers, Data starts at row 2
    salesperson_data = df.iloc[2:8, 0:3].copy()  # 6 salespeople
    salesperson_data.columns = ['id', 'name', 'age']
    salesperson_data = salesperson_data.dropna(subset=['id'])
    salesperson_data['id'] = salesperson_data['id'].astype(int)
    salesperson_data['age'] = salesperson_data['age'].astype(int)
    
    # ORDERS table: columns 4-7 (ID, order_date, salesperson_id, Amount)
    # Find how many orders we have
    orders_data = df.iloc[2:, 4:8].copy()
    orders_data.columns = ['id', 'order_date', 'salesperson_id', 'amount']
    orders_data = orders_data.dropna(subset=['id'])
    orders_data['id'] = orders_data['id'].astype(int)
    orders_data['salesperson_id'] = orders_data['salesperson_id'].astype(int)
    orders_data['amount'] = orders_data['amount'].astype(float)
    # Convert order_date to date
    orders_data['order_date'] = pd.to_datetime(orders_data['order_date']).dt.date
    
    # TRAINING table: columns 9-12 (ID, salesperson_id, Start_date, End_date)
    training_data = df.iloc[2:, 9:13].copy()
    training_data.columns = ['id', 'salesperson_id', 'start_date', 'end_date']
    training_data = training_data.dropna(subset=['id'])
    training_data['id'] = training_data['id'].astype(int)
    training_data['salesperson_id'] = training_data['salesperson_id'].astype(int)
    training_data['start_date'] = pd.to_datetime(training_data['start_date']).dt.date
    training_data['end_date'] = pd.to_datetime(training_data['end_date']).dt.date
    
    # BONUS_PAY table: columns 14-17 (ID, Year, Tier, Bonus)
    bonus_data = df.iloc[2:, 14:18].copy()
    bonus_data.columns = ['id', 'year', 'tier', 'bonus']
    bonus_data = bonus_data.dropna(subset=['id'])
    bonus_data['id'] = bonus_data['id'].astype(int)
    bonus_data['year'] = bonus_data['year'].astype(int)
    bonus_data['tier'] = bonus_data['tier'].astype(int)
    bonus_data['bonus'] = bonus_data['bonus'].astype(int)
    
    return salesperson_data, orders_data, training_data, bonus_data


# ──────────────────────────────────────────────
# DS-2  —  wide-format agent commissions from Excel
# ──────────────────────────────────────────────

def load_ds2_from_excel(filepath):
    """
    Load DS-2 agent commissions table from Excel.
    The sheet has 36 monthly columns (Jan-2022 to Dec-2024).
    """
    # Read DS-2 sheet - it has proper headers
    df = pd.read_excel(filepath, sheet_name="DS-2")
    
    # Rename the datetime columns to string format (Jan-2022, Feb-2022, etc.)
    new_columns = {}
    for col in df.columns:
        if isinstance(col, datetime):
            new_columns[col] = col.strftime('%b-%Y')
    
    df = df.rename(columns=new_columns)
    
    # Clean column names to match our expected format
    df = df.rename(columns={
        'Agent Name': 'agent_name',
        'Agent ID': 'agent_id',
        'Upline Manager': 'upline_manager',
        'Upline ID': 'upline_id',
        'Agency Name': 'agency_name'
    })
    
    return df


# ──────────────────────────────────────────────
# Write everything into DuckDB
# ──────────────────────────────────────────────

def seed(excel_filepath=EXCEL_FILE):
    """
    Seed the DuckDB database with data from the Excel file.
    """
    print(f"[db_setup] Reading data from {excel_filepath}...")
    
    # Load DS-1 tables
    SALESPERSON, ORDERS, TRAINING, BONUS_PAY = load_ds1_from_excel(excel_filepath)
    
    # Load DS-2 table
    DS2 = load_ds2_from_excel(excel_filepath)
    
    # Get list of month columns for DS-2
    MONTHS = [col for col in DS2.columns if col not in 
              ['agent_name', 'agent_id', 'upline_manager', 'upline_id', 'agency_name']]
    
    print(f"[db_setup] Loaded DS-1: {len(SALESPERSON)} salespeople, {len(ORDERS)} orders, "
          f"{len(TRAINING)} training records, {len(BONUS_PAY)} bonus tiers")
    print(f"[db_setup] Loaded DS-2: {len(DS2)} agent records with {len(MONTHS)} monthly columns")
    
    # Connect to DuckDB and create tables
    conn = duckdb.connect(DB_PATH)

    # Clear all existing data before inserting new values
    print("[db_setup] Clearing existing database tables...")
    
    # DS-1 Tables
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
            id     INTEGER,
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