"""Reproducible, entirely synthetic public demo. Never reads the case-study workbook."""

import calendar
import json
import random
from datetime import date, timedelta
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "demo.duckdb"
DEFAULT_CONTRACT = ROOT / "data" / "dataset.json"


def generate(path=DEFAULT_DB, contract_path=DEFAULT_CONTRACT):
    path, contract_path = Path(path), Path(contract_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and contract_path.exists():
        return
    # Build separately so an interrupted generation never leaves a partial dataset.
    final_path = path
    path = path.with_suffix(".building.duckdb")
    if path.exists():
        path.unlink()
    rng = random.Random(42)
    names = [
        "Alex Morgan",
        "Casey Rivera",
        "Drew Park",
        "Emery Stone",
        "Finley Brooks",
        "Harper Lane",
        "Jordan Reed",
        "Kai Bennett",
        "Logan Ellis",
        "Morgan Hayes",
        "Quinn Blake",
        "Riley Chen",
    ]
    regions = ["North", "South", "East", "West"]
    conn = duckdb.connect(str(path))
    conn.execute(
        "CREATE TABLE reps(rep_id INTEGER PRIMARY KEY, name VARCHAR, region VARCHAR)"
    )
    conn.execute(
        "CREATE TABLE orders(order_id INTEGER PRIMARY KEY, order_date DATE, rep_id INTEGER REFERENCES reps(rep_id), amount DECIMAL(12,2), product VARCHAR)"
    )
    conn.execute(
        "CREATE TABLE eligibility_periods(period_id INTEGER PRIMARY KEY, rep_id INTEGER REFERENCES reps(rep_id), start_date DATE, end_date DATE)"
    )
    conn.execute(
        "CREATE TABLE bonus_tiers(year INTEGER, revenue_threshold DECIMAL(12,2), bonus DECIMAL(12,2))"
    )
    conn.execute(
        "CREATE TABLE commissions(agent_id INTEGER, agent_name VARCHAR, manager VARCHAR, agency VARCHAR, month DATE, commission DECIMAL(12,2))"
    )
    conn.executemany(
        "INSERT INTO reps VALUES (?,?,?)",
        [(i + 1, n, regions[i % 4]) for i, n in enumerate(names)],
    )
    products = ["Analytics", "Automation", "Platform"]
    orders = [
        (
            i + 1,
            date(2024, 1, 1) + timedelta(days=rng.randrange(731)),
            rng.randrange(1, 13),
            round(rng.uniform(150, 4500), 2),
            rng.choice(products),
        )
        for i in range(2400)
    ]
    conn.executemany("INSERT INTO orders VALUES (?,?,?,?,?)", orders)
    periods = []
    for rep in range(1, 13):
        for year in (2024, 2025):
            periods.append(
                (
                    len(periods) + 1,
                    rep,
                    date(year, 1 + (rep % 3), 1),
                    date(
                        year,
                        10 + (rep % 3),
                        calendar.monthrange(year, 10 + (rep % 3))[1],
                    ),
                )
            )
    conn.executemany("INSERT INTO eligibility_periods VALUES (?,?,?,?)", periods)
    conn.executemany(
        "INSERT INTO bonus_tiers VALUES (?,?,?)",
        [
            (y, t, b)
            for y in (2024, 2025)
            for t, b in [
                (0, 0),
                (100000, 1000),
                (200000, 2500),
                (300000, 5000),
                (400000, 8000),
                (500000, 12000),
            ]
        ],
    )
    rows = []
    for i, name in enumerate(names):
        contexts = [("Taylor Gray", "Northstar")]
        if i % 3 == 0:
            contexts.append(("Avery Wells", "Waypoint"))
        for manager, agency in contexts:
            for year in (2024, 2025):
                for month in range(1, 13):
                    rows.append(
                        (
                            i + 1,
                            name,
                            manager,
                            agency,
                            date(year, month, 1),
                            round(rng.uniform(300, 6500), 2),
                        )
                    )
    conn.executemany("INSERT INTO commissions VALUES (?,?,?,?,?,?)", rows)
    conn.close()
    contract = {
        "id": "northstar-demo-v1",
        "title": "Northstar Sales",
        "description": "A fictional sales and commissions workspace spanning 2024–2025.",
        "provenance": "Entirely synthetic data, generated deterministically with seed 42. No real people, compensation records or case-study workbook contents.",
        "definitions": [
            {
                "term": "Revenue / total sales",
                "definition": "SUM(orders.amount). Include all orders unless the question explicitly asks for eligible or ineligible sales.",
            },
            {
                "term": "Eligible sales",
                "definition": "Orders whose order_date falls inclusively within ANY eligibility_periods interval for the same rep_id. Use EXISTS to avoid double counting overlapping intervals.",
            },
            {
                "term": "Ineligible sales",
                "definition": "Orders with NO matching eligibility period for that rep_id and order_date. Use NOT EXISTS.",
            },
            {
                "term": "Bonus",
                "definition": "For each rep and calendar year, calculate eligible revenue, then select the greatest bonus_tiers.bonus where revenue_threshold is <= that revenue and the year matches. bonus_tiers is a threshold lookup, not an order-level join.",
            },
            {
                "term": "Commissions",
                "definition": "Each row is one agent, manager, agency and calendar month. An agent may have multiple manager/agency contexts. Aggregate all contexts only when explicitly requested; otherwise resolve ambiguous names.",
            },
            {
                "term": "Time scope",
                "definition": "When no period is specified, use the full available dataset and state that assumption. Calendar years and inclusive date boundaries apply. Do not silently substitute eligibility for total sales.",
            },
        ],
        "entities": [
            {
                "table": "commissions",
                "name_column": "agent_name",
                "context_columns": ["manager", "agency"],
            }
        ],
        "examples": [
            {
                "id": "revenue-region",
                "title": "Revenue by region",
                "question": "Compare total revenue by region across all available years.",
                "sql": "SELECT r.region, ROUND(SUM(o.amount),2) AS revenue FROM orders o JOIN reps r ON o.rep_id=r.rep_id GROUP BY r.region ORDER BY revenue DESC",
                "chart": {"type": "bar", "x": "region", "y": "revenue"},
            },
            {
                "id": "monthly-trend",
                "title": "Monthly sales trend",
                "question": "Show monthly total revenue in 2025.",
                "sql": "SELECT DATE_TRUNC('month',order_date) AS month, ROUND(SUM(amount),2) AS revenue FROM orders WHERE order_date >= DATE '2025-01-01' AND order_date < DATE '2026-01-01' GROUP BY 1 ORDER BY 1",
                "chart": {"type": "line", "x": "month", "y": "revenue"},
            },
            {
                "id": "top-reps",
                "title": "Top salespeople",
                "question": "Who are the five highest-revenue salespeople across all years?",
                "sql": "SELECT r.name, ROUND(SUM(o.amount),2) AS revenue FROM orders o JOIN reps r ON o.rep_id=r.rep_id GROUP BY r.name ORDER BY revenue DESC LIMIT 5",
                "chart": {"type": "bar", "x": "name", "y": "revenue"},
            },
            {
                "id": "eligible-revenue",
                "title": "Eligibility check",
                "question": "Compare eligible and ineligible revenue across all available years.",
                "sql": "SELECT CASE WHEN EXISTS(SELECT 1 FROM eligibility_periods e WHERE e.rep_id=o.rep_id AND o.order_date BETWEEN e.start_date AND e.end_date) THEN 'Eligible' ELSE 'Ineligible' END AS eligibility, ROUND(SUM(o.amount),2) AS revenue FROM orders o GROUP BY 1 ORDER BY 1",
                "chart": {"type": "bar", "x": "eligibility", "y": "revenue"},
            },
        ],
    }
    path.replace(final_path)
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(json.dumps(contract, indent=2) + "\n")


if __name__ == "__main__":
    generate()
