import collections
import unittest
from decimal import Decimal
from datatalker.app import catalog, chart_spec
from datatalker.sql import execute, validate


class RuntimeTests(unittest.TestCase):
    def query(self, sql):
        return execute(sql, catalog.path, catalog.tables)

    def test_catalog(self):
        self.assertEqual(len(catalog.tables), 5)
        self.assertEqual(sum(t["rows"] for t in catalog.tables.values()), 2832)
        self.assertEqual(len(catalog.relationships), 2)

    def test_reviewed_examples_against_python_aggregates(self):
        with catalog.connect() as db:
            reps = {
                i: (name, region)
                for i, name, region in db.execute("SELECT * FROM reps").fetchall()
            }
            orders = db.execute("SELECT * FROM orders").fetchall()
            periods = db.execute(
                "SELECT rep_id,start_date,end_date FROM eligibility_periods"
            ).fetchall()
        totals = collections.defaultdict(Decimal)
        months = collections.defaultdict(Decimal)
        people = collections.defaultdict(Decimal)
        eligible = collections.defaultdict(Decimal)
        for _, date, rep, amount, _ in orders:
            name, region = reps[rep]
            totals[region] += amount
            people[name] += amount
            if date.year == 2025:
                months[date.replace(day=1).isoformat()] += amount
            key = (
                "Eligible"
                if any(rep == r and start <= date <= end for r, start, end in periods)
                else "Ineligible"
            )
            eligible[key] += amount
        expected = {
            "revenue-region": sorted(
                [[k, float(v)] for k, v in totals.items()], key=lambda r: -r[1]
            ),
            "monthly-trend": sorted([[k, float(v)] for k, v in months.items()]),
            "top-reps": sorted(
                [[k, float(v)] for k, v in people.items()], key=lambda r: -r[1]
            )[:5],
            "eligible-revenue": sorted([[k, float(v)] for k, v in eligible.items()]),
        }
        for item in catalog.contract["examples"]:
            with self.subTest(example=item["id"]):
                self.assertEqual(self.query(item["sql"])["rows"], expected[item["id"]])

    def test_unsafe_sql_rejected(self):
        bad = [
            "SELECT * FROM orders; DELETE FROM orders",
            "DELETE FROM orders",
            "WITH gone AS (DELETE FROM orders RETURNING *) SELECT * FROM gone",
            "SELECT * FROM read_csv_auto('/etc/passwd')",
            "SELECT * FROM information_schema.tables",
            "SELECT * FROM duckdb_settings()",
            "SELECT getenv('GROQ_API_KEY') FROM orders LIMIT 1",
            "COPY orders TO '/tmp/out.csv'",
            "SELECT * INTO stolen FROM orders",
            "ATTACH '/tmp/other.db' AS other",
            "SELECT * FROM unknown_table",
            "SELECT * FROM range(1000000000)",
            "SELECT * FROM query('SELECT * FROM orders')",
            "SELECT * FROM main.orders",
        ]
        for sql in bad:
            with self.subTest(sql=sql), self.assertRaises(ValueError):
                validate(sql, catalog.tables)

    def test_output_cap(self):
        r = self.query("SELECT * FROM orders ORDER BY order_id")
        self.assertEqual(len(r["rows"]), 200)
        self.assertTrue(r["truncated"])

    def test_unknown_column(self):
        with self.assertRaises(ValueError):
            self.query("SELECT imaginary FROM orders")

    def test_entity_ambiguity(self):
        self.assertEqual(
            len(catalog.ambiguity("Show commissions for Alex Morgan")["choices"]), 2
        )
        self.assertIsNotNone(
            catalog.ambiguity("Total commissions for Alex Morgan by month")
        )
        self.assertIsNone(
            catalog.ambiguity(
                "Show commissions for Alex Morgan under Taylor Gray at Northstar"
            )
        )
        self.assertIsNone(
            catalog.ambiguity("Show combined commissions for Alex Morgan")
        )

    def test_chart_restrictions(self):
        r = self.query(
            "SELECT product, SUM(amount) AS revenue FROM orders GROUP BY product"
        )
        self.assertIsNotNone(
            chart_spec({"type": "bar", "x": "product", "y": "revenue"}, r)
        )
        self.assertIsNone(
            chart_spec({"type": "bar", "x": "product", "y": "product"}, r)
        )
        self.assertIsNone(
            chart_spec({"type": "code", "x": "product", "y": "revenue"}, r)
        )
        self.assertIsNone(
            chart_spec({"type": "bar", "x": "missing", "y": "revenue"}, r)
        )

    def test_cte_and_window_queries(self):
        r = self.query(
            "WITH revenue AS (SELECT rep_id, SUM(amount) AS amount FROM orders GROUP BY rep_id) SELECT rep_id,amount,RANK() OVER (ORDER BY amount DESC) AS rank FROM revenue ORDER BY rank"
        )
        self.assertEqual(len(r["rows"]), 12)

    def test_worker_timeout(self):
        import subprocess

        with self.assertRaises(subprocess.TimeoutExpired):
            execute("SELECT * FROM orders", catalog.path, catalog.tables, timeout=0.001)


if __name__ == "__main__":
    unittest.main()
