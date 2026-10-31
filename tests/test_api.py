import unittest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from datatalker.app import app, daily, requests


class APITests(unittest.TestCase):
    def setUp(self):
        requests.clear()
        daily.update(day="", calls=0)
        self.client = TestClient(app)

    def test_public_health_and_catalog(self):
        self.assertEqual(self.client.get("/api/health").json()["status"], "ok")
        c = self.client.get("/api/catalog").json()
        self.assertNotIn("sql", c["examples"][0])
        self.assertEqual(len(c["tables"]), 5)
        self.assertEqual(self.client.get("/").status_code, 200)

    def test_examples_never_call_model(self):
        with patch(
            "datatalker.app.complete",
            new=AsyncMock(side_effect=AssertionError("Must not call model")),
        ):
            r = self.client.post("/api/examples/revenue-region")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["model_calls"], 0)
        self.assertEqual(self.client.post("/api/examples/missing").status_code, 404)

    def test_request_bounds(self):
        for body in (
            {"question": "a"},
            {"question": "a" * 1201},
            {"question": "Hello", "history": [{"question": "Q"}] * 7},
            {"question": "Hi", "database_path": "/etc/passwd"},
        ):
            self.assertEqual(self.client.post("/api/query", json=body).status_code, 422)

    def test_ambiguous_name_avoids_model(self):
        with patch(
            "datatalker.app.complete",
            new=AsyncMock(side_effect=AssertionError("Must not guess context")),
        ):
            r = self.client.post(
                "/api/query",
                json={"question": "Total commissions for Alex Morgan by month"},
            )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(r.json()["choices"]), 2)

    def test_valid_plan_executes_with_one_model_call(self):
        import json

        plan = {
            "sql": "SELECT COUNT(*) AS orders FROM orders",
            "title": "Order count",
            "explanation": "Count all orders",
            "assumptions": [],
            "chart": {"type": "none"},
            "clarification": "",
        }
        mock = AsyncMock(return_value=(json.dumps(plan), {"total_tokens": 100}))
        with patch("datatalker.app.complete", new=mock):
            r = self.client.post("/api/query", json={"question": "How many orders?"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["rows"], [[2400]])
        self.assertEqual(r.json()["answer"], "orders: 2400")
        self.assertEqual(mock.await_count, 1)

    def test_query_repair_is_bounded(self):
        import json

        mock = AsyncMock(return_value=(json.dumps({"sql": "DELETE FROM orders"}), {}))
        with patch("datatalker.app.complete", new=mock):
            r = self.client.post("/api/query", json={"question": "Delete data"})
        self.assertEqual(r.status_code, 503)
        self.assertEqual(mock.await_count, 2)

    def test_grouped_followup_repaired_before_execution(self):
        import json

        scalar = {"sql": "SELECT SUM(amount) AS revenue FROM orders"}
        grouped = {
            "sql": "SELECT product,SUM(amount) AS revenue FROM orders GROUP BY product"
        }
        mock = AsyncMock(
            side_effect=[(json.dumps(scalar), {}), (json.dumps(grouped), {})]
        )
        with patch("datatalker.app.complete", new=mock):
            r = self.client.post(
                "/api/query",
                json={
                    "question": "Show it for all years",
                    "history": [
                        {"question": "Revenue by product", "sql": grouped["sql"]}
                    ],
                },
            )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(r.json()["columns"]), 2)
        self.assertEqual(mock.await_count, 2)

    def test_daily_allowance_prevents_provider_call(self):
        import datetime

        daily.update(
            day=datetime.datetime.now(datetime.timezone.utc).date().isoformat(),
            calls=100000,
        )
        with patch("datatalker.app.complete", new=AsyncMock()) as mock:
            r = self.client.post("/api/query", json={"question": "How many orders?"})
        self.assertEqual(r.status_code, 503)
        mock.assert_not_awaited()

    def test_hourly_limit(self):
        for _ in range(15):
            self.client.post(
                "/api/query", json={"question": "Show commissions for Alex Morgan"}
            )
        self.assertEqual(
            self.client.post(
                "/api/query", json={"question": "Show commissions for Alex Morgan"}
            ).status_code,
            429,
        )


if __name__ == "__main__":
    unittest.main()
