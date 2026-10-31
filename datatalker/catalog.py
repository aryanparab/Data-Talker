import json
import re
from pathlib import Path
import duckdb


def ident(value):
    return '"' + value.replace('"', '""') + '"'


class Catalog:
    def __init__(self, path, contract_path):
        self.path = Path(path).resolve()
        self.contract = json.loads(Path(contract_path).read_text())
        self.tables = {}
        with self.connect() as conn:
            for (table,) in conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='main' ORDER BY table_name"
            ).fetchall():
                cols = conn.execute("DESCRIBE " + ident(table)).fetchall()
                self.tables[table] = {
                    "columns": {r[0]: str(r[1]) for r in cols},
                    "rows": conn.execute(
                        "SELECT COUNT(*) FROM " + ident(table)
                    ).fetchone()[0],
                }
            self.relationships = [
                {
                    "table": r[0],
                    "columns": list(r[1]),
                    "referenced_table": r[2],
                    "referenced_columns": list(r[3]),
                }
                for r in conn.execute(
                    "SELECT table_name,constraint_column_names,referenced_table,referenced_column_names FROM duckdb_constraints() WHERE constraint_type='FOREIGN KEY'"
                ).fetchall()
            ]
        self.entities = []
        with self.connect() as conn:
            for spec in self.contract.get("entities", []):
                table = spec["table"]
                col = spec["name_column"]
                contexts = spec["context_columns"]
                if table not in self.tables or any(
                    c not in self.tables[table]["columns"] for c in [col, *contexts]
                ):
                    raise ValueError(
                        "Entity configuration does not match the live schema"
                    )
                values = [
                    r[0]
                    for r in conn.execute(
                        f"SELECT DISTINCT {ident(col)} FROM {ident(table)} ORDER BY 1 LIMIT 100"
                    ).fetchall()
                ]
                self.entities.append((spec, values))

    def connect(self):
        return duckdb.connect(
            str(self.path),
            read_only=True,
            config={
                "enable_external_access": "false",
                "autoload_known_extensions": "false",
                "autoinstall_known_extensions": "false",
                "threads": "1",
            },
        )

    def public(self):
        return {
            "title": self.contract["title"],
            "description": self.contract["description"],
            "provenance": self.contract["provenance"],
            "tables": self.tables,
            "relationships": self.relationships,
            "definitions": self.contract.get("definitions", []),
            "examples": [
                {k: v for k, v in e.items() if k != "sql"}
                for e in self.contract.get("examples", [])
            ],
        }

    def prompt_data(self):
        return {
            "tables": self.tables,
            "relationships": self.relationships,
            "definitions": self.contract.get("definitions", []),
            "description": self.contract["description"],
        }

    def ambiguity(self, question):
        aggregate = bool(
            re.search(
                r"\bcombined\b|\b(?:all|across|compare|by|per)\s+(?:(?:all|the)\s+)?(?:contexts?|agencies|managers?)\b",
                question,
                re.I,
            )
        )
        if aggregate:
            return None
        matches = []
        for spec, values in self.entities:
            for value in values:
                if isinstance(value, str) and re.search(
                    r"(?<!\w)" + re.escape(value) + r"(?!\w)", question, re.I
                ):
                    matches.append((spec, value))
        if len(matches) != 1:
            return None
        spec, value = matches[0]
        cols = spec["context_columns"]
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT DISTINCT {','.join(ident(c) for c in cols)} FROM {ident(spec['table'])} WHERE {ident(spec['name_column'])}=? ORDER BY ALL LIMIT 11",
                [value],
            ).fetchall()
        filters = []
        for row in rows:
            context = dict(zip(cols, row))
            if any(
                str(v).casefold() in question.casefold() for v in row if v is not None
            ):
                filters.append(context)
        if len(filters) == 1:
            return None
        if len(rows) < 2:
            return None
        return {
            "table": spec["table"],
            "name_column": spec["name_column"],
            "value": value,
            "choices": [dict(zip(cols, row)) for row in rows[:10]],
            "has_more": len(rows) > 10,
        }
