"""Conservative SQL validation plus a separate, bounded execution process."""

import json
import os
import subprocess
import sys
import sqlglot
from sqlglot import exp

ALLOWED_FUNCTIONS = {
    "AND",
    "OR",
    "EXISTS",
    "SUM",
    "COUNT",
    "AVG",
    "MIN",
    "MAX",
    "ROUND",
    "ABS",
    "COALESCE",
    "NULLIF",
    "CAST",
    "TRY_CAST",
    "EXTRACT",
    "YEAR",
    "MONTH",
    "DAY",
    "DATE_TRUNC",
    "TIMESTAMP_TRUNC",
    "STRFTIME",
    "LOWER",
    "UPPER",
    "LENGTH",
    "SUBSTRING",
    "CONCAT",
    "GREATEST",
    "LEAST",
    "ROW_NUMBER",
    "RANK",
    "DENSE_RANK",
    "LAG",
    "LEAD",
    "IF",
    "CASE",
    "TIME_TO_STR",
    "TS_OR_DS_TO_DATE",
    "DATE_ADD",
    "DATE_SUB",
    "DATE_DIFF",
    "CURRENT_DATE",
    "CURRENT_TIMESTAMP",
}


def validate(sql, tables):
    if not isinstance(sql, str) or not sql.strip() or len(sql) > 12000:
        raise ValueError("Provide a nonempty query under 12,000 characters.")
    statements = sqlglot.parse(sql, read="duckdb")
    if len(statements) != 1 or not isinstance(
        statements[0], (exp.Select, exp.Union, exp.Intersect, exp.Except)
    ):
        raise ValueError("Only one read-only SELECT query is allowed.")
    tree = statements[0]
    if sum(1 for _ in tree.walk()) > 1500:
        raise ValueError("Query is too complex.")
    ctes = {c.alias_or_name.casefold() for c in tree.find_all(exp.CTE)}
    forbidden = {
        "Insert",
        "Update",
        "Delete",
        "Create",
        "Drop",
        "Alter",
        "Command",
        "Copy",
        "Attach",
        "Detach",
        "Pragma",
        "Transaction",
        "Into",
        "Lock",
        "LoadData",
    }
    for node in tree.walk():
        if type(node).__name__ in forbidden:
            raise ValueError("Only read-only analytical queries are allowed.")
        if isinstance(node, exp.Table):
            if not isinstance(node.this, exp.Identifier) or node.db or node.catalog:
                raise ValueError(
                    "External and qualified table sources are unavailable."
                )
            if node.name.casefold() not in {t.casefold() for t in tables} | ctes:
                raise ValueError("Query references a table outside this dataset.")
        if isinstance(node, exp.Func):
            name = (
                node.name.upper()
                if isinstance(node, exp.Anonymous)
                else node.sql_name().upper()
            )
            if name not in ALLOWED_FUNCTIONS:
                raise ValueError("Unsupported function: " + name)
    if not list(tree.find_all(exp.Table)):
        raise ValueError("Query must read the supplied dataset.")
    return tree.sql(dialect="duckdb")


def execute(sql, path, tables, timeout=8):
    safe = validate(sql, tables)
    # Do not pass model keys, platform credentials, or the parent's environment.
    env = {
        k: os.environ[k]
        for k in ("PATH", "SYSTEMROOT", "LANG", "LC_ALL")
        if k in os.environ
    }
    proc = subprocess.run(
        [sys.executable, "-m", "datatalker.query_worker"],
        input=json.dumps({"sql": safe, "path": str(path)}),
        text=True,
        capture_output=True,
        timeout=timeout,
        env=env,
    )
    if proc.returncode:
        raise ValueError(
            "Query exceeded its execution limits or could not be executed."
        )
    result = json.loads(proc.stdout)
    if "error" in result:
        raise ValueError(result["error"])
    result["sql"] = safe
    return result
