"""No model access; one query, one read-only connection, then exit."""

import datetime
import decimal
import json
import math
import sys


def main():
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (6, 6))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    request = json.loads(sys.stdin.read(20000))
    import duckdb

    conn = duckdb.connect(
        request["path"],
        read_only=True,
        config={
            "enable_external_access": "false",
            "autoload_known_extensions": "false",
            "autoinstall_known_extensions": "false",
            "threads": "1",
            "memory_limit": "128MB",
            "max_temp_directory_size": "0B",
        },
    )
    conn.execute("SET lock_configuration=true")
    cursor = conn.execute(request["sql"])
    columns = [{"name": d[0], "type": str(d[1])} for d in cursor.description]
    raw = cursor.fetchmany(201)

    def clean(v):
        if isinstance(v, (datetime.date, datetime.datetime)):
            return v.isoformat()
        if isinstance(v, decimal.Decimal):
            return float(v)
        if isinstance(v, float) and not math.isfinite(v):
            return None
        if isinstance(v, str):
            return v[:4000]
        return v

    rows = []
    size = 0
    truncated = len(raw) > 200
    for row in raw[:200]:
        out = [clean(v) for v in row]
        size += len(json.dumps(out))
        if size > 200000:
            truncated = True
            break
        rows.append(out)
    print(json.dumps({"columns": columns, "rows": rows, "truncated": truncated}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"error": str(exc)[:1000]}))
