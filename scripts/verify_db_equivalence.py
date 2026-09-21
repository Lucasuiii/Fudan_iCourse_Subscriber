#!/usr/bin/env python3
"""Verify that two SQLite databases contain the same logical rows."""

from __future__ import annotations

import os
import sqlite3
import sys


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _tables(conn: sqlite3.Connection, schema: str) -> list[str]:
    return [
        row[0]
        for row in conn.execute(
            f"SELECT name FROM {schema}.sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        )
    ]


def _columns(
    conn: sqlite3.Connection, schema: str, table: str,
) -> list[str]:
    quoted = table.replace("'", "''")
    return [
        row[1]
        for row in conn.execute(f"PRAGMA {schema}.table_info('{quoted}')")
    ]


def verify_equivalent(left_path: str, right_path: str) -> None:
    for path in (left_path, right_path):
        if not os.path.isfile(path):
            raise ValueError("database file is missing")

    left_uri = f"file:{os.path.abspath(left_path)}?mode=ro"
    right_uri = f"file:{os.path.abspath(right_path)}?mode=ro"
    conn = sqlite3.connect(left_uri, uri=True)
    try:
        conn.execute("PRAGMA query_only = ON")
        conn.execute("ATTACH DATABASE ? AS rotated", (right_uri,))

        original_tables = _tables(conn, "main")
        rotated_tables = _tables(conn, "rotated")
        if original_tables != rotated_tables:
            raise ValueError("database table sets differ")

        for table in original_tables:
            if _columns(conn, "main", table) != _columns(
                conn, "rotated", table,
            ):
                raise ValueError("database column sets differ")

            quoted = _quote_identifier(table)
            left_count = conn.execute(
                f"SELECT COUNT(*) FROM main.{quoted}"
            ).fetchone()[0]
            right_count = conn.execute(
                f"SELECT COUNT(*) FROM rotated.{quoted}"
            ).fetchone()[0]
            if left_count != right_count:
                raise ValueError("database row counts differ")

            left_only = conn.execute(
                f"SELECT 1 FROM ("
                f"SELECT * FROM main.{quoted} "
                f"EXCEPT SELECT * FROM rotated.{quoted}"
                f") LIMIT 1"
            ).fetchone()
            right_only = conn.execute(
                f"SELECT 1 FROM ("
                f"SELECT * FROM rotated.{quoted} "
                f"EXCEPT SELECT * FROM main.{quoted}"
                f") LIMIT 1"
            ).fetchone()
            if left_only or right_only:
                raise ValueError("database row contents differ")
    finally:
        conn.close()


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} ORIGINAL ROTATED", file=sys.stderr)
        return 2
    try:
        verify_equivalent(sys.argv[1], sys.argv[2])
    except (OSError, sqlite3.Error, ValueError) as exc:
        print(
            f"database equivalence check failed: {type(exc).__name__}",
            file=sys.stderr,
        )
        return 1
    print("Database equivalence check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
