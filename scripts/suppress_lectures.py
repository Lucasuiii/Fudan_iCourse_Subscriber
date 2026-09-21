#!/usr/bin/env python3
"""Erase selected lecture content while keeping persistent tombstones."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.database import Database


def main() -> int:
    if len(sys.argv) != 4:
        print(f"usage: {sys.argv[0]} DATABASE COURSE_ID SUB_IDS", file=sys.stderr)
        return 2
    db_path, course_id, raw_sub_ids = sys.argv[1:]
    sub_ids = [value.strip() for value in raw_sub_ids.split(",") if value.strip()]
    if not course_id or not sub_ids:
        print("error: course and lecture selection are required", file=sys.stderr)
        return 2
    db = Database(db_path)
    try:
        changed = db.suppress_lectures(course_id, sub_ids)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        db.conn.close()
    print(f"Suppressed {changed} lecture(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
