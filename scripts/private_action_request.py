"""Validate a private frontend request and emit shell-safe environment values.

The course and lecture IDs live inside a GitHub Actions secret.  Only an opaque
request UUID is sent as workflow_dispatch metadata.  Capturing this program's
stdout and evaluating it in the workflow keeps selected IDs out of the run
metadata and prevents a stale secret from being used by a later request.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys


_REQUEST_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_ID_LIST_RE = re.compile(r"^[A-Za-z0-9._:-]+(?:,[A-Za-z0-9._:-]+)*$")


def _validated_id_list(value: object, *, required: bool) -> str:
    text = str(value or "").strip()
    if not text:
        if required:
            raise ValueError("required selection is empty")
        return ""
    if len(text) > 10000 or not _ID_LIST_RE.fullmatch(text):
        raise ValueError("selection has an invalid format")
    return text


def parse_request(kind: str, request_id: str, envelope: str) -> dict[str, str]:
    if not _REQUEST_ID_RE.fullmatch(request_id or ""):
        raise ValueError("request ID has an invalid format")
    try:
        payload = json.loads(envelope)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("private request is missing or invalid") from exc
    if not isinstance(payload, dict) or payload.get("request_id") != request_id:
        raise ValueError("private request does not match this workflow run")

    course_ids = _validated_id_list(payload.get("course_ids"), required=True)
    sub_ids = _validated_id_list(payload.get("sub_ids"), required=False)
    if kind == "single-run":
        return {"COURSE_IDS": course_ids}
    if kind == "export":
        return {
            "EXPORT_COURSE_ID": course_ids,
            "EXPORT_SUB_IDS": sub_ids,
        }
    if kind == "delete":
        return {
            "COURSE_IDS_INPUT": course_ids,
            "SUB_IDS_INPUT": sub_ids,
        }
    raise ValueError("unknown private request kind")


def main() -> int:
    if len(sys.argv) != 3:
        print("error: expected request kind and request ID", file=sys.stderr)
        return 2
    try:
        values = parse_request(
            sys.argv[1], sys.argv[2], os.environ.get("PRIVATE_ACTION_REQUEST", "")
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    for name, value in values.items():
        print(f"export {name}={shlex.quote(value)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
