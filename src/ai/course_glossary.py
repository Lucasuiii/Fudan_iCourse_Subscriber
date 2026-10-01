"""Small course-specific terminology hints, not transcription evidence."""
import json
import unicodedata
from pathlib import Path

GLOSSARY_PATH = Path(__file__).resolve().parents[2] / "prompts" / "course_glossary.json"
MAX_TERMS = 30
MAX_TERM_CHARS = 40


def _normalize(title: str) -> str:
    return "".join(unicodedata.normalize("NFKC", title).split()).casefold()


def course_terms(title: str, path: Path | None = None) -> list[str]:
    """Exact normalized title/alias matching; invalid config never blocks ASR."""
    if not title:
        return []
    try:
        entries = json.loads((path or GLOSSARY_PATH).read_text(encoding="utf-8"))
        if not isinstance(entries, list):
            return []
        for entry in entries[:100]:
            if not isinstance(entry, dict):
                continue
            aliases = entry.get("courses", [])
            if not isinstance(aliases, list) or not any(
                isinstance(alias, str) and _normalize(alias) == _normalize(title)
                for alias in aliases
            ):
                continue
            values = entry.get("terms", [])
            if not isinstance(values, list):
                return []
            terms = []
            for value in values:
                if not isinstance(value, str):
                    continue
                value = value.strip()
                if (not value or len(value) > MAX_TERM_CHARS
                        or any(unicodedata.category(c).startswith("C") for c in value)
                        or any(c in value for c in "<>") or value in terms):
                    continue
                terms.append(value)
                if len(terms) == MAX_TERMS:
                    break
            return terms
    except (OSError, ValueError, TypeError):
        pass
    return []


def terminology_reference(title: str) -> str:
    terms = course_terms(title)
    if not terms:
        return ""
    return ("\n\n<terminology_reference>\n"
            + json.dumps(terms, ensure_ascii=False)
            + "\n</terminology_reference>")
