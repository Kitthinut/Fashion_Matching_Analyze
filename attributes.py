import json
from pathlib import Path

SCHEMA_FILE = Path(__file__).resolve().parent / "attributes.json"


def _resolve_schema_path(path=None):
    if path is None:
        return SCHEMA_FILE

    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = (Path(__file__).resolve().parent / candidate).resolve()
    return candidate


def load_schema(path=None):
    schema_path = _resolve_schema_path(path)
    return json.loads(schema_path.read_text(encoding="utf-8"))


def head_spec(schema=None):
    """Generates head metadata directly from schema."""
    if schema is None:
        schema = load_schema()
    return {
        cat["key"]: {
            "mode": cat["mode"],
            "options": [o["name"] for o in cat["options"]]
        }
        for cat in schema["categories"]
    }