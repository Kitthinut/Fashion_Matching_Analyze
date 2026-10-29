import json
from pathlib import Path

SCHEMA_FILE = Path(__file__).parent / "attributes.json"


def load_schema(path=SCHEMA_FILE):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def head_spec(schema):
    """{'neck': {'mode': 'single', 'options': ['crew neck', ...]}, ...}"""
    return {c["key"]: {"mode": c["mode"], "options": [o["name"] for o in c["options"]]}
            for c in schema["categories"]}
