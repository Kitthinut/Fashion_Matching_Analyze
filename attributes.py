"""Load garment attribute schema from attributes.json (project root)."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

_CANDIDATES = (
    ROOT / "attributes.json",
    ROOT / "schema.json",
    ROOT / "1_label" / "attributes.json",
)


def _find_schema_path(explicit=None):
    """Resolve path to attributes.json relative to project root or cwd."""
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            p = ROOT / explicit
        if p.is_file():
            return p
        raise FileNotFoundError(f"Schema not found: {explicit}")
    for p in _CANDIDATES:
        if p.is_file():
            return p
    cwd = Path.cwd() / "attributes.json"
    if cwd.is_file():
        return cwd
    return None


def load_schema(path=None):
    """Load schema from attributes.json at project root."""
    found = _find_schema_path(path)
    if found is None:
        raise FileNotFoundError(
            f"attributes.json not found. Place it in the project root ({ROOT})."
        )
    data = json.loads(found.read_text(encoding="utf-8"))
    if "categories" not in data:
        raise ValueError(f"{found} must contain a top-level 'categories' list")
    return data


def head_spec(schema=None):
    """Convert schema categories into the compact dict expected by MultiHeadNet."""
    schema = schema or load_schema()
    return {
        cat["key"]: {
            "mode": cat["mode"],
            "options": [o["name"] for o in cat["options"]],
        }
        for cat in schema["categories"]
    }


def default_pattern(chosen, probs, thr=0.5):
    """If nothing patterned scores above thr, force a solid/no-pattern option."""
    if "pattern" not in probs:
        return chosen
    solid_names = {"no pattern", "solid", "plain", "none"}
    hits = [
        o for o, v in probs["pattern"].items()
        if v >= thr and o.lower() not in solid_names
    ]
    if hits:
        chosen["pattern"] = hits
        return chosen
    solid_opts = [o for o in probs["pattern"] if o.lower() in solid_names]
    if solid_opts:
        best = max(solid_opts, key=lambda o: probs["pattern"][o])
        chosen["pattern"] = [best]
    else:
        chosen["pattern"] = []
    return chosen