"""Load garment attribute schema from attributes.json (or schema.json)."""

import json
from pathlib import Path

# Search order for the schema file
_CANDIDATES = (
    "attributes.json",
    "schema.json",
    "1_label/attributes.json",
    "1_label/schema.json",
)


def _find_schema_path(explicit=None):
    """Resolve path to attributes.json / schema.json relative to this file or cwd."""
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p
        raise FileNotFoundError(f"Schema not found: {explicit}")

    here = Path(__file__).resolve().parent
    roots = [here, here.parent, Path.cwd()]
    for root in roots:
        for name in _CANDIDATES:
            p = root / name
            if p.is_file():
                return p
    return None


def load_schema(path=None):
    """Load schema from attributes.json.
    Expected format:
      {
        "categories": [
          {
            "key": "length",
            "title": "Length / Shirt",
            "mode": "single",          # or "multi"
            "options": [
              {"name": "cropped", "aliases": ["crop"]},
              ...
            ]
          },
          ...
        ]
      }
    Used by annotate, train, merge, search, and test_images.
    """
    found = _find_schema_path(path)
    if found is None:
        raise FileNotFoundError(
            "attributes.json not found. Place it next to attributes.py "
            "(or pass path= to load_schema)."
        )
    data = json.loads(found.read_text(encoding="utf-8"))
    if "categories" not in data:
        raise ValueError(f"{found} must contain a top-level 'categories' list")
    return data


def head_spec(schema=None):
    """Convert schema categories into the compact dict expected by MultiHeadNet.
    Keys → {mode, options: [name, ...]}."""
    schema = schema or load_schema()
    return {
        cat["key"]: {
            "mode": cat["mode"],
            "options": [o["name"] for o in cat["options"]],
        }
        for cat in schema["categories"]
    }


def default_pattern(chosen, probs, thr=0.5):
    """If pattern is multi-label and nothing scores above thr, force 'no pattern'.
    Requires a solid/none option in the schema (e.g. 'no pattern', 'solid', 'plain').
    Called after pick() in test/inference so solid shirts are labelled correctly."""
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