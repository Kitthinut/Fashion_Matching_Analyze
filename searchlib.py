"""Text-query search over index.json (used by search.py and search_app.py)."""
import html
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from analysis import COLOR_TABLE

COLOR_ALIASES = {"grey": "gray", "navy blue": "navy", "dark blue": "navy", "sky blue": "light blue",
                 "violet": "purple", "tan": "beige", "off white": "cream", "crimson": "red",
                 "lime": "green", "dark grey": "charcoal", "dark gray": "charcoal"}
STOP = {"a", "an", "the", "and", "with", "in", "of", "shirt", "shirts", "top", "tops", "tee", "t",
        "clothes", "clothing", "color", "colour", "find", "show", "me", "sleeve", "sleeves",
        "neck", "pattern", "looking", "for", "or", "length", "type"}


def norm(s):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower().replace("-", " ")).split())


def build_vocab(schema):
    v = {}
    for cat in schema["categories"]:
        for opt in cat["options"]:
            for ph in [opt["name"]] + opt.get("aliases", []):
                v[norm(ph)] = ("attr", cat["key"], opt["name"])
    for c in COLOR_TABLE:
        v[norm(c)] = ("color", None, c)
    for a, c in COLOR_ALIASES.items():
        v[norm(a)] = ("color", None, c)
    return v


def parse_query(text, schema):
    """-> (groups, ignored_words).  Same single-choice category = OR, multi = AND, colours = AND."""
    vocab = build_vocab(schema)
    modes = {c["key"]: c["mode"] for c in schema["categories"]}
    words, i, hits, ignored = norm(text).split(), 0, [], []
    maxlen = max(len(p.split()) for p in vocab)
    while i < len(words):
        found = False
        for n in range(min(maxlen, len(words) - i), 0, -1):
            ph = " ".join(words[i:i + n])
            if ph not in vocab and ph.endswith("s") and ph[:-1] in vocab:
                ph = ph[:-1]
            if ph in vocab:
                hits.append(vocab[ph])
                i += n
                found = True
                break
        if not found:
            if words[i] not in STOP:
                ignored.append(words[i])
            i += 1
    groups, by_key = [], {}
    for kind, key, name in hits:
        if kind == "color":
            if not any(g["type"] == "color" and g["name"] == name for g in groups):
                groups.append({"type": "color", "name": name})
        else:
            g = by_key.get(key)
            if g is None:
                g = by_key[key] = {"type": "attr", "key": key, "mode": modes[key], "options": []}
                groups.append(g)
            if name not in g["options"]:
                g["options"].append(name)
    return groups, ignored


def attr_group(schema, key, option):
    mode = next(c["mode"] for c in schema["categories"] if c["key"] == key)
    return {"type": "attr", "key": key, "mode": mode, "options": [option]}


def describe_group(g):
    if g["type"] == "color":
        return f"color = {g['name']}"
    joiner = " / " if g["mode"] == "single" else " + "
    return f"{g['key']} = {joiner.join(g['options'])}"


def score_group(region, g):
    if g["type"] == "color":
        cols = ((region.get("analysis") or {}).get("colors")) or []
        frac = next((c["fraction"] for c in cols if c["name"] == g["name"]), 0.0)
        return min(1.0, frac / 0.4)                 # 40% of the garment = full score
    probs = region["attributes"].get(g["key"], {})
    vals = [probs.get(o, 0.0) for o in g["options"]]
    return max(vals) if g["mode"] == "single" else min(vals)


def describe_region(region):
    parts = []
    for key, probs in region["attributes"].items():
        top = sorted(probs.items(), key=lambda kv: -kv[1])
        chosen = [n for n, p in top if p >= 0.5] or [top[0][0]]
        parts.append("+".join(chosen))
    cols = ((region.get("analysis") or {}).get("colors")) or []
    cs = ", ".join(f"{c['name']} {c['fraction']:.0%}" for c in cols[:2])
    return " | ".join(parts) + (f"\ncolor: {cs}" if cs else "")


def load_index(path="index.json"):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def search(index, groups, min_score=0.3, limit=50):
    root, res = Path(index["image_root"]), []
    for rel, info in index["images"].items():
        best = None
        for r in info["regions"]:
            sc = [score_group(r, g) for g in groups]
            ov = sum(sc) / len(sc)
            if min(sc) >= min_score and (best is None or ov > best[0]):
                best = (ov, sc, r)
        if best:
            res.append({"path": str(root / rel), "rel": rel, "score": best[0],
                        "term_scores": best[1], "summary": describe_region(best[2])})
    res.sort(key=lambda r: -r["score"])
    return res[:limit]


def open_path(p):
    p = str(p)
    if sys.platform.startswith("win"):
        os.startfile(p)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", p])
    else:
        subprocess.Popen(["xdg-open", p])


def export_results(results, query, out_root="results"):
    """Copy matches into results/<query>/ and write an index.html gallery."""
    folder = Path(out_root) / (re.sub(r"[^a-z0-9]+", "_", query.lower()).strip("_")[:50] or "query")
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    cards, rows = [], []
    for i, r in enumerate(results, 1):
        name = f"{i:02d}_{Path(r['path']).name}"
        shutil.copy2(r["path"], folder / name)
        rows.append({"file": name, "score": round(r["score"], 3), "original": r["path"]})
        cards.append(f"<div class=c><img src='{html.escape(name)}'><b>{r['score']:.2f}</b> "
                     f"{html.escape(Path(r['path']).name)}<br><small>"
                     f"{html.escape(r['summary']).replace(chr(10), '<br>')}</small></div>")
    (folder / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    page = (f"<meta charset=utf-8><title>{html.escape(query)}</title><style>body{{font-family:sans-serif;"
            f"background:#fafafa}}.c{{display:inline-block;width:230px;margin:8px;vertical-align:top;"
            f"background:#fff;border:1px solid #ddd;padding:6px}}img{{width:100%;height:230px;"
            f"object-fit:contain;background:#eee}}</style><h2>“{html.escape(query)}” – {len(rows)} match(es)</h2>"
            + "".join(cards))
    (folder / "index.html").write_text(page, encoding="utf-8")
    return folder / "index.html"
