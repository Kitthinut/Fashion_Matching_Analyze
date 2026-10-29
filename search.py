"""Command-line search.   python search.py "navy v-neck striped short sleeve cropped" --export"""
import argparse

from attributes import load_schema
from searchlib import (describe_group, export_results, load_index, open_path, parse_query, search)

ap = argparse.ArgumentParser()
ap.add_argument("query")
ap.add_argument("--index", default="index.json")
ap.add_argument("--min-score", type=float, default=0.3)
ap.add_argument("--top", type=int, default=20)
ap.add_argument("--export", action="store_true", help="copy matches to results/ and open an HTML gallery")
a = ap.parse_args()

groups, ignored = parse_query(a.query, load_schema())
if not groups:
    raise SystemExit("Nothing understood. Try e.g. 'navy v-neck stripe short sleeve cropped'")
print("Understood:", "; ".join(describe_group(g) for g in groups))
if ignored:
    print("Ignored words:", ", ".join(ignored))
res = search(load_index(a.index), groups, a.min_score, a.top)
for r in res:
    print(f"{r['score']:.2f}  {r['rel']}   [{r['summary'].splitlines()[0]}]")
print(f"{len(res)} match(es)")
if a.export and res:
    page = export_results(res, a.query)
    print("gallery:", page)
    open_path(page)
