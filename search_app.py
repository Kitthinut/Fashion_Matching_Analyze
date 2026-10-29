import argparse
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

from PIL import Image, ImageTk

from attributes import load_schema
from searchlib import (attr_group, describe_group, export_results, load_index, open_path,
                       parse_query, search)

THUMB, PAD = 200, 6


class SearchApp:
    def __init__(self, win, index_path):
        self.win, self.schema = win, load_schema()
        self.index = load_index(index_path)
        self.results, self.thumbs, self.last_query = [], [], ""
        win.title(f"Garment search  -  {len(self.index['images'])} images indexed")
        win.geometry("1180x800")

        top = tk.Frame(win, padx=8, pady=6)
        top.pack(fill="x")
        tk.Label(top, text="Search:").pack(side="left")
        self.q = tk.Entry(top, font=("Arial", 12))
        self.q.pack(side="left", fill="x", expand=True, padx=6)
        self.q.bind("<Return>", lambda e: self.do_search())
        self.q.focus_set()
        tk.Button(top, text="Search", command=self.do_search, bg="#cfe8cf").pack(side="left")
        tk.Button(top, text="Export results", command=self.export).pack(side="left", padx=4)

        flt = tk.Frame(win, padx=8)
        flt.pack(fill="x")
        self.filters = {}
        for cat in self.schema["categories"]:
            tk.Label(flt, text=cat["title"].split(" (")[0] + ":").pack(side="left")
            var = tk.StringVar(value="(any)")
            tk.OptionMenu(flt, var, "(any)", *[o["name"] for o in cat["options"]]).pack(side="left", padx=(0, 8))
            self.filters[cat["key"]] = var

        opt = tk.Frame(win, padx=8)
        opt.pack(fill="x")
        self.min_score = tk.Scale(opt, from_=0.05, to=0.95, resolution=0.05, orient="horizontal",
                                  length=220, label="Minimum match score per term")
        self.min_score.set(0.30)
        self.min_score.pack(side="left")
        tk.Label(opt, text="  Max results:").pack(side="left")
        self.maxn = tk.Spinbox(opt, from_=5, to=500, width=5)
        self.maxn.delete(0, "end"); self.maxn.insert(0, "40")
        self.maxn.pack(side="left")
        tk.Button(opt, text="Clear", command=self.clear).pack(side="left", padx=10)

        self.status = tk.Label(win, anchor="w", justify="left", padx=8, fg="#333")
        self.status.pack(fill="x")

        box = tk.Frame(win)
        box.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(box, bg="#f3f3f3", highlightthickness=0)
        sb = tk.Scrollbar(box, command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.inner = tk.Frame(self.canvas, bg="#f3f3f3")
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.bind_all("<MouseWheel>", lambda e: self.canvas.yview_scroll(-1 * (e.delta // 120), "units"))
        self.canvas.bind_all("<Button-4>", lambda e: self.canvas.yview_scroll(-3, "units"))
        self.canvas.bind_all("<Button-5>", lambda e: self.canvas.yview_scroll(3, "units"))
        self.say("Try:  navy v-neck stripe short sleeve cropped      (colours: red, blue, black, ...)")

    def say(self, t):
        self.status.config(text=t)

    def clear(self):
        self.q.delete(0, "end")
        for v in self.filters.values():
            v.set("(any)")

    def do_search(self):
        groups, ignored = parse_query(self.q.get(), self.schema)
        for key, var in self.filters.items():
            if var.get() != "(any)":
                groups = [g for g in groups if g.get("key") != key] + [attr_group(self.schema, key, var.get())]
        if not groups:
            self.say("Nothing understood - type attributes (e.g. 'red polo neck dot') or use the dropdowns.")
            return
        try:
            limit = int(self.maxn.get())
        except ValueError:
            limit = 40
        self.last_query = self.q.get().strip() or " ".join(describe_group(g) for g in groups)
        self.results = search(self.index, groups, self.min_score.get(), limit)
        msg = "Understood: " + "; ".join(describe_group(g) for g in groups)
        if ignored:
            msg += f"   (ignored: {', '.join(ignored)})"
        self.say(f"{msg}\n{len(self.results)} match(es) of {len(self.index['images'])} images")
        self.show()

    def show(self):
        for w in self.inner.winfo_children():
            w.destroy()
        self.thumbs = []
        cols = max(1, self.canvas.winfo_width() // (THUMB + 2 * PAD + 8))
        for i, r in enumerate(self.results):
            f = tk.Frame(self.inner, bd=1, relief="solid", bg="white")
            f.grid(row=i // cols, column=i % cols, padx=PAD, pady=PAD, sticky="n")
            try:
                img = Image.open(r["path"]).convert("RGB")
                img.thumbnail((THUMB, THUMB))
                ph = ImageTk.PhotoImage(img)
                self.thumbs.append(ph)
                lb = tk.Label(f, image=ph, bg="white", cursor="hand2")
                lb.pack()
                lb.bind("<Button-1>", lambda e, p=r["path"]: open_path(p))
            except Exception:
                tk.Label(f, text="(cannot load)", width=28, height=8).pack()
            tk.Label(f, text=f"{r['score']:.2f}  {Path(r['path']).name}", bg="white",
                     font=("Arial", 9, "bold"), wraplength=THUMB).pack()
            tk.Label(f, text=r["summary"], bg="white", font=("Arial", 8), fg="#555",
                     wraplength=THUMB, justify="left").pack()
        self.inner.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.canvas.yview_moveto(0)

    def export(self):
        if not self.results:
            messagebox.showinfo("Nothing to export", "Run a search first.")
            return
        page = export_results(self.results, self.last_query)
        self.say(f"Exported {len(self.results)} images to {page.parent}")
        open_path(page)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="index.json")
    a = ap.parse_args()
    if not Path(a.index).exists():
        raise SystemExit(f"{a.index} not found. Run: python build_index.py database")
    root = tk.Tk()
    SearchApp(root, a.index)
    root.mainloop()
