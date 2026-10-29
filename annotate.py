import argparse
import json
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

import numpy as np
from PIL import Image, ImageTk

from analysis import analyze_crop
from attributes import load_schema

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
NONE = "__none__"


class Annotator:
    def __init__(self, win, folder, ann_path):
        self.win, self.folder, self.ann_path = win, Path(folder), Path(ann_path)
        self.schema = load_schema()
        self.files = sorted(p.relative_to(self.folder).as_posix() for p in self.folder.rglob("*")
                            if p.suffix.lower() in EXTS)
        if not self.files:
            messagebox.showerror("No images", f"No images found in {self.folder}")
            raise SystemExit
        self.data = (json.loads(self.ann_path.read_text(encoding="utf-8"))
                     if self.ann_path.exists() else {"images": {}})
        self.data["image_root"] = str(self.folder.resolve())
        self.idx, self.start, self.cur_box, self.box_id = 0, None, None, None
        self.vars = {}
        self.build_ui()
        win.after(150, self.show_image)

    # ------------------------------------------------------------------ UI
    def build_ui(self):
        w = self.win
        w.title("Garment annotator")
        w.geometry("1380x880")
        self.canvas = tk.Canvas(w, bg="#222", cursor="crosshair")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        r = tk.Frame(w, width=470, padx=8, pady=6)
        r.pack(side="right", fill="y")
        self.title = tk.Label(r, anchor="w", font=("Arial", 10, "bold"))
        self.title.pack(fill="x")

        for cat in self.schema["categories"]:
            lf = tk.LabelFrame(r, text=cat["title"], padx=4)
            lf.pack(fill="x", pady=2)
            key, names = cat["key"], [o["name"] for o in cat["options"]]
            if cat["mode"] == "single":
                var = tk.StringVar(value="")
                self.vars[key] = var
                items = [(n, n) for n in names] + [("(skip)", "")]
                for j, (txt, val) in enumerate(items):
                    tk.Radiobutton(lf, text=txt, variable=var, value=val, anchor="w").grid(
                        row=j // 3, column=j % 3, sticky="w", padx=4)
            else:
                d = {n: tk.BooleanVar() for n in names + [NONE]}
                self.vars[key] = d
                for j, n in enumerate(names + [NONE]):
                    tk.Checkbutton(lf, text="(none of these)" if n == NONE else n, variable=d[n],
                                   anchor="w").grid(row=j // 3, column=j % 3, sticky="w", padx=4)

        row = tk.Frame(r)
        row.pack(fill="x", pady=3)
        tk.Button(row, text="Add region (Enter)", command=self.add_region, bg="#cfe8cf").pack(side="left", expand=True, fill="x")
        tk.Button(row, text="Update selected", command=self.update_region).pack(side="left", expand=True, fill="x")
        tk.Button(row, text="Delete", command=self.delete_region).pack(side="left", expand=True, fill="x")
        self.keep = tk.BooleanVar(value=False)
        tk.Checkbutton(r, text="Keep ticks after adding (faster for similar items)", variable=self.keep, anchor="w").pack(fill="x")

        tk.Label(r, text="Regions in this image (click to view/edit):", anchor="w").pack(fill="x")
        self.rl = tk.Listbox(r, height=4, exportselection=False)
        self.rl.pack(fill="x")
        self.rl.bind("<<ListboxSelect>>", self.on_region_select)

        tk.Button(r, text="Preview colors of current crop", command=self.analyze_current).pack(fill="x", pady=2)

        self.info = tk.Text(r, height=8, wrap="word", font=("Consolas", 9))
        self.info.pack(fill="both", expand=True)

        row = tk.Frame(r)
        row.pack(fill="x", pady=4)
        tk.Button(row, text="< Prev", command=lambda: self.goto(self.idx - 1)).pack(side="left", expand=True, fill="x")
        tk.Button(row, text="Save", command=self.save).pack(side="left", expand=True, fill="x")
        tk.Button(row, text="Next >", command=lambda: self.goto(self.idx + 1)).pack(side="left", expand=True, fill="x")

        w.bind("<Return>", lambda e: self.add_region())
        w.bind("<Left>", lambda e: self.goto(self.idx - 1))
        w.bind("<Right>", lambda e: self.goto(self.idx + 1))
        w.bind("<Control-s>", lambda e: self.save())
        w.protocol("WM_DELETE_WINDOW", self.on_close)

    # ----------------------------------------------------- attribute <-> UI
    def get_attributes(self):
        attrs = {}
        for cat in self.schema["categories"]:
            k, v = cat["key"], self.vars[cat["key"]]
            if cat["mode"] == "single":
                if v.get():
                    attrs[k] = v.get()
            else:
                ticked = [n for n, b in v.items() if n != NONE and b.get()]
                if ticked:
                    attrs[k] = ticked
                elif v[NONE].get():
                    attrs[k] = []          # explicitly "no pattern"
        return attrs

    def set_attributes(self, attrs):
        for cat in self.schema["categories"]:
            k, v = cat["key"], self.vars[cat["key"]]
            if cat["mode"] == "single":
                v.set(attrs.get(k, ""))
            else:
                for n, b in v.items():
                    b.set(n in attrs.get(k, []))
                v[NONE].set(k in attrs and attrs[k] == [])

    @staticmethod
    def short(attrs):
        return " | ".join("+".join(v) if isinstance(v, list) else v for v in attrs.values() if v) or "-"

    # ---------------------------------------------------------------- image
    def entry(self):
        return self.data["images"].setdefault(self.files[self.idx], {"regions": []})

    def show_image(self):
        self.img = Image.open(self.folder / self.files[self.idx]).convert("RGB")
        cw, ch = max(self.canvas.winfo_width(), 400), max(self.canvas.winfo_height(), 400)
        self.scale = min(cw / self.img.width, ch / self.img.height)
        self.photo = ImageTk.PhotoImage(self.img.resize(
            (int(self.img.width * self.scale), int(self.img.height * self.scale))))
        self.cur_box = self.box_id = None
        self.refresh_regions()
        self.redraw()

    def redraw(self, highlight=None):
        c = self.canvas
        c.delete("all")
        c.create_image(0, 0, anchor="nw", image=self.photo)
        for i, r in enumerate(self.entry()["regions"]):
            x0, y0, x1, y1 = [v * self.scale for v in r["box"]]
            col = "red" if i == highlight else "#00e676"
            c.create_rectangle(x0, y0, x1, y1, outline=col, width=2)
            c.create_text(x0 + 4, y0 + 4, anchor="nw", fill=col, font=("Arial", 10, "bold"),
                          text=f"{i+1}: {self.short(r['attributes'])}")
        self.box_id = None

    def refresh_regions(self):
        e = self.entry()
        self.title.config(text=f"[{self.idx+1}/{len(self.files)}] {self.files[self.idx]}  ({len(e['regions'])} regions)")
        self.rl.delete(0, "end")
        for i, r in enumerate(e["regions"]):
            self.rl.insert("end", f"{i+1}: {self.short(r['attributes'])}")

    # ---------------------------------------------------------------- mouse
    def on_press(self, e):
        self.start, self.cur_box = (e.x, e.y), None

    def on_drag(self, e):
        if not self.start:
            return
        if self.box_id:
            self.canvas.delete(self.box_id)
        self.box_id = self.canvas.create_rectangle(*self.start, e.x, e.y, outline="yellow", width=2, dash=(4, 2))

    def on_release(self, e):
        if not self.start:
            return
        x0, x1 = sorted((self.start[0], e.x))
        y0, y1 = sorted((self.start[1], e.y))
        self.start = None
        if x1 - x0 < 8 or y1 - y0 < 8:
            self.cur_box = None
            self.redraw()
            return
        W, H, s = self.img.width, self.img.height, self.scale
        cl = lambda v, m: max(0, min(int(v), m))
        self.cur_box = [cl(x0 / s, W), cl(y0 / s, H), cl(x1 / s, W), cl(y1 / s, H)]
        self.say(f"Crop box: {self.cur_box}\nPick attributes, then press Enter.")

    # -------------------------------------------------------------- regions
    def run_analysis(self, box):
        crop = np.array(self.img.crop(tuple(box)))[:, :, ::-1].copy()
        return analyze_crop(crop)

    def add_region(self):
        attrs = self.get_attributes()
        if not attrs:
            messagebox.showwarning("Nothing picked", "Pick at least one attribute first.")
            return
        box = self.cur_box or [0, 0, self.img.width, self.img.height]
        self.win.config(cursor="watch"); self.win.update_idletasks()
        try:
            analysis = self.run_analysis(box)
        except Exception as ex:                   # analysis must never block labelling
            analysis = None
            self.say(f"(color analysis skipped: {ex})")
        self.win.config(cursor="")
        self.entry()["regions"].append({"box": box, "attributes": attrs, "analysis": analysis})
        self.cur_box = None
        if not self.keep.get():
            self.set_attributes({})
        self.refresh_regions(); self.redraw(); self.save()
        if analysis:
            self.show_analysis(analysis)

    def sel_region(self):
        s = self.rl.curselection()
        return s[0] if s else None

    def update_region(self):
        i = self.sel_region()
        if i is None:
            return self.say("Select a region in the list first.")
        self.entry()["regions"][i]["attributes"] = self.get_attributes()
        self.refresh_regions(); self.redraw(i); self.save()

    def delete_region(self):
        i = self.sel_region()
        if i is None:
            return self.say("Select a region in the list first.")
        del self.entry()["regions"][i]
        self.refresh_regions(); self.redraw(); self.save()

    def on_region_select(self, _=None):
        i = self.sel_region()
        if i is None:
            return
        r = self.entry()["regions"][i]
        self.set_attributes(r["attributes"])
        self.redraw(i)
        if r.get("analysis"):
            self.show_analysis(r["analysis"])

    def show_analysis(self, an):
        cols = ", ".join(f"{c['name']} {c['fraction']:.0%}" for c in an["colors"])
        self.say(f"Colors: {cols}")

    def analyze_current(self):
        box = self.cur_box or [0, 0, self.img.width, self.img.height]
        try:
            self.show_analysis(self.run_analysis(box))
        except Exception as ex:
            self.say(f"Analysis failed: {ex}")

    def say(self, text):
        self.info.delete("1.0", "end")
        self.info.insert("end", text)

    # ------------------------------------------------------------ nav / save
    def goto(self, i):
        self.save()
        self.idx = max(0, min(i, len(self.files) - 1))
        self.show_image()

    def save(self):
        self.data["images"] = {k: v for k, v in self.data["images"].items() if v["regions"]}
        self.ann_path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def on_close(self):
        self.save()
        self.win.destroy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?", default="dataset")
    ap.add_argument("--ann", default="annotations.json")
    a = ap.parse_args()
    root = tk.Tk()
    Annotator(root, a.folder, a.ann)
    root.mainloop()