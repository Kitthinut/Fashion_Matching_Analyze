"""GUI annotator (1_label/annotate.py). Draw per-category boxes and save annotations."""

import argparse
import json
import shutil
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

# 1_label/ -> project root
ROOT = Path(__file__).resolve().parents[1]
LABEL_DIR = Path(__file__).resolve().parent
for p in (ROOT, LABEL_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from analysis import analyze_crop
from attributes import load_schema

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
NONE = "__none__"


class Annotator:
    def __init__(self, win, folder, ann_path):
        self.win, self.folder, self.ann_path = win, Path(folder), Path(ann_path)
        self.schema = load_schema()
        self.files = sorted(
            p.relative_to(self.folder).as_posix()
            for p in self.folder.rglob("*")
            if p.suffix.lower() in EXTS
        )
        if not self.files:
            messagebox.showerror("No images", f"No images found in {self.folder}")
            raise SystemExit
        self.data = (
            json.loads(self.ann_path.read_text(encoding="utf-8"))
            if self.ann_path.exists()
            else {"images": {}}
        )
        self.data["image_root"] = str(self.folder.resolve())
        self.trash = self.folder.resolve().parent / "deleted_pictures"
        self.idx, self.start, self.cur_box, self.box_id = 0, None, None, None
        self.vars = {}
        self.category_frames = {}
        self.build_ui()
        win.after(150, self.begin)

    def build_ui(self):
        """Build canvas + right-side category / region / nav controls."""
        w = self.win
        w.title("Garment Annotator")
        w.geometry("1380x900")
        self.canvas = tk.Canvas(w, bg="#222", cursor="crosshair")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        r = tk.Frame(w, width=470, padx=8, pady=6)
        r.pack(side="right", fill="y")
        self.title = tk.Label(
            r, anchor="w", justify="left", wraplength=450, font=("Arial", 10, "bold")
        )
        self.title.pack(fill="x")

        tk.Label(r, text="Category for this crop:", anchor="w").pack(fill="x", pady=(8, 0))
        categories = self.schema["categories"]
        self.category_by_key = {cat["key"]: cat for cat in categories}
        self.category_key_by_title = {cat["title"]: cat["key"] for cat in categories}
        self.category_var = tk.StringVar(value=categories[0]["title"])
        self.category_picker = ttk.Combobox(
            r,
            state="readonly",
            textvariable=self.category_var,
            values=[cat["title"] for cat in categories],
        )
        self.category_picker.pack(fill="x")
        self.category_picker.bind("<<ComboboxSelected>>", self.on_category_change)

        self.category_container = tk.Frame(r)
        self.category_container.pack(fill="x")
        for cat in categories:
            lf = tk.LabelFrame(self.category_container, text=cat["title"], padx=4)
            self.category_frames[cat["key"]] = lf
            key, names = cat["key"], [o["name"] for o in cat["options"]]
            if cat["mode"] == "single":
                var = tk.StringVar(value="")
                self.vars[key] = var
                items = [(n, n) for n in names] + [("(skip)", "")]
                for j, (txt, val) in enumerate(items):
                    tk.Radiobutton(lf, text=txt, variable=var, value=val, anchor="w").grid(
                        row=j // 3, column=j % 3, sticky="w", padx=4
                    )
            else:
                d = {n: tk.BooleanVar() for n in names + [NONE]}
                self.vars[key] = d
                for j, n in enumerate(names + [NONE]):
                    tk.Checkbutton(
                        lf,
                        text="(none of these)" if n == NONE else n,
                        variable=d[n],
                        anchor="w",
                    ).grid(row=j // 3, column=j % 3, sticky="w", padx=4)
        self.on_category_change()

        row = tk.Frame(r)
        row.pack(fill="x", pady=3)
        tk.Button(row, text="Add region (Enter)", command=self.add_region, bg="#cfe8cf").pack(
            side="left", expand=True, fill="x"
        )
        tk.Button(row, text="Update selected", command=self.update_region).pack(
            side="left", expand=True, fill="x"
        )
        tk.Button(row, text="Delete", command=self.delete_region).pack(
            side="left", expand=True, fill="x"
        )
        self.keep = tk.BooleanVar(value=False)
        tk.Checkbutton(r, text="Keep ticks after adding", variable=self.keep, anchor="w").pack(
            fill="x"
        )

        tk.Label(r, text="Regions in this image:", anchor="w").pack(fill="x")
        self.rl = tk.Listbox(r, height=5, exportselection=False)
        self.rl.pack(fill="x")
        self.rl.bind("<<ListboxSelect>>", self.on_region_select)

        tk.Button(
            r, text="Preview shirt colors (Length crop only)", command=self.analyze_current
        ).pack(fill="x", pady=2)

        self.info = tk.Text(r, height=6, wrap="word", font=("Consolas", 9))
        self.info.pack(fill="both", expand=True)

        row = tk.Frame(r)
        row.pack(fill="x", pady=(4, 0))
        tk.Label(row, text="Go to #").pack(side="left")
        self.goto_entry = tk.Entry(row, width=6)
        self.goto_entry.pack(side="left", padx=2)
        self.goto_entry.bind("<Return>", lambda e: (self.goto_number(), "break")[1])
        tk.Button(row, text="Go", command=self.goto_number).pack(side="left")
        tk.Button(
            row, text="Next unlabelled (Ctrl+N)", command=self.jump_unlabelled, bg="#fff3c4"
        ).pack(side="right", expand=True, fill="x", padx=(6, 0))

        row = tk.Frame(r)
        row.pack(fill="x", pady=4)
        tk.Button(row, text="< Prev", command=lambda: self.goto(self.idx - 1)).pack(
            side="left", expand=True, fill="x"
        )
        tk.Button(row, text="Save", command=self.save).pack(side="left", expand=True, fill="x")
        tk.Button(row, text="Next >", command=lambda: self.goto(self.idx + 1)).pack(
            side="left", expand=True, fill="x"
        )
        tk.Button(
            r, text="Delete this picture... (Del)", command=self.delete_picture, fg="#b00020"
        ).pack(fill="x")

        w.bind("<Return>", lambda e: None if self.typing() else self.add_region())
        w.bind("<Delete>", lambda e: None if self.typing() else self.delete_picture())
        w.bind("<Left>", lambda e: None if self.typing() else self.goto(self.idx - 1))
        w.bind("<Right>", lambda e: None if self.typing() else self.goto(self.idx + 1))
        w.bind("<Control-n>", lambda e: self.jump_unlabelled())
        w.bind("<Control-s>", lambda e: self.save())
        w.protocol("WM_DELETE_WINDOW", self.on_close)

    def typing(self):
        try:
            return isinstance(self.win.focus_get(), (tk.Entry, tk.Spinbox))
        except KeyError:
            return False

    def labelled(self, rel):
        return bool(self.data["images"].get(rel, {}).get("regions"))

    def first_unlabelled(self, start=0):
        n = len(self.files)
        for k in range(n):
            i = (start + k) % n
            if not self.labelled(self.files[i]):
                return i
        return None

    def begin(self):
        self.idx = self.ask_start()
        self.show_image()

    def ask_start(self):
        n = len(self.files)
        done = [i for i, f in enumerate(self.files) if self.labelled(f)]
        first_un = self.first_unlabelled(0)
        after_last = min(done[-1] + 1, n - 1) if done else 0

        dlg = tk.Toplevel(self.win)
        dlg.title("Where do you want to start?")
        dlg.transient(self.win)
        dlg.resizable(False, False)
        choice = tk.StringVar(
            value="resume" if (done and first_un is not None) else "begin"
        )
        box = tk.Frame(dlg, padx=16, pady=10)
        box.pack()
        tk.Label(
            box,
            text=f"{n} pictures | {len(done)} labelled | {n - len(done)} left",
            font=("Arial", 10, "bold"),
        ).pack(anchor="w")

        def radio(text, val):
            tk.Radiobutton(box, text=text, variable=choice, value=val, anchor="w").pack(
                fill="x"
            )

        if first_un is not None:
            radio(f"First unlabelled (#{first_un + 1})", "resume")
        if done:
            radio(f"After last labelled (#{after_last + 1})", "after")
        radio("Start from beginning", "begin")
        result = {"idx": 0}

        def go(_=None):
            c = choice.get()
            if c == "resume" and first_un is not None:
                i = first_un
            elif c == "after":
                i = after_last
            else:
                i = 0
            result["idx"] = max(0, min(i, n - 1))
            dlg.destroy()

        tk.Button(dlg, text="Start", command=go, bg="#cfe8cf").pack(pady=8)
        dlg.protocol("WM_DELETE_WINDOW", go)
        dlg.grab_set()
        self.win.wait_window(dlg)
        return result["idx"]

    def jump_unlabelled(self):
        j = self.first_unlabelled(self.idx + 1)
        if j is None:
            self.say("All pictures are labelled.")
        else:
            self.goto(j)

    def goto_number(self):
        try:
            i = int(self.goto_entry.get()) - 1
        except ValueError:
            return self.say(f"Type 1..{len(self.files)}")
        self.goto(i)
        self.goto_entry.delete(0, "end")

    def delete_picture(self):
        rel = self.files[self.idx]
        if not messagebox.askyesno("Delete", f"Remove {rel}?"):
            return
        dst = self.trash / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst = dst.with_name(f"{dst.stem}_{int(time.time())}{dst.suffix}")
        try:
            shutil.move(str(self.folder / rel), str(dst))
        except OSError as ex:
            messagebox.showerror("Error", str(ex))
            return
        self.data["images"].pop(rel, None)
        del self.files[self.idx]
        if not self.files:
            self.save()
            self.win.destroy()
            return
        self.idx = min(self.idx, len(self.files) - 1)
        self.save()
        self.show_image()

    def get_attributes(self):
        k = self.category_key_by_title[self.category_var.get()]
        cat, v = self.category_by_key[k], self.vars[k]
        if cat["mode"] == "single":
            return {k: v.get()} if v.get() else {}
        ticked = [n for n, b in v.items() if n != NONE and b.get()]
        if ticked:
            return {k: ticked}
        if v[NONE].get():
            return {k: []}
        return {}

    def set_attributes(self, attrs):
        selected_key = next(
            (cat["key"] for cat in self.schema["categories"] if cat["key"] in attrs), None
        )
        if selected_key:
            self.category_var.set(self.category_by_key[selected_key]["title"])
            self.on_category_change()
        for cat in self.schema["categories"]:
            k, v = cat["key"], self.vars[cat["key"]]
            value = attrs.get(k) if k == selected_key else None
            if cat["mode"] == "single":
                v.set(value or "")
            else:
                for n, b in v.items():
                    b.set(n in (value or []))
                v[NONE].set(value == [])

    def on_category_change(self, _=None):
        for frame in self.category_frames.values():
            frame.pack_forget()
        key = self.category_key_by_title.get(self.category_var.get())
        if key:
            self.category_frames[key].pack(fill="x", pady=2)

    def short(self, attrs):
        labels = []
        for key, value in attrs.items():
            if not value:
                continue
            title = self.category_by_key.get(key, {}).get("title", key)
            labels.append(
                f"{title}: {'+'.join(value) if isinstance(value, list) else value}"
            )
        return " | ".join(labels) or "-"

    def entry(self):
        return self.data["images"].setdefault(self.files[self.idx], {"regions": []})

    def show_image(self):
        self.img = Image.open(self.folder / self.files[self.idx]).convert("RGB")
        cw = max(self.canvas.winfo_width(), 400)
        ch = max(self.canvas.winfo_height(), 400)
        self.scale = min(cw / self.img.width, ch / self.img.height)
        self.photo = ImageTk.PhotoImage(
            self.img.resize(
                (int(self.img.width * self.scale), int(self.img.height * self.scale))
            )
        )
        self.cur_box = self.box_id = None
        self.refresh_regions()
        self.redraw()

    def redraw(self, highlight=None):
        c = self.canvas
        c.delete("all")
        c.create_image(0, 0, anchor="nw", image=self.photo)
        for i, r in enumerate(self.entry()["regions"]):
            x0, y0, x1, y1 = [v * self.scale for v in r["box"]]
            is_length = "length" in r["attributes"]
            col = (
                ("#ff1744" if i == highlight else "#00e676")
                if is_length
                else ("#ff9100" if i == highlight else "#00e5ff")
            )
            dash = None if is_length else (4, 4)
            c.create_rectangle(x0, y0, x1, y1, outline=col, width=2, dash=dash)
            c.create_text(
                x0 + 4, y0 + 4, anchor="nw", fill=col, font=("Arial", 10, "bold"),
                text=f"{i + 1}: {self.short(r['attributes'])}",
            )
        self.box_id = None

    def refresh_regions(self):
        e = self.entry()
        done = sum(1 for f in self.files if self.labelled(f))
        self.title.config(
            text=(
                f"[{self.idx + 1}/{len(self.files)}] {self.files[self.idx]} "
                f"({len(e['regions'])} regions)\n"
                f"Labelled: {done}/{len(self.files)}"
            )
        )
        self.rl.delete(0, "end")
        for i, r in enumerate(e["regions"]):
            prefix = "[Length]" if "length" in r["attributes"] else "[Cat]"
            self.rl.insert("end", f"{i + 1}: {prefix} {self.short(r['attributes'])}")

    def on_press(self, e):
        self.start, self.cur_box = (e.x, e.y), None

    def on_drag(self, e):
        if not self.start:
            return
        if self.box_id:
            self.canvas.delete(self.box_id)
        self.box_id = self.canvas.create_rectangle(
            *self.start, e.x, e.y, outline="yellow", width=2, dash=(4, 2)
        )

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
        self.say(f"Crop box: {self.cur_box}\nPick attributes, then Enter.")

    def run_analysis(self, box):
        crop = np.array(self.img.crop(tuple(box)))[:, :, ::-1].copy()
        return analyze_crop(crop)

    def add_region(self):
        attrs = self.get_attributes()
        if not attrs:
            messagebox.showwarning("Nothing picked", "Pick a label first.")
            return
        box = self.cur_box or [0, 0, self.img.width, self.img.height]
        shirt_colors = None
        if "length" in attrs:
            try:
                shirt_colors = self.run_analysis(box).get("colors", [])
            except Exception as ex:
                self.say(f"(Color analysis skipped: {ex})")
        region_data = {"box": box, "attributes": attrs}
        if shirt_colors:
            region_data["shirt_colors"] = shirt_colors
        self.entry()["regions"].append(region_data)
        self.cur_box = None
        if not self.keep.get():
            self.set_attributes({})
        self.refresh_regions()
        self.redraw()
        self.save()
        if shirt_colors:
            self.show_shirt_colors(shirt_colors)

    def sel_region(self):
        s = self.rl.curselection()
        return s[0] if s else None

    def update_region(self):
        i = self.sel_region()
        if i is None:
            return self.say("Select a region first.")
        attrs = self.get_attributes()
        if not attrs:
            messagebox.showwarning("Nothing picked", "Pick a label first.")
            return
        self.entry()["regions"][i]["attributes"] = attrs
        self.refresh_regions()
        self.redraw(i)
        self.save()

    def delete_region(self):
        i = self.sel_region()
        if i is None:
            return self.say("Select a region first.")
        del self.entry()["regions"][i]
        self.refresh_regions()
        self.redraw()
        self.save()

    def on_region_select(self, _=None):
        i = self.sel_region()
        if i is None:
            return
        r = self.entry()["regions"][i]
        self.set_attributes(r["attributes"])
        self.redraw(i)
        if r.get("shirt_colors"):
            self.show_shirt_colors(r["shirt_colors"])

    def show_shirt_colors(self, colors):
        cols = ", ".join(f"{c['name']} ({c['fraction']:.0%})" for c in colors)
        self.say(f"Shirt Color: {cols}")

    def analyze_current(self):
        if "length" not in self.get_attributes():
            return self.say("Color analysis is for the Length category.")
        box = self.cur_box or [0, 0, self.img.width, self.img.height]
        try:
            self.show_shirt_colors(self.run_analysis(box).get("colors", []))
        except Exception as ex:
            self.say(f"Analysis failed: {ex}")

    def say(self, text):
        self.info.delete("1.0", "end")
        self.info.insert("end", text)

    def goto(self, i):
        self.save()
        self.idx = max(0, min(i, len(self.files) - 1))
        self.show_image()

    def save(self):
        self.data["images"] = {
            k: v for k, v in self.data["images"].items() if v["regions"]
        }
        self.ann_path.parent.mkdir(parents=True, exist_ok=True)
        self.ann_path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def on_close(self):
        self.save()
        self.win.destroy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?", default=ROOT / "dataset" / "train")
    ap.add_argument("--ann", default=None, help="annotations JSON path")
    a = ap.parse_args()
    folder = Path(a.folder)
    # Default ann next to split name: annotations_train.json etc.
    if a.ann:
        ann = Path(a.ann)
    else:
        split = folder.name  # train / validation / test
        ann = ROOT / f"annotations_{split}.json"
        if not ann.exists() and (folder / "annotations.json").exists():
            ann = folder / "annotations.json"
    root = tk.Tk()
    Annotator(root, folder, ann)
    root.mainloop()