# Garment AI: Install & Run Guide

Label clothing photos, train a model, then search your photo folder with text like
`navy v-neck striped short sleeve cropped` and see the matching pictures.

**What it recognises**

| Category | Options |
|---|---|
| Length | regular, cropped |
| Clothes type | no bottom, bottom half shirt, bottom down shirt |
| Neck type | crew neck, v-neck, off shoulder, polo neck |
| Pattern (can be several) | dot, stripe, check, corner only, graphic |
| Sleeves | long, short, sleeveless, camis |
| Colors | measured automatically from the pixels |

---

## 1. Requirements

- **Python 3.9 or newer** (3.10 or 3.11 recommended). Check with `python --version`.
- **VS Code** with the **Python** extension (Extensions panel, search "Python", install).
- About **3 GB free disk space** (PyTorch is large).
- Internet on the first training run (it downloads pretrained ResNet weights, about 45 MB).
- A GPU is optional. Training works on CPU, just slower.

> **tkinter** (used for the annotate and search windows) must be available:
> - **Windows:** included with the python.org installer.
> - **macOS:** included with the python.org installer. With Homebrew run `brew install python-tk`.
> - **Linux:** `sudo apt install python3-tk`

---

## 2. Install

1. CLone this github repo.
2. Open the clone project in VS code.
3. Open a terminal: **Terminal > New Terminal**.
4. Create a virtual environment and install the packages:

   **Windows (PowerShell)**
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
   If activation is blocked, run this once and try again:
   `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

   **macOS / Linux**
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
5. Tell VS Code to use it: press **Ctrl+Shift+P** (Cmd+Shift+P on Mac), choose
   **Python: Select Interpreter**, then pick the one inside `.venv`.

You should now see `(.venv)` at the start of your terminal line. Keep the terminal open
for the steps below. Every time you reopen VS Code, activate `.venv` again.

---

## 3. Add your pictures

```
AI-pytorch
├─ dataset/    <- pictures you will LABEL and TRAIN on
└─ database/   <- pictures you want to SEARCH
```

- Supported: `.jpg .jpeg .png .webp .bmp` (sub-folders are fine).
- `dataset/` and `database/` can hold the same pictures. `database/` can also be much bigger.
- Best results: one garment per picture, front view, plain background, flat-lay or mannequin.

---

## 4. Label your pictures

```
python 1_label/annotate.py
```

A window opens showing your first picture.

1. **Drag** on the picture to crop the area relevant to one category (for example, crop
   the garment length for Length or the neckline for Neck). No drag means the whole image
   is used.
2. **Choose one category** on the right, then select its label. Each crop is saved with
   labels for that category only. Patterns can have multiple selections (or "none of these").
3. Press **Enter** (or click *Add region*). Colors are analysed automatically.
4. Press the **Right arrow** for the next picture. Everything saves as you go to `annotations.json`.

| Key / button | Action |
|---|---|
| Enter | add region |
| Left / Right arrow | previous / next picture |
| Ctrl+S | save |
| Click a region in the list | view it, then *Update selected* or *Delete* |
| "Keep ticks after adding" | keeps the selected category's choices for the next crop |

Tips:
- Label **at least 30 examples of every option** you want it to learn, and keep them balanced.
- You may skip a category on a picture. It is ignored for that picture during training.
- Optional: enter *Pixels per cm* if you know the scale, to get length and width in cm.

---

## 5. Train, index and search (one command)

```
python run_all.py
```

This runs three steps in order:

1. **Train** the model on `annotations.json`, saved as `model.pt`.
   It prints label counts, per-category training/validation loss, and per-category
   accuracy or F1 after each epoch.
2. **Index** every picture in `database/` into `index.json`.
3. **Open the search window.**

Useful options:

```
python run_all.py --epochs 30            # train longer
python run_all.py --skip-train           # reuse the existing model.pt
python run_all.py --database my_photos   # search a different folder
python run_all.py --no-gui               # train + index only
```

In VS Code you can also use the **Run and Debug** panel (Ctrl+Shift+D), where each step
has a ready-made button.

---

## 6. Search

In the search window, type words in any order:

```
navy v-neck striped short sleeve cropped
red polo neck dots
long sleeve button down grey
```

- The status line shows how your words were understood.
- Or use the **dropdowns** for each category.
- Each result shows predictions under their category names (for example, `Length: cropped`).
- Results are ranked by match score. The **Minimum match score** slider controls how strict it is.
  Lower it if you get no results.
- **Click a picture** to open it full size.
- **Export results** copies the matches to `results/<your query>/` and opens an `index.html` gallery.

Rules: options in the same single-choice category mean "either" (`long short`);
patterns must all match (`dot stripe`); colors must all match (`red blue`).

Command-line version:

```
python search.py "navy v-neck stripe" --export
```

**Later runs:** after the first training you only need
`python search_app.py` to search again, or `python build_index.py database` after adding
new pictures to `database/` (only new pictures are processed).

---

## 7. Files

| File | Purpose |
|---|---|
| `1_label/annotate.py` | GUI for drawing garment regions, selecting labels, and saving `annotations.json` |
| `1_label/analysis.py` | segments a garment crop and estimates its dominant colors |
| `attributes.py` / `attributes.json` | loads the shared category and label definitions |
| `2_train/dataset.py` / `2_train/model.py` | prepares annotated regions and defines the training model |
| `2_train/train.py` | trains the model from `annotations.json` and writes `model.pt` |
| `2_train/inference.py` | loads `model.pt` and predicts attributes for an image |
| `build_index.py` | predicts attributes and analyzes colors for `database/`, writing `index.json` |
| `searchlib.py` | parses search terms and matches them against `index.json` |
| `search_app.py` / `search.py` | graphical / command-line search |
| `run_all.py` | runs training, indexing, and graphical search in sequence |

### What happens in `annotate.py` -> `analysis.py`

1. `annotate.py` scans the selected dataset folder (including subfolders) for supported
   image files and opens them in a labeling window.
2. You draw a rectangle around a garment, or use the full image, then select its
   attributes from the categories in `attributes.json`.
3. When you add the region, `annotate.py` crops the original image and calls
   `analysis.analyze_crop(crop)`.
4. `analysis.py` estimates a foreground mask with GrabCut and image cleanup, then
   clusters the masked pixels to estimate dominant colors. Each color includes a
   name, RGB and hex values, and an approximate fraction of the garment pixels.
5. `annotate.py` saves the region's box, chosen attributes, and color analysis together
   in `annotations.json`. If automatic color analysis raises an error, the region and
   labels are still saved; its analysis is left empty.

The analysis is a color estimate, not a clothing-attribute classifier. Garment labels
are selected by the annotator during this step; the trained model predicts attributes
later when `build_index.py` processes the search database.

To test the pixel analysis on one image without training, run `python analysis.py
photo.jpg`. It prints the estimated colors and writes `photo_mask.png` with the
detected garment area overlaid in green.

---

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `No module named 'tkinter'` | install tkinter (see Requirements) |
| `No module named 'torch'` | the venv is not active or the install failed. Run `pip install -r requirements.txt` again |
| `No labelled regions found` | run `annotate.py` first and add regions. `annotations.json` must exist |
| `model.pt` not found | you used `--skip-train` before training once. Run `python run_all.py` |
| `index.json not found` | run `python build_index.py database` |
| Download error on first training (no internet) | run `python train.py --no-pretrained` (lower accuracy) |
| Search returns 0 matches | lower the minimum score slider, use fewer words, or label more examples |
| Wrong predictions | add more labelled examples (especially for the wrong option) and retrain |
| Colors or measurements look wrong | the cut-out is poor. Use plain backgrounds and check with `analysis.py` |
| Indexing is slow | each picture is cut out and colour-analysed (about 0.5 s). Results are cached, so re-runs skip unchanged pictures |
| Changed `attributes.json` | retrain and re-index (the model keeps its own copy of the options) |

## 9. Good to know

- Pixel measurements are in **pixels** unless you give a scale, and they are only reliable
  for front-facing garments, not clothes worn on a person.
- The "cropped vs regular" value from pixels is just a hint. The trained model makes the real call.
- Everything runs **locally**. Your pictures are never uploaded anywhere.
