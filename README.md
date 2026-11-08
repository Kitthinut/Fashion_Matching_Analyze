```markdown
# Garment labeling kit

Thanks for helping label! OvO

## What you need

- Python 3.9+
- A folder of dataset

### Install (one time)

```bash
pip install pillow numpy opencv-python
```

If the app window does not open on Linux:

```bash
sudo apt install python3-tk
```

## Folder layout

```
friend_kit/
  annotate.py
  analysis.py
  attributes.py
  attributes.json
  dataset/          ← put the images here
```

Supported image types: `.jpg` `.jpeg` `.png` `.webp` `.bmp`

## Run

```bash
cd friend_kit
python annotate.py dataset
```

Labels are saved automatically to `annotations.json` in the same folder.

To choose a different save path:

```bash
python annotate.py dataset --ann annotations.json
```

## How to label

1. Start at **first unlabelled** (or pick another option in the startup dialog).
2. **Draw a box** on the part of the shirt for one category  
   (e.g. whole shirt for length, collar area for neck, sleeve area for sleeve).
3. In the right panel, pick that **category**, then tick the correct option(s).
4. Press **Enter** or click **Add region**.
5. Draw another box for the next category on the same image (separate boxes).
6. Click **Next >** when the image is done.
7. Repeat until finished. Closing the window also saves.

### Tips

- One **box per category** (length, neck, sleeve, pattern, …).
- For solid/plain shirts, use **no pattern** (or solid/plain if that is the option name).
- Skip bad photos with **Delete this picture** if needed.
- Use **Next unlabelled** to jump to work that is still open.

### Shortcuts

| Key | Action |
|-----|--------|
| Enter | Add region |
| ← → | Previous / next image |
| Ctrl+N | Next unlabelled |
| Ctrl+S | Save |
| Del | Delete this picture |

## When you are done

Send back:

1. `annotations.json`
2. Your `dataset/` folder (or only the images you labeled)

Zip is fine:

```text
alice_labels.zip
  annotations.json
  dataset/
    img001.jpg
    img002.jpg
    ...
```
