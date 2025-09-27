# scripts/ingest.py
import json
import cv2
from pathlib import Path
from shutil import copy2
from sklearn.model_selection import train_test_split

MASTER_CLASSES = {"helmet": 0, "vest": 1, "gloves": 2, "boots": 3, "person": 4}

PLATFORMS = {
    "gloves.v1i.yolov8": {
        "glove": 0, "helmet": 1, "shoe": 2, "vest": 3
    },

    "Hard Hat Universe.v1i.yolov8": {
        "glove": 0, "helmet": 1, "person": 2, "shoe": 3, "vest": 4
    },
    "Hard Hat Universe.v1i.yolov8 (1)": {
        "glove": 0, "helmet": 1, "person": 2, "shoe": 3, "vest": 4
    },
    "Hard Hat Universe.v1i.yolov8 (3)": {
        "glove": 0, "helmet": 1, "person": 2, "shoe": 3, "vest": 4
    },
    "Hard Hat Universe.v2i.yolov8": {
        "glove": 0, "helmet": 1, "person": 2, "shoe": 3, "vest": 4
    },

    "helmet detection.v1i.yolov8": {
        "Gloves": 0, "Helmet": 1, "Shoes": 3, "Vest": 4
    },

    "helmet-vest and boots detection.v2i.yolov8": {
        "boots": 0, "gloves": 1, "helmet": 2, "vest": 6
    },

    "PPE Detection.v1i.yolov8": {
        "boots": 0, "gloves": 1, "helmet": 2, "vest": 3
    },

    "ppe equipment.v1i.yolov8": {
        "Gloves": 0, "boots": 1, "safe_hat": 2, "vest": 3
    },
}

SYNONYMS = {
    "glove": "gloves",
    "gloves": "gloves",
    "shoe": "boots",
    "shoes": "boots",
    "boots": "boots",
    "helmet": "helmet",
    "safe_hat": "helmet",
    "vest": "vest",
}


def build_remap(platforms, synonyms, master_classes):
    remaps = {}
    for pname, cmap in platforms.items():
        mapping = {}
        for lbl, orig in cmap.items():
            key = str(lbl).strip().lower()
            try:
                orig_i = int(orig)
            except Exception:
                continue
            if key in master_classes:
                mapping[orig_i] = master_classes[key]
                continue
            syn = synonyms.get(key)
            if syn and syn in master_classes:
                mapping[orig_i] = master_classes[syn]
        remaps[pname] = mapping
    return remaps

REMAP = build_remap(PLATFORMS, SYNONYMS, MASTER_CLASSES)

BLUR_THRESH = 100.0
DARK_THRESH = 50

RAW_ROOT = Path("data/raw")
PROC_IMG = Path("data/processed/all_images")
PROC_LBL = Path("data/processed/all_labels")
SPLITS_DIR = Path("data/processed/splits")

for d in (PROC_IMG, PROC_LBL, SPLITS_DIR):
    d.mkdir(parents=True, exist_ok=True)

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

def is_blurry(img):
    return cv2.Laplacian(img, cv2.CV_64F).var() < BLUR_THRESH

def is_too_dark(img):
    return img.mean() < DARK_THRESH

def write_yolo_line(out_dir: Path, fname: str, line: str):
    outfile = out_dir / Path(fname).with_suffix(".txt").name
    with outfile.open("a", encoding="utf8") as f:
        f.write(line)

def convert_coco_to_yolo(coco_json: Path, out_dir: Path, remap: dict):
    data = json.loads(coco_json.read_text(encoding="utf8"))
    imgs = {im["id"]:(im["width"], im["height"], im["file_name"]) for im in data.get("images", [])}
    for ann in data.get("annotations", []):
        cat_id = ann.get("category_id")
        candidates = [cat_id, (cat_id - 1 if isinstance(cat_id, int) else None), (cat_id + 1 if isinstance(cat_id, int) else None)]
        cls = None
        for c in candidates:
            if c is None:
                continue
            cls = remap.get(int(c))
            if cls is not None:
                orig_used = c
                break
        if cls is None:
            continue
        w,h,fname = imgs[ann["image_id"]]
        x,y,bw,bh = ann["bbox"]
        xc = (x + bw/2)/w
        yc = (y + bh/2)/h
        bw_norm = bw / w
        bh_norm = bh / h
        line = f"{cls} {xc:.6f} {yc:.6f} {bw_norm:.6f} {bh_norm:.6f}\n"
        write_yolo_line(out_dir, fname, line)

def find_image_path(img_dir: Path, stem: str):
    for ext in IMG_EXTS:
        p = img_dir / (stem + ext)
        if p.exists():
            return p
    return None

def ingest():
    files = []
    skipped = []
    for plat in sorted(RAW_ROOT.iterdir()):
        if not plat.is_dir():
            continue
        remap = REMAP.get(plat.name, {})
        if not remap:
            skipped.append(plat.name)
            continue

        img_dir, lbl_dir = plat / "images", plat / "labels"

        if img_dir.exists() and lbl_dir.exists():
            for img in img_dir.glob("*.*"):
                if img.suffix.lower() not in IMG_EXTS:
                    continue
                lbl = lbl_dir / img.with_suffix(".txt").name
                if not lbl.exists():
                    continue
                im = cv2.imread(str(img))
                if im is None:
                    print(f"WARNING: could not read image {img}")
                    continue
                if is_blurry(im) or is_too_dark(im):
                    continue
                lines_out = []
                for ln in lbl.read_text(encoding="utf8").splitlines():
                    parts = ln.split()
                    if not parts:
                        continue
                    try:
                        orig_i = int(parts[0])
                    except ValueError:
                        continue
                    cls = remap.get(orig_i)
                    if cls is None:
                        cls = remap.get(orig_i - 1) or remap.get(orig_i + 1)
                    if cls is not None:
                        coords = parts[1:]
                        lines_out.append(f"{cls} {' '.join(coords)}")
                if not lines_out:
                    continue
                out_img = PROC_IMG / img.name
                out_lbl = PROC_LBL / lbl.name
                copy2(img, out_img)
                out_lbl.write_text("\n".join(lines_out), encoding="utf8")
                copy2(out_lbl, out_img.with_suffix(".txt"))
                files.append(img.name)

        for coco in plat.glob("*.json"):
            convert_coco_to_yolo(coco, PROC_LBL, remap)
            data = json.loads(coco.read_text(encoding="utf8"))
            for iminfo in data.get("images", []):
                fname = iminfo.get("file_name")
                if not fname:
                    continue
                src = plat / "images" / fname
                if not src.exists():
                    stem = Path(fname).stem
                    src = find_image_path(plat / "images", stem)
                if not src or not src.exists():
                    continue
                img = cv2.imread(str(src))
                if img is None:
                    print(f"WARNING: could not read image {src}")
                    continue
                if is_blurry(img) or is_too_dark(img):
                    continue
                dst_img = PROC_IMG / src.name
                copy2(src, dst_img)
                lbl = PROC_LBL / Path(fname).with_suffix(".txt").name
                if lbl.exists():
                    copy2(lbl, dst_img.with_suffix(".txt"))
                files.append(src.name)

    files = sorted(set(files))
    if not files:
        print("No files ingested. Check your platform mappings and dataset layout.")
        if skipped:
            print("Skipped datasets (no mapping provided):")
            for s in skipped:
                print(" -", s)
        return

    trvl, tst = train_test_split(files, test_size=0.15, random_state=42)
    trn, val = train_test_split(trvl, test_size=0.1765, random_state=42)

    def abs_paths_for(names):
        return "\n".join(str((PROC_IMG / name).as_posix()) for name in names)

    (SPLITS_DIR / "train.txt").write_text(abs_paths_for(trn), encoding="utf8")
    (SPLITS_DIR / "val.txt").write_text(abs_paths_for(val), encoding="utf8")
    (SPLITS_DIR / "test.txt").write_text(abs_paths_for(tst), encoding="utf8")

    print(f"Ingested {len(files)} images → {len(trn)}/{len(val)}/{len(tst)} splits.")
    if skipped:
        print("\nDatasets skipped (no mapping):")
        for s in skipped:
            print(" -", s)

if __name__ == "__main__":
    print("Remap config (per-platform):")
    for k, v in REMAP.items():
        print(f" - {k}: {v}")
    ingest()
