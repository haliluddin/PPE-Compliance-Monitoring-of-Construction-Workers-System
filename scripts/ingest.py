import json
import cv2
from pathlib import Path
from shutil import copy2
from sklearn.model_selection import train_test_split

MASTER_CLASSES = {"helmet": 0, "vest": 1, "gloves": 2, "boots": 3}
PLATFORMS = {
    "ppe-kit-detection-construction-site-workers": {"helmet": 0, "gloves": 1, "vest": 2, "boots": 3},
    "helmet-vest and boots detection.v8i.yolov8": {"boots": 0, "helmet": 1, "vest": 6}
}
REMAP = {pname: {orig: MASTER_CLASSES[lbl] for lbl, orig in cmap.items() if lbl in MASTER_CLASSES} for pname, cmap in PLATFORMS.items()}
BLUR_THRESH = 100.0
DARK_THRESH = 50
RAW_ROOT = Path("../data/raw")
PROC_IMG = Path("../data/processed/all_images")
PROC_LBL = Path("../data/processed/all_labels")
SPLITS_DIR = Path("../data/processed/splits")

for d in (PROC_IMG, PROC_LBL, SPLITS_DIR):
    d.mkdir(parents=True, exist_ok=True)

def is_blurry(img):
    return cv2.Laplacian(img, cv2.CV_64F).var() < BLUR_THRESH

def is_too_dark(img):
    return img.mean() < DARK_THRESH

def convert_coco_to_yolo(coco_json: Path, out_dir: Path, remap: dict):
    data = json.loads(coco_json.read_text())
    imgs = {im["id"]: (im["width"], im["height"], im["file_name"]) for im in data["images"]}
    for ann in data["annotations"]:
        orig = ann["category_id"] - 1
        cls = remap.get(orig)
        if cls is None:
            continue
        w, h, fname = imgs[ann["image_id"]]
        x, y, bw, bh = ann["bbox"]
        xc = (x + bw / 2) / w
        yc = (y + bh / 2) / h
        bw /= w
        bh /= h
        line = f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n"
        (out_dir / Path(fname).with_suffix(".txt").name).open("a").write(line)

def ingest():
    files = []
    for plat in RAW_ROOT.iterdir():
        if not plat.is_dir():
            continue
        remap = REMAP.get(plat.name, {})
        img_dir = plat / "images"
        lbl_dir = plat / "labels"
        if img_dir.exists() and lbl_dir.exists():
            for img in img_dir.glob("*.*"):
                lbl = lbl_dir / img.with_suffix(".txt").name
                if not lbl.exists():
                    continue
                im = cv2.imread(str(img))
                if is_blurry(im) or is_too_dark(im):
                    continue
                lines = []
                for ln in Path(lbl).read_text().splitlines():
                    orig, *coords = ln.split()
                    cls = remap.get(int(orig))
                    if cls is not None:
                        lines.append(f"{cls} {' '.join(coords)}")
                if not lines:
                    continue
                out_img = PROC_IMG / img.name
                out_lbl = PROC_LBL / lbl.name
                copy2(img, out_img)
                out_lbl.write_text("\n".join(lines))
                files.append(img.name)
        for coco in plat.glob("*.json"):
            convert_coco_to_yolo(coco, PROC_LBL, remap)
            data = json.loads(coco.read_text())
            for iminfo in data["images"]:
                src = plat / "images" / iminfo["file_name"]
                if not src.exists():
                    continue
                img = cv2.imread(str(src))
                if is_blurry(img) or is_too_dark(img):
                    continue
                lbl = PROC_LBL / src.with_suffix(".txt").name
                if not lbl.exists():
                    continue
                copy2(src, PROC_IMG / src.name)
                files.append(src.name)
    files = sorted(set(files))
    trvl, tst = train_test_split(files, test_size=0.15, random_state=42)
    trn, val = train_test_split(trvl, test_size=0.1765, random_state=42)
    (SPLITS_DIR / "train.txt").write_text("\n".join(trn))
    (SPLITS_DIR / "val.txt").write_text("\n".join(val))
    (SPLITS_DIR / "test.txt").write_text("\n".join(tst))
    print(f"Ingested {len(files)} → {len(trn)}/{len(val)}/{len(tst)}")

if __name__ == "__main__":
    ingest()
