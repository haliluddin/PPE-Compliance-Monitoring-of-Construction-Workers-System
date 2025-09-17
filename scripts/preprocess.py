# scripts/preprocess.py
import albumentations as A
import cv2
from pathlib import Path
from shutil import copy2

PROC_IMG    = Path("data/processed/all_images")
PROC_LBL    = Path("data/processed/all_labels")
SPLITS_DIR  = Path("data/processed/splits")

AUG_IMG     = Path("data/augmented/images")
AUG_LBL     = Path("data/augmented/labels")
N_AUG_PER   = 3

for d in (AUG_IMG, AUG_LBL):
    d.mkdir(parents=True, exist_ok=True)

train_list_path = SPLITS_DIR / "train.txt"
if not train_list_path.exists():
    raise FileNotFoundError(f"Train split file not found: {train_list_path}")
train_files = set(Path(p).name for p in train_list_path.read_text(encoding="utf8").splitlines())

pipeline = A.Compose(
    [
        A.Resize(416, 416),
        A.RandomRotate90(),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15),
    ],
    bbox_params=A.BboxParams(format='yolo', label_fields=['bbox_class_labels']),
    keypoint_params=A.KeypointParams(format='xy', label_fields=['kp_class_labels'], remove_invisible=False),
)

def load_annotation_file(label_path: Path):
    entries = []
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        cls = int(parts[0])
        coords = list(map(float, parts[1:]))
        if len(coords) == 4:
            entries.append({'type': 'bbox', 'class': cls, 'coords': coords})
        else:
            entries.append({'type': 'poly', 'class': cls, 'coords': coords})
    return entries

def write_annotation_file(entries, out_label_path: Path):
    lines = []
    for e in entries:
        cls = e['class']
        coords = e['coords']
        lines.append(f"{cls} {' '.join(f'{v:.6f}' for v in coords)}")
    out_label_path.write_text("\n".join(lines))

def augment_all():
    for img_path in PROC_IMG.glob("*.*"):
        if img_path.name not in train_files:
            continue

        lbl_path = PROC_LBL / img_path.with_suffix(".txt").name
        if not lbl_path.exists():
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"Warning: can't read {img_path}")
            continue
        orig_h, orig_w = img_bgr.shape[:2]

        entries = load_annotation_file(lbl_path)
        bboxes = []
        bbox_class_labels = []
        keypoints = []
        kp_class_labels = []
        poly_group_sizes = []
        poly_classes = []
        entry_is_bbox = []

        for e in entries:
            if e['type'] == 'bbox':
                entry_is_bbox.append('bbox')
                bboxes.append(tuple(e['coords']))
                bbox_class_labels.append(e['class'])
            else:
                entry_is_bbox.append('poly')
                coords = e['coords']
                if len(coords) % 2 != 0:
                    raise ValueError(f"Invalid polygon coords length in {lbl_path}: {coords}")
                n_pts = len(coords) // 2
                poly_group_sizes.append(n_pts)
                poly_classes.append(e['class'])
                for i in range(0, len(coords), 2):
                    nx = coords[i]
                    ny = coords[i+1]
                    keypoints.append((nx * orig_w, ny * orig_h))
                    kp_class_labels.append(e['class'])

        for i in range(N_AUG_PER):
            augmented = pipeline(
                image=img_bgr,
                bboxes=bboxes,
                bbox_class_labels=bbox_class_labels,
                keypoints=keypoints,
                kp_class_labels=kp_class_labels,
            )
            stem = img_path.stem + f"_aug{i}"
            ext  = img_path.suffix

            out_img = AUG_IMG / (stem + ext)
            out_lbl = AUG_LBL / (stem + ".txt")

            save_img = augmented['image']
            cv2.imwrite(str(out_img), save_img)

            aug_bboxes = augmented.get('bboxes', [])
            aug_bbox_classes = augmented.get('bbox_class_labels', [])
            aug_keypoints = augmented.get('keypoints', [])

            polys_reconstructed = []
            kp_idx = 0
            for group_i, pts_count in enumerate(poly_group_sizes):
                coords_norm = []
                for _ in range(pts_count):
                    x_px, y_px = aug_keypoints[kp_idx]
                    coords_norm.append(x_px / save_img.shape[1])
                    coords_norm.append(y_px / save_img.shape[0])
                    kp_idx += 1
                cls = poly_classes[group_i]
                polys_reconstructed.append({'class': cls, 'coords': coords_norm})

            final_entries = []
            bbox_i = 0
            poly_i = 0
            for typ in entry_is_bbox:
                if typ == 'bbox':
                    if bbox_i >= len(aug_bboxes):
                        bbox_i += 1
                        continue
                    cls = aug_bbox_classes[bbox_i]
                    coords = list(aug_bboxes[bbox_i])
                    final_entries.append({'type': 'bbox', 'class': int(cls), 'coords': coords})
                    bbox_i += 1
                else:
                    if poly_i >= len(polys_reconstructed):
                        poly_i += 1
                        continue
                    poly = polys_reconstructed[poly_i]
                    final_entries.append({'type': 'poly', 'class': int(poly['class']), 'coords': poly['coords']})
                    poly_i += 1

            write_annotation_file(final_entries, out_lbl)
            copy2(out_lbl, out_img.with_suffix(".txt"))

        copy2(img_path, AUG_IMG / img_path.name)
        copy2(lbl_path, AUG_LBL / lbl_path.name)
        copy2(lbl_path, AUG_IMG / lbl_path.name)

    print("Augmentation complete (train split only).")

if __name__ == "__main__":
    augment_all()
