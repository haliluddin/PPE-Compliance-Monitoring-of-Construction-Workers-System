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

def poly_to_bbox_norm(coords):
    xs = coords[0::2]
    ys = coords[1::2]
    minx = min(xs); maxx = max(xs)
    miny = min(ys); maxy = max(ys)
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0
    w = maxx - minx
    h = maxy - miny
    return cx, cy, w, h

def clip_coords01(coords):
    return [min(max(v, 0.0), 1.0) for v in coords]

def write_annotation_file(entries, out_label_path: Path):
    lines = []
    for e in entries:
        cls = e['class']
        coords = e['coords']
        if e['type'] == 'bbox':
            cx, cy, w, h = coords
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        else:
            # poly: coords are normalized x1 y1 x2 y2 ...
            # clip polygon coords to [0,1]
            clipped = clip_coords01(coords)
            # count how many vertices are within bounds after clipping
            pairs = list(zip(clipped[0::2], clipped[1::2]))
            verts_in = sum(1 for x,y in pairs if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0)
            if verts_in < 3:
                # skip degenerate polygons
                continue
            # compute bbox from clipped polygon
            cx, cy, w, h = poly_to_bbox_norm(clipped)
            # ensure bbox is within reasonable bounds
            cx = min(max(cx, 0.0), 1.0)
            cy = min(max(cy, 0.0), 1.0)
            w = min(max(w, 0.0), 1.0)
            h = min(max(h, 0.0), 1.0)
            poly_str = " ".join(f"{v:.6f}" for v in clipped)
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {poly_str}")
    out_label_path.write_text("\n".join(lines))

def validate_labels(label_dir: Path):
    bad = []
    for p in label_dir.glob("*.txt"):
        for ln in p.read_text(encoding="utf8").splitlines():
            parts = ln.split()
            if not parts:
                continue
            try:
                cls = int(parts[0])
            except Exception:
                bad.append((p.name, "bad_class", ln))
                continue
            nums = list(map(float, parts[1:]))
            if len(nums) < 4:
                bad.append((p.name, "too_few_numbers", ln))
                continue
            # check bbox
            bx = nums[0]; by = nums[1]; bw = nums[2]; bh = nums[3]
            if not (0.0 <= bx <= 1.0 and 0.0 <= by <= 1.0 and 0.0 <= bw <= 1.0 and 0.0 <= bh <= 1.0):
                bad.append((p.name, "bbox_out_of_range", ln))
            # if there are segment coords, check them
            if len(nums) > 4:
                seg = nums[4:]
                if len(seg) % 2 != 0:
                    bad.append((p.name, "odd_segment_length", ln))
                    continue
                for v in seg:
                    if v < 0.0 or v > 1.0:
                        bad.append((p.name, "segment_out_of_range", ln))
                        break
    return bad

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

            # rebuild polys from keypoints, normalize relative to augmented image
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

            # now combine bbox and polys in original order
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

        # copy originals as well
        copy2(img_path, AUG_IMG / img_path.name)
        copy2(lbl_path, AUG_LBL / lbl_path.name)
        copy2(lbl_path, AUG_IMG / lbl_path.name)

    print("Augmentation complete (train split only).")
    # quick validation report
    bad = validate_labels(AUG_LBL)
    if bad:
        print("Label validation issues found (sample):")
        for fn, reason, ln in bad[:20]:
            print(f" - {fn}: {reason}")
    else:
        print("No obvious label issues detected in augmented labels.")

if __name__ == "__main__":
    augment_all()
