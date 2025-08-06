import albumentations as A
import cv2
from pathlib import Path
from shutil import copy2

PROC_IMG    = Path("../data/processed/all_images")
PROC_LBL    = Path("../data/processed/all_labels")
AUG_IMG     = Path("../data/augmented/images")
AUG_LBL     = Path("../data/augmented/labels")
N_AUG_PER   = 3

for d in (AUG_IMG, AUG_LBL):
    d.mkdir(parents=True, exist_ok=True)

pipeline = A.Compose([
    A.Resize(416, 416),
    A.RandomRotate90(),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def load_bboxes(label_path: Path):
    bboxes, class_labels = [], []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        class_labels.append(int(parts[0]))
        bboxes.append(list(map(float, parts[1:])))
    return bboxes, class_labels

def save_bboxes(bboxes, class_labels, out_label_path: Path):
    lines = [
        f"{cls} {' '.join(f'{v:.6f}' for v in box)}"
        for box, cls in zip(bboxes, class_labels)
    ]
    out_label_path.write_text("\n".join(lines))

def augment_all():
    for img_path in PROC_IMG.glob("*.*"):
        lbl_path = PROC_LBL / img_path.with_suffix(".txt").name
        if not lbl_path.exists():
            continue

        img = cv2.imread(str(img_path))
        img = img.astype("float32") / 255.0

        bboxes, class_labels = load_bboxes(lbl_path)

        for i in range(N_AUG_PER):
            augmented = pipeline(image=img, bboxes=bboxes, class_labels=class_labels)
            stem = img_path.stem + f"_aug{i}"
            ext  = img_path.suffix

            out_img = AUG_IMG / (stem + ext)
            out_lbl = AUG_LBL / (stem + ".txt")

            save_img = (augmented["image"] * 255).astype("uint8")
            cv2.imwrite(str(out_img), save_img)
            save_bboxes(augmented["bboxes"], augmented["class_labels"], out_lbl)

        copy2(img_path, AUG_IMG / img_path.name)
        copy2(lbl_path, AUG_LBL / lbl_path.name)

    print("Augmentation complete.")

if __name__ == "__main__":
    augment_all()
