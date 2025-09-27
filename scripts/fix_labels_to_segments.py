# scripts/yolov8/fix_labels_to_segments.py
import shutil
from pathlib import Path

LAB_DIR = Path("data/augmented/labels")
BACKUP_DIR = LAB_DIR.parent / "labels_backup"

def clip01(v): 
    return min(max(v, 0.0), 1.0)

def bbox_to_rect_segment(cx, cy, w, h):
    # cx,cy,w,h are normalized center form -> convert to corner polygon normalized
    x_min = cx - w / 2.0
    x_max = cx + w / 2.0
    y_min = cy - h / 2.0
    y_max = cy + h / 2.0
    # rectangle polygon: (x_min,y_min), (x_max,y_min), (x_max,y_max), (x_min,y_max)
    seg = [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]
    seg = [clip01(v) for v in seg]
    return seg

def valid_polygon(seg):
    # seg is list [x1,y1,x2,y2,...] normalized
    if len(seg) < 6:
        return False
    pairs = list(zip(seg[0::2], seg[1::2]))
    # count vertices inside bounds (after clipping)
    inside = sum(1 for x,y in pairs if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0)
    return inside >= 3

def process_file(p: Path):
    lines = [ln.strip() for ln in p.read_text(encoding='utf8').splitlines() if ln.strip()]
    out_lines = []
    seen = set()
    for ln in lines:
        parts = ln.split()
        if len(parts) < 5:
            # fewer than class + 4 numbers => skip
            continue
        try:
            cls = int(parts[0])
            nums = [float(x) for x in parts[1:]]
        except Exception:
            continue

        if len(nums) == 4:
            # bbox only -> convert to rectangle segment
            cx, cy, w, h = nums
            # clip bbox
            cx = clip01(cx); cy = clip01(cy); w = max(0.0, min(1.0, w)); h = max(0.0, min(1.0, h))
            seg = bbox_to_rect_segment(cx, cy, w, h)
            # compute bbox again from seg to keep bbox consistent
            xs = seg[0::2]; ys = seg[1::2]
            minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
            bx = (minx + maxx) / 2.0
            by = (miny + maxy) / 2.0
            bw = maxx - minx
            bh = maxy - miny
            out = f"{cls} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f} " + " ".join(f"{v:.6f}" for v in seg)
        else:
            # already has segment coords (bbox + seg or longer). Ensure first 4 are bbox
            # Some lines might already be: cls cx cy w h x1 y1 ...
            # We'll clip everything and validate polygon
            nums_clipped = [clip01(v) for v in nums]
            if len(nums_clipped) >= 5:
                bx, by, bw, bh = nums_clipped[0:4]
                seg = nums_clipped[4:]
                if not valid_polygon(seg):
                    # fallback: generate rect from bbox
                    seg = bbox_to_rect_segment(bx, by, bw, bh)
                out = f"{cls} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f} " + " ".join(f"{v:.6f}" for v in seg)
            else:
                # malformed line -> skip
                continue

        if out in seen:
            # skip duplicate
            continue
        seen.add(out)
        out_lines.append(out)

    # write back only if there is at least one line
    if out_lines:
        p.write_text("\n".join(out_lines), encoding='utf8')
    else:
        # if everything removed, write empty file (Ultralytics may then ignore image)
        p.write_text("", encoding='utf8')

def main():
    if not LAB_DIR.exists():
        print("Label dir not found:", LAB_DIR)
        return
    # backup
    if BACKUP_DIR.exists():
        print("Backup dir already exists:", BACKUP_DIR)
    else:
        shutil.copytree(LAB_DIR, BACKUP_DIR)
        print("Backup created at", BACKUP_DIR)
    # process
    count = 0
    for p in sorted(LAB_DIR.glob("*.txt")):
        process_file(p)
        count += 1
    print("Processed", count, "label files. Originals are in", BACKUP_DIR)

if __name__ == "__main__":
    main()
