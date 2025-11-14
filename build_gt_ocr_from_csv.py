# build_gt_ocr_from_csv.py
import csv, json, sys
if len(sys.argv) < 3:
    print("Usage: python build_gt_ocr_from_csv.py gt_csv out_json"); sys.exit(1)
csvp = sys.argv[1]; outp = sys.argv[2]
rows = []
with open(csvp, newline='') as f:
    r = csv.DictReader(f)
    for row in r:
        bbox = [float(row['x1']), float(row['y1']), float(row['x2']), float(row['y2'])]
        rows.append({"image_id": row['image_id'], "person_bbox": bbox, "gt_id": row['gt_id']})
json.dump(rows, open(outp,"w"))
print("wrote", outp)
