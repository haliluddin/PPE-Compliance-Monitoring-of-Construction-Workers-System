# build_gt_from_csv.py
import csv, json, sys
def build(csv_path, out_path):
    frames = {}
    with open(csv_path, newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            image_id = row['image_id']
            person = {
                "bbox": [float(row['x1']), float(row['y1']), float(row['x2']), float(row['y2'])],
                "id": row.get('person_id') or "UNID",
                "violations": [v.strip() for v in (row.get('violations') or '').replace(';',',').split(',') if v.strip()]
            }
            frames.setdefault(image_id, []).append(person)
    out = [{"image_id": k, "people": v} for k, v in frames.items()]
    with open(out_path, "w") as fw:
        json.dump(out, fw)
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python build_gt_from_csv.py gt.csv gt_frame_results.json")
        sys.exit(1)
    build(sys.argv[1], sys.argv[2])
