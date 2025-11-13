# sys_to_csv.py
import json, csv, sys
if len(sys.argv) < 3:
    print("Usage: python sys_to_csv.py sys_json_path out_csv_path")
    sys.exit(1)
sys_json = sys.argv[1]
out_csv = sys.argv[2]
data = json.load(open(sys_json))
with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["image_id","person_id","x1","y1","x2","y2","violations"])
    for frame in data:
        image_id = frame.get("image_id")
        for p in frame.get("people", []):
            pid = p.get("id") or "UNID"
            bbox = p.get("bbox") or [0,0,0,0]
            vs = ";".join(p.get("violations", []))
            w.writerow([image_id, pid, bbox[0], bbox[1], bbox[2], bbox[3], vs])
print("Wrote", out_csv)
