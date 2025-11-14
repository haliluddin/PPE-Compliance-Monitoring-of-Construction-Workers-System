# extract_sys_pose.py
import json, sys
if len(sys.argv) < 3:
    print("Usage: python extract_sys_pose.py sys_json out_pose_json")
    raise SystemExit(1)
sys_in = sys.argv[1]; outp = sys.argv[2]
data = json.load(open(sys_in))
out = []
for frame in data:
    image_id = frame.get("image_id")
    for p in frame.get("people", []):
        bbox = p.get("bbox") or [0,0,0,0]
        lands = p.get("landmarks")
        # if landmarks is None skip or include empty
        if lands:
            out.append({"image_id": image_id, "person_bbox": bbox, "landmarks": lands})
json.dump(out, open(outp,"w"))
print("wrote", outp)
