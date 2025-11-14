# extract_sys_ocr.py
import json,sys
inpath = sys.argv[1]
outpath = sys.argv[2]
data = json.load(open(inpath))
out = []
for frame in data:
    image_id = frame.get("image_id")
    for p in frame.get("people",[]):
        bbox = p.get("bbox")
        out.append({
            "image_id": image_id,
            "person_bbox": bbox,
            "pred_text": p.get("ocr_text",""),
            "matched_id": p.get("matched_id_raw", p.get("id"))
        })
json.dump(out, open(outpath,"w"))
print("wrote", outpath)
