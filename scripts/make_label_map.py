import json
from pathlib import Path

train_json = Path("data/new/train/_annotations.coco.json")
with train_json.open() as f:
    coco = json.load(f)

cats = coco.get("categories", [])
cats_sorted = sorted(cats, key=lambda c: c["name"])
out = Path("data/label_map.pbtxt")
with out.open("w") as w:
    for new_id, c in enumerate(cats_sorted, start=1):
        w.write("item {\n")
        w.write(f"  id: {new_id}\n")
        w.write(f'  name: "{c["name"]}"\n')
        w.write("}\n\n")
print("Wrote", out)
