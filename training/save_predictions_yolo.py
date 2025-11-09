# save_predictions_yolo.py
import json, os
from ultralytics import YOLO
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--weights", required=True)  # e.g. runs/segment/.../best.pt or last.pt
parser.add_argument("--images_dir", default="test_images")
parser.add_argument("--coco_images_json", default="data/coco/valid/_annotations.coco.json")
parser.add_argument("--out_json", default="predictions_yolov8.json")
parser.add_argument("--conf", type=float, default=0.001, help="confidence threshold for predict()")
parser.add_argument("--imgsz", type=int, default=416, help="inference image size")
args = parser.parse_args()

YOLO_TO_COCO = {
    0: 1,  # glove -> GT id 1
    1: 2,  # helmet -> GT id 2
    2: 3,  # person -> GT id 3
    3: 4,  # shoe -> GT id 4
    4: 5   # vest -> GT id 5
}

coco = json.load(open(args.coco_images_json))
id2file = {img['id']: img['file_name'] for img in coco['images']}

model = YOLO(args.weights)

results = []
for image_id, file_name in tqdm(id2file.items()):
    img_path = os.path.join(args.images_dir, file_name)
    if not os.path.exists(img_path):
        print("Missing", img_path); continue
    # run prediction
    res = model.predict(source=img_path, imgsz=args.imgsz, conf=args.conf, verbose=False)
    r = res[0]
    boxes = getattr(r, "boxes", None)
    if boxes is None:
        continue
    # iterate detections
    for i in range(len(boxes)):
        xyxy = boxes.xyxy[i].cpu().numpy().tolist()  # [x1,y1,x2,y2]
        conf = float(boxes.conf[i].cpu().numpy())
        cls = int(boxes.cls[i].cpu().numpy())
        # map YOLO class index -> GT COCO id
        if cls in YOLO_TO_COCO:
            coco_id = int(YOLO_TO_COCO[cls])
        else:
            # unknown class index — skip or keep original
            print(f"Warning: unknown YOLO class index {cls} for image_id {image_id}; skipping")
            continue
        x1,y1,x2,y2 = xyxy
        w = x2 - x1; h = y2 - y1
        entry = {
            "image_id": int(image_id),
            "category_id": coco_id,
            "bbox": [float(x1), float(y1), float(w), float(h)],
            "score": float(conf)
        }
        # segmentation: not added here; add if you want mask outputs saved
        results.append(entry)

json.dump(results, open(args.out_json, "w"))
print("Saved", args.out_json, "with", len(results), "entries")
