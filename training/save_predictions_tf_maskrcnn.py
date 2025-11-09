# save_predictions_tf_maskrcnn.py
import tensorflow as tf, json, os
from tqdm import tqdm
import numpy as np
import argparse
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--saved_model_dir", required=True)
parser.add_argument("--coco_images_json", default="data/coco/valid/_annotations.coco.json")
parser.add_argument("--images_dir", default="test_images")
parser.add_argument("--out_json", default="predictions_maskrcnn.json")
args = parser.parse_args()

coco = json.load(open(args.coco_images_json))
id2file = {img['id']: img['file_name'] for img in coco['images']}

detect_fn = tf.saved_model.load(args.saved_model_dir)

results = []
for image_id, file_name in tqdm(id2file.items()):
    img_path = os.path.join(args.images_dir, file_name)
    img = np.array(Image.open(img_path).convert("RGB"))
    input_tensor = tf.convert_to_tensor(img)[tf.newaxis, ...]
    detections = detect_fn(input_tensor)
    # detections contain 'detection_boxes','detection_scores','detection_classes','num_detections'
    num = int(detections['num_detections'][0])
    boxes = detections['detection_boxes'][0][:num].numpy()  # y1,x1,y2,x2 normalized
    scores = detections['detection_scores'][0][:num].numpy()
    classes = detections['detection_classes'][0][:num].numpy().astype(int)
    H, W = img.shape[:2]
    for b, s, c in zip(boxes, scores, classes):
        y1, x1, y2, x2 = b
        x = float(x1 * W); y = float(y1 * H); w = float((x2 - x1) * W); h = float((y2 - y1) * H)
        results.append({"image_id": int(image_id), "category_id": int(c), "bbox": [x,y,w,h], "score": float(s)})
json.dump(results, open(args.out_json, "w"))
print("Saved", args.out_json)
