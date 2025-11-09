# save_predictions_timm_maskrcnn.py
import torch, json, os
from PIL import Image
from torchvision import transforms
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--weights", required=True)  # model_final.pth
parser.add_argument("--images_dir", default="test_images")
parser.add_argument("--coco_images_json", default="annotations/instances_test_coco.json")
parser.add_argument("--out_json", default="predictions_efficientnet.json")
args = parser.parse_args()

# Load model - replace with your model constructor
from my_model_def import get_model  # implement to match training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(num_classes=6)  # change accordingly
model.load_state_dict(torch.load(args.weights))
model.to(device).eval()

coco = json.load(open(args.coco_images_json))
id2file = {img['id']: img['file_name'] for img in coco['images']}

transform = transforms.Compose([transforms.ToTensor()])  # replace with your eval transforms

results = []
for image_id, file_name in tqdm(id2file.items()):
    img_path = os.path.join(args.images_dir, file_name)
    img = Image.open(img_path).convert("RGB")
    inp = transform(img).to(device)[None]
    with torch.no_grad():
        outputs = model(inp)[0]  # dict with boxes (x1,y1,x2,y2), scores, labels
    boxes = outputs['boxes'].cpu().numpy()
    scores = outputs['scores'].cpu().numpy()
    labels = outputs['labels'].cpu().numpy()
    for b,s,l in zip(boxes,scores,labels):
        x1,y1,x2,y2 = b
        w = x2-x1; h = y2-y1
        results.append({"image_id": int(image_id), "category_id": int(l), "bbox":[float(x1),float(y1),float(w),float(h)], "score": float(s)})
json.dump(results, open(args.out_json,"w"))
print("Saved", args.out_json)
