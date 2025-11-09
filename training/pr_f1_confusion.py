# pr_f1_confusion.py  (robust version)
import json, numpy as np
from collections import defaultdict
from tqdm import tqdm
import argparse
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import matplotlib.pyplot as plt

def bbox_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH
    areaA = boxA[2] * boxA[3]
    areaB = boxB[2] * boxB[3]
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0

parser = argparse.ArgumentParser()
parser.add_argument("--gt", default="data/coco/valid/_annotations.coco.json")
parser.add_argument("--pred", required=True)
parser.add_argument("--iou", type=float, default=0.5)
parser.add_argument("--out_prefix", default="eval_out")
args = parser.parse_args()

# load GT
coco = json.load(open(args.gt))
images = {im['id']: im for im in coco['images']}
gt_by_image = defaultdict(list)
gt_cat_ids = set()
for ann in coco['annotations']:
    gt_by_image[ann['image_id']].append({'bbox': ann['bbox'], 'category_id': ann['category_id'], 'used': False})
    gt_cat_ids.add(int(ann['category_id']))

# load predictions
preds = json.load(open(args.pred))
preds_by_image = defaultdict(list)
pred_cat_ids = set()
for p in preds:
    preds_by_image[p['image_id']].append(p)
    pred_cat_ids.add(int(p['category_id']))
for k in preds_by_image:
    preds_by_image[k] = sorted(preds_by_image[k], key=lambda x: -x.get('score', 0.0))

# union of category ids (GT + preds)
all_cat_ids = sorted(set(list(gt_cat_ids) + list(pred_cat_ids)))

# build names dict (GT names where available; otherwise use placeholder name)
cats = {}
for c in coco.get('categories', []):
    cats[int(c['id'])] = c['name']
for cid in all_cat_ids:
    if cid not in cats:
        cats[cid] = f"pred_{cid}"

# initialize per-class counts and confusion
per_class = {cid: {"TP":0, "FP":0, "FN":0} for cid in all_cat_ids}
confusion = {cid: defaultdict(int) for cid in all_cat_ids}  # GT -> predicted counts

# process per image with best-IoU matching
for img_id in tqdm(images.keys()):
    gts = [g.copy() for g in gt_by_image.get(img_id, [])]  # copy so we can modify 'used'
    preds_img = preds_by_image.get(img_id, [])

    # For each prediction (sorted by score desc), find best unmatched GT by IoU
    for p in preds_img:
        pbox = p['bbox']
        pcat = int(p['category_id'])
        best_iou = 0.0
        best_gt = None
        best_gt_idx = None
        for gi, gt in enumerate(gts):
            if gt.get('used'): 
                continue
            iou = bbox_iou(pbox, gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt = gt
                best_gt_idx = gi

        if best_gt is not None and best_iou >= args.iou:
            # match found
            gts[best_gt_idx]['used'] = True
            gt_cat = int(best_gt['category_id'])
            # update confusion always (GT->Pred)
            confusion[gt_cat][pcat] += 1
            if gt_cat == pcat:
                per_class[pcat]['TP'] += 1
            else:
                # misclassification: treat as FP for predicted class, FN for GT class
                per_class[pcat]['FP'] += 1
                per_class[gt_cat]['FN'] += 1
        else:
            # no matching GT -> false positive for predicted class
            if pcat not in per_class:
                per_class[pcat] = {"TP":0,"FP":0,"FN":0}
            per_class[pcat]['FP'] += 1

    # remaining unmatched GTs are false negatives
    for gt in gts:
        if not gt.get('used', False):
            gt_cat = int(gt['category_id'])
            per_class[gt_cat]['FN'] += 1

# compute precision/recall/f1
metrics = {}
for cid in all_cat_ids:
    s = per_class.get(cid, {"TP":0,"FP":0,"FN":0})
    TP, FP, FN = s['TP'], s['FP'], s['FN']
    prec = TP/(TP+FP) if (TP+FP)>0 else 0.0
    rec = TP/(TP+FN) if (TP+FN)>0 else 0.0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    metrics[cid] = {"name":cats.get(cid, str(cid)),"TP":TP,"FP":FP,"FN":FN,"precision":prec,"recall":rec,"f1":f1}

# print per-class metrics
print("Per-class metrics:")
for cid in all_cat_ids:
    m = metrics[cid]
    print(f"{cid} ({m['name']}): TP={m['TP']} FP={m['FP']} FN={m['FN']} P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}")

# build confusion matrix (rows: GT, cols: Pred)
mat = np.zeros((len(all_cat_ids), len(all_cat_ids)), dtype=int)
id_to_idx = {cid:i for i,cid in enumerate(all_cat_ids)}
for gt_c in all_cat_ids:
    for pred_c, cnt in confusion[gt_c].items():
        mat[id_to_idx[gt_c], id_to_idx[pred_c]] = cnt

plt.figure(figsize=(max(8, len(all_cat_ids)), max(6, len(all_cat_ids))))
sns.heatmap(mat, annot=True, fmt='d', xticklabels=[cats[c] for c in all_cat_ids], yticklabels=[cats[c] for c in all_cat_ids], cmap="Blues")
plt.xlabel("Predicted"); plt.ylabel("Ground Truth"); plt.title("Confusion matrix (GT rows -> Pred columns)")
plt.tight_layout(); plt.savefig(f"{args.out_prefix}_confusion.png")
print("Saved confusion heatmap as", f"{args.out_prefix}_confusion.png")
