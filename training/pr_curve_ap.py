# pr_curve_ap.py (robust version)
import numpy as np, json
from collections import defaultdict
from sklearn.metrics import auc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
from pr_f1_confusion import bbox_iou  # reuse your bbox IoU function

parser = argparse.ArgumentParser()
parser.add_argument("--gt", default="data/coco/valid/_annotations.coco.json")
parser.add_argument("--pred", required=True)
parser.add_argument("--class_id", type=int, required=False)
parser.add_argument("--out_prefix", default="pr")
parser.add_argument("--iou", type=float, default=0.5)
args = parser.parse_args()

coco = json.load(open(args.gt))
images = {im['id']: im for im in coco['images']}
gt_by_image = defaultdict(list)
for ann in coco['annotations']:
    gt_by_image[ann['image_id']].append(ann)

preds = json.load(open(args.pred))
preds_by_image = defaultdict(list)
for p in preds:
    preds_by_image[p['image_id']].append(p)

# classes present either in GT or preds
classes = sorted({int(c['category_id']) for c in preds} | {int(a['category_id']) for a in coco['annotations']})

def compute_pr_for_class(cid):
    # collect predictions for this class
    all_preds = []
    for img_id, plist in preds_by_image.items():
        for p in plist:
            if int(p['category_id']) == cid:
                all_preds.append((img_id, p['bbox'], float(p.get('score', 0.0))))
    # sort by score desc
    all_preds = sorted(all_preds, key=lambda x: -x[2])

    # total ground truths for this class
    total_gts = sum(1 for a in coco['annotations'] if int(a['category_id']) == cid)

    # if there are no predictions:
    if len(all_preds) == 0:
        # return a trivial PR curve and AP=0.0
        rec = np.array([0.0, 1.0])
        prec = np.array([1.0, 0.0])
        return rec, prec, 0.0

    tp_arr = []
    fp_arr = []
    # track which GTs are used by their (img_id, ann_index)
    gt_used = {}
    # Build per-image lists of GTs of this class
    for img_id, bbox, score in all_preds:
        gts = [g for g in gt_by_image.get(img_id, []) if int(g['category_id']) == cid]
        matched = False
        # greedy match to first unused GT with IoU >= threshold
        for idx, g in enumerate(gts):
            key = (img_id, idx)
            if gt_used.get(key, False):
                continue
            if bbox_iou(bbox, g['bbox']) >= args.iou:
                matched = True
                gt_used[key] = True
                break
        tp_arr.append(1 if matched else 0)
        fp_arr.append(0 if matched else 1)

    tp_cum = np.cumsum(tp_arr)
    fp_cum = np.cumsum(fp_arr)
    # avoid div by zero
    rec = tp_cum / (total_gts if total_gts > 0 else 1)
    prec = tp_cum / (tp_cum + fp_cum + 1e-12)

    # prepare arrays for AUC: need at least 2 points
    if len(rec) < 2:
        # pad to 2 points
        rec = np.concatenate(([0.0], rec, [1.0]))
        prec = np.concatenate(([1.0], prec, [0.0]))
        ap = 0.0
    else:
        # ensure monotonic x for auc
        # prepend (0,1) and append (1,0) to stabilize area computation
        rec_pad = np.concatenate(([0.0], rec, [1.0]))
        prec_pad = np.concatenate(([1.0], prec, [0.0]))
        try:
            ap = auc(rec_pad, prec_pad)
        except Exception:
            ap = 0.0

        rec = rec_pad
        prec = prec_pad

    return rec, prec, float(ap)

# plotting helpers
def plot_single(rec, prec, title, path):
    plt.figure()
    plt.plot(rec, prec, marker='.')
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

if args.class_id is not None:
    cid = args.class_id
    rec, prec, ap = compute_pr_for_class(cid)
    print(f"Class {cid} AP = {ap:.6f}")
    plot_single(rec, prec, f"PR class {cid} AP={ap:.4f}", f"{args.out_prefix}_class_{cid}_pr.png")
    print("Saved", f"{args.out_prefix}_class_{cid}_pr.png")
else:
    # draw multiple PRs on same plot
    plt.figure(figsize=(8,6))
    for cid in classes:
        rec, prec, ap = compute_pr_for_class(cid)
        label = f"{cid} AP={ap:.3f}"
        plt.plot(rec, prec, label=label, linewidth=1)
        print(f"Class {cid} AP={ap:.6f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR curves per class")
    plt.legend(loc='lower left', fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    outp = f"{args.out_prefix}_all_classes_pr.png"
    plt.savefig(outp)
    plt.close()
    print("Saved PR plot:", outp)
