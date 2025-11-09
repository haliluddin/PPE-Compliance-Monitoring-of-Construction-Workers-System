# coco_eval.py
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json, sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gt", default="data/coco/valid/_annotations.coco.json")
parser.add_argument("--pred", required=True)
parser.add_argument("--iou_type", default="bbox", choices=["bbox","segm"])
args = parser.parse_args()

cocoGt = COCO(args.gt)
cocoDt = cocoGt.loadRes(args.pred)
cocoEval = COCOeval(cocoGt, cocoDt, iouType=args.iou_type)
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

# optionally save out per-category AP
stats = {}
cats = cocoGt.loadCats(cocoGt.getCatIds())
for cat in cats:
    catId = cat['id']
    p = cocoEval.eval['precision']  # shape [TxRxKxAxM]
    # For simplicity we won't extract per-cat from here; alternative: run COCOeval with cat-specific eval.
