# ocr_eval.py
import json, editdistance, argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--gt_ocr", required=True)  # list of {image_id, person_bbox, gt_id_str}
parser.add_argument("--sys_ocr", required=True) # list of {image_id, person_bbox, pred_text, matched_id (UNID or ID)}
args = parser.parse_args()

gt_list = json.load(open(args.gt_ocr))
sys_list = json.load(open(args.sys_ocr))

# map by (image_id, approx person bbox center) -> gt/pred
def center_key(bbox):
    x,y,w,h = bbox
    return (int(x + w/2), int(y + h/2))

gt_map = {}
for e in gt_list:
    gt_map[(e['image_id'], center_key(e['person_bbox']))] = e['gt_id']

# compare
cer_list = []
TP=FP=FN=0
for s in sys_list:
    key = (s['image_id'], center_key(s['person_bbox']))
    pred_str = s.get('pred_text',"")
    matched = key in gt_map
    if matched:
        gt_str = gt_map[key]
        cer = editdistance.eval(pred_str, gt_str)/max(1,len(gt_str))
        cer_list.append(cer)
        if pred_str == gt_str: TP+=1
        else: FP+=1  # incorrect text but present (counted as FP for exact-match)
    else:
        # system produced OCR where no GT -> FP
        FP+=1

# count missed (FN) = GT entries without system match
for k in gt_map:
    # naive: if no sys entry for same key
    # you may need robust spatial matching
    found = any(k[0]==s['image_id'] and abs(k[1][0]-center_key(s['person_bbox'])[0])<10 and abs(k[1][1]-center_key(s['person_bbox'])[1])<10 for s in sys_list)
    if not found: FN+=1

print("CER mean:", sum(cer_list)/len(cer_list) if cer_list else None)
prec = TP/(TP+FP) if (TP+FP)>0 else 0.0
rec = TP/(TP+FN) if (TP+FN)>0 else 0.0
f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
print("ID match P/R/F1:", prec, rec, f1)