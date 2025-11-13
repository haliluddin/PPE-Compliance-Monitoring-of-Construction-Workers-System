import json
import argparse
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--gt_frames", required=True)
parser.add_argument("--sys_frames", required=True)
parser.add_argument("--frame_gap", type=int, default=3)
parser.add_argument("--out_prefix", default="eval")
args = parser.parse_args()

gt = json.load(open(args.gt_frames))
sys = json.load(open(args.sys_frames))

gt_by_img = {f['image_id']: f for f in gt}
sys_by_img = {f['image_id']: f for f in sys}

violation_types = set()
for f in gt:
    for p in f.get('people', []):
        for v in p.get('violations', []):
            violation_types.add(v)
for f in sys:
    for p in f.get('people', []):
        for v in p.get('violations', []):
            violation_types.add(v)
violation_types = sorted(violation_types)

frame_metrics = {v: {'TP': 0, 'FP': 0, 'FN': 0} for v in violation_types}

for img_id, gtf in gt_by_img.items():
    g_viol = set()
    for p in gtf.get('people', []):
        g_viol.update(p.get('violations', []))
    s_viol = set()
    if img_id in sys_by_img:
        for p in sys_by_img[img_id].get('people', []):
            s_viol.update(p.get('violations', []))
    for v in violation_types:
        if v in g_viol and v in s_viol:
            frame_metrics[v]['TP'] += 1
        elif v in g_viol and v not in s_viol:
            frame_metrics[v]['FN'] += 1
        elif v not in g_viol and v in s_viol:
            frame_metrics[v]['FP'] += 1

frame_prec = []
frame_rec = []
frame_f1 = []
for v in violation_types:
    s = frame_metrics[v]
    TP, FP, FN = s['TP'], s['FP'], s['FN']
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    frame_prec.append(prec)
    frame_rec.append(rec)
    frame_f1.append(f1)
    print("Frame-level", v, "P,R,F1", prec, rec, f1)

def build_events(frames_by_img):
    events = defaultdict(list)
    sorted_imgs = sorted(frames_by_img.keys(), key=lambda x: (int(x.split('_')[0]) if '_' in x and x.split('_')[0].isdigit() else 0, int(x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else 0))
    for v in violation_types:
        current = None
        for img_id in sorted_imgs:
            present = any(v in p.get('violations', []) for p in frames_by_img[img_id].get('people', []))
            if present:
                if current is None:
                    current = [img_id, img_id]
                else:
                    current[1] = img_id
            else:
                if current is not None:
                    events[v].append(tuple(current))
                    current = None
        if current is not None:
            events[v].append(tuple(current))
    return events

gt_events = build_events(gt_by_img)
sys_events = build_events(sys_by_img)

event_prec = []
event_rec = []
event_f1 = []
for v in violation_types:
    gt_ev = gt_events[v]
    sys_ev = sys_events[v]
    TP = 0
    FN = 0
    FP = 0
    matched_sys = set()
    for i, g in enumerate(gt_ev):
        s_found = False
        for j, s in enumerate(sys_ev):
            if j in matched_sys:
                continue
            if not (g[1] < s[0] or s[1] < g[0]):
                s_found = True
                matched_sys.add(j)
                break
        if s_found:
            TP += 1
        else:
            FN += 1
    FP = len(sys_ev) - len(matched_sys)
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    event_prec.append(prec)
    event_rec.append(rec)
    event_f1.append(f1)
    print("Event-level", v, "P,R,F1", prec, rec, f1)

x = np.arange(len(violation_types))
width = 0.2

plt.figure(figsize=(max(8, len(violation_types) * 0.6), 6))
plt.bar(x - width, frame_prec, width)
plt.bar(x, frame_rec, width)
plt.bar(x + width, frame_f1, width)
plt.xticks(x, violation_types, rotation=45, ha='right')
plt.ylabel("Score")
plt.title("Frame-level Precision Recall F1")
plt.legend(["Precision", "Recall", "F1"])
plt.tight_layout()
frame_img = f"{args.out_prefix}_frame_metrics.png"
plt.savefig(frame_img)
plt.close()

plt.figure(figsize=(max(8, len(violation_types) * 0.6), 6))
plt.bar(x - width, event_prec, width)
plt.bar(x, event_rec, width)
plt.bar(x + width, event_f1, width)
plt.xticks(x, violation_types, rotation=45, ha='right')
plt.ylabel("Score")
plt.title("Event-level Precision Recall F1")
plt.legend(["Precision", "Recall", "F1"])
plt.tight_layout()
event_img = f"{args.out_prefix}_event_metrics.png"
plt.savefig(event_img)
plt.close()

print("Saved frame metrics image to", frame_img)
print("Saved event metrics image to", event_img)
