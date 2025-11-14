# pose_eval.py
import json, sys, math
def center_key(bbox): 
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2), int((y1+y2)/2)
def load_map(fname):
    d = {}
    for e in json.load(open(fname)):
        key = (e['image_id'], center_key(e['person_bbox']))
        d[key] = e['landmarks']
    return d
gt = load_map(sys.argv[1])
sysmap = load_map(sys.argv[2])
th = int(sys.argv[3]) if len(sys.argv)>3 else 10
vals = []
for k,glands in gt.items():
    if k in sysmap:
        sl = sysmap[k]
        # iterate min length
        n = min(len(glands), len(sl))
        hits = 0
        for i in range(n):
            gx,gy = glands[i]
            sx,sy = sl[i]
            dist = math.hypot(gx-sx, gy-sy)
            if dist <= th: hits += 1
        vals.append(hits / max(1,n))
# PCK@th
print("PCK@%d mean: %.4f (n=%d)"%(th, sum(vals)/len(vals) if vals else 0.0, len(vals)))
