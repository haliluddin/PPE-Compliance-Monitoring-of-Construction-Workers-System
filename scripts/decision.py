import cv2, time, sys, re
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import easyocr
import mediapipe as mp

INPUT_PATH = "4271760-hd_1920_1080_30fps.mp4"
YOLO_MODEL = "yolov8n.pt"
PPE_MODEL_PATH = "runs/segment/ppe_yolov8n_seg_run1/weights/best.pt"
DETECT_CONF = 0.35
DETECT_IOU = 0.45
PPE_CLASS_IDS = [0,1,3,4]
WINDOW = "Monitor"
SCALED = True
WMAX, HMAX = 1280, 720
OCR_INTERVAL = 6
OCR_GPU = False
CROP_PAD = 0.12
VIS_THRESH = 0.3
MAX_PEOPLE = 30

yolo = YOLO(YOLO_MODEL)
yolo_ppe = YOLO(PPE_MODEL_PATH)
ocr_reader = easyocr.Reader(['en'], gpu=OCR_GPU)

mp_pose = mp.solutions.pose
pose_conf = dict(static_image_mode=False, model_complexity=1, smooth_landmarks=True,
                 enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
left_indices = {lm.value for lm in mp_pose.PoseLandmark if "LEFT" in lm.name}
right_indices = {lm.value for lm in mp_pose.PoseLandmark if "RIGHT" in lm.name}

def ask_ids():
    while True:
        s = input("Enter registered IDs (comma separated): ").strip()
        parts = [p.strip() for p in s.split(",") if p.strip()]
        ids = []
        for p in parts:
            m = re.findall(r"\d+", p)
            if m: ids.append(m[0])
        if ids: return set(ids)
        print("Enter at least one numeric ID")

def open_cap(path):
    try: i = int(path); return cv2.VideoCapture(i)
    except Exception: return cv2.VideoCapture(path)

def scale_to_fit(img):
    h,w = img.shape[:2]; s = min(1.0, WMAX/w, HMAX/h)
    if s < 1.0: return cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    return img

def expand_bbox(b, w, h, pad=CROP_PAD):
    x1,y1,x2,y2 = b; ww=x2-x1; hh=y2-y1; pad_px=int(max(ww,hh)*pad)
    return max(0,int(x1)-pad_px), max(0,int(y1)-pad_px), min(w-1,int(x2)+pad_px), min(h-1,int(y2)+pad_px)

def combine_boxes_by_class(r, H, W):
    out = {cid:[] for cid in PPE_CLASS_IDS}
    boxes = getattr(r, "boxes", None)
    if boxes is None or len(boxes)==0: return out
    try:
        xyxy = boxes.xyxy.cpu().numpy(); confs = boxes.conf.cpu().numpy(); cls = boxes.cls.cpu().numpy().astype(int)
    except Exception:
        xyxy = np.array(boxes.xyxy); confs = np.array(boxes.conf); cls = np.array(boxes.cls).astype(int)
    for (x1,y1,x2,y2),c,cl in zip(xyxy, confs, cls):
        if int(cl) not in PPE_CLASS_IDS: continue
        x1c = max(0,float(x1)); y1c = max(0,float(y1)); x2c = min(W-1,float(x2)); y2c = min(H-1,float(y2))
        out[int(cl)].append((x1c,y1c,x2c,y2c,float(c)))
    return out

def iou(A,B):
    xA=max(A[0],B[0]); yA=max(A[1],B[1]); xB=min(A[2],B[2]); yB=min(A[3],B[3])
    interW=max(0.0,xB-xA); interH=max(0.0,yB-yA); inter=interW*interH
    aA=max(0.0,A[2]-A[0])*max(0.0,A[3]-A[1]); aB=max(0.0,B[2]-B[0])*max(0.0,B[3]-B[1])
    u=aA+aB-inter
    return inter/u if u>0 else 0.0

def check_ppe(boxes_by_class, person_bbox, landmarks, xoff, yoff, cw, ch):
    flags = {"helmet":False,"vest":False,"left_glove":False,"right_glove":False,"left_shoe":False,"right_shoe":False}
    LM = mp_pose.PoseLandmark
    def abs_lm(idx):
        lm = landmarks.landmark[idx]; return (xoff + lm.x*cw, yoff + lm.y*ch, getattr(lm,"visibility",0.0))
    head_pts=[]
    for idx in [LM.NOSE.value,LM.LEFT_EYE.value,LM.RIGHT_EYE.value,LM.LEFT_EAR.value,LM.RIGHT_EAR.value]:
        lm = landmarks.landmark[idx]
        if getattr(lm,"visibility",0.0) >= VIS_THRESH:
            head_pts.append((xoff+lm.x*cw, yoff+lm.y*ch))
    if head_pts:
        hx = int(np.mean([p[0] for p in head_pts])); hy=int(np.mean([p[1] for p in head_pts]))
    else:
        hx=int((person_bbox[0]+person_bbox[2])/2); hy=int(person_bbox[1]+(person_bbox[3]-person_bbox[1])*0.12)
    lwx,lwy,lwv = abs_lm(LM.LEFT_WRIST.value); rwx,rwy,rwv = abs_lm(LM.RIGHT_WRIST.value)
    lax,lay,lav = abs_lm(LM.LEFT_ANKLE.value); rax,ray,rav = abs_lm(LM.RIGHT_ANKLE.value)
    try:
        lsx,lsy,lsv = abs_lm(LM.LEFT_SHOULDER.value); rsx,rsy,rsv = abs_lm(LM.RIGHT_SHOULDER.value)
        lhx,lhy,lhv = abs_lm(LM.LEFT_HIP.value); rhx,rhy,rhv = abs_lm(LM.RIGHT_HIP.value)
    except Exception:
        lsx=rsx=lhx=rhx=lsy=rsy=lhy=rhy=None; lsv=rsv=lhv=rhv=0.0
    glove_boxes = boxes_by_class.get(0,[]); helmet_boxes = boxes_by_class.get(1,[]); shoe_boxes = boxes_by_class.get(3,[]); vest_boxes = boxes_by_class.get(4,[])
    wrist_radius = max(16,int((person_bbox[2]-person_bbox[0])*0.04))
    if lwv >= VIS_THRESH:
        wb = (lwx-wrist_radius, lwy-wrist_radius, lwx+wrist_radius, lwy+wrist_radius)
        for bx in glove_boxes:
            if iou(wb, (bx[0],bx[1],bx[2],bx[3]))>0.02: flags["left_glove"]=True; break
    if rwv >= VIS_THRESH:
        wb = (rwx-wrist_radius, rwy-wrist_radius, rwx+wrist_radius, rwy+wrist_radius)
        for bx in glove_boxes:
            if iou(wb, (bx[0],bx[1],bx[2],bx[3]))>0.02: flags["right_glove"]=True; break
    ankle_radius = max(20,int((person_bbox[3]-person_bbox[1])*0.05))
    if lav >= 0.05:
        ab = (lax-ankle_radius, lay-ankle_radius, lax+ankle_radius, lay+ankle_radius)
        for bx in shoe_boxes:
            if iou(ab,(bx[0],bx[1],bx[2],bx[3]))>0.02: flags["left_shoe"]=True; break
    if rav >= 0.05:
        ab = (rax-ankle_radius, ray-ankle_radius, rax+ankle_radius, ray+ankle_radius)
        for bx in shoe_boxes:
            if iou(ab,(bx[0],bx[1],bx[2],bx[3]))>0.02: flags["right_shoe"]=True; break
    head_w = max(20,(person_bbox[2]-person_bbox[0])*0.25); head_h = max(20,(person_bbox[3]-person_bbox[1])*0.18)
    head_box = (hx-head_w, hy-head_h, hx+head_w, hy+head_h)
    for bx in helmet_boxes:
        if iou(head_box,(bx[0],bx[1],bx[2],bx[3]))>0.05: flags["helmet"]=True; break
    if lsx is not None:
        tx1=min([p for p in [lsx,rsx,lhx,rhx] if p is not None])-10
        tx2=max([p for p in [lsx,rsx,lhx,rhx] if p is not None])+10
        ty1=min([p for p in [lsy,rsy,lhy,rhy] if p is not None])-10
        ty2=max([p for p in [lsy,rsy,lhy,rhy] if p is not None])+10
        for bx in vest_boxes:
            if iou((tx1,ty1,tx2,ty2),(bx[0],bx[1],bx[2],bx[3]))>0.05: flags["vest"]=True; break
    else:
        cx=int((person_bbox[0]+person_bbox[2])/2); cy=int((person_bbox[1]+person_bbox[3])/2)
        cb = (cx-30, cy-30, cx+30, cy+30)
        for bx in vest_boxes:
            if iou(cb,(bx[0],bx[1],bx[2],bx[3]))>0.05: flags["vest"]=True; break
    return flags

def draw_pose(frame, landmarks, xoff, yoff, cw, ch):
    for connection in mp_pose.POSE_CONNECTIONS:
        s,e = connection
        lm_s = landmarks.landmark[s]; lm_e = landmarks.landmark[e]
        if getattr(lm_s,"visibility",1.0) < VIS_THRESH or getattr(lm_e,"visibility",1.0) < VIS_THRESH:
            continue
        x1 = int(xoff + lm_s.x * cw); y1 = int(yoff + lm_s.y * ch)
        x2 = int(xoff + lm_e.x * cw); y2 = int(yoff + lm_e.y * ch)
        cv2.line(frame, (x1,y1), (x2,y2), (255,255,255), 3, lineType=cv2.LINE_AA)
        cv2.line(frame, (x1,y1), (x2,y2), (200,200,200), 1, lineType=cv2.LINE_AA)
    for idx, lm in enumerate(landmarks.landmark):
        if getattr(lm,"visibility",0.0) < VIS_THRESH: continue
        cx = int(xoff + lm.x * cw); cy = int(yoff + lm.y * ch)
        if idx in left_indices: col=(230,100,50)
        elif idx in right_indices: col=(10,140,255)
        else: col=(200,200,200)
        cv2.circle(frame, (cx,cy), 6, (255,255,255), -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, (cx,cy), 4, col, -1, lineType=cv2.LINE_AA)

def detect_torso(reader, crop, regset):
    if crop is None or crop.size==0: return None, None, 0.0
    try: rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    except Exception: rgb = crop
    try: res = reader.readtext(rgb, detail=1)
    except Exception:
        try: res = reader.readtext(rgb)
        except Exception: return None, None, 0.0
    best_conf = 0.0; best = None; best_txt = None
    for it in res:
        if len(it)==3: txt = str(it[1]).strip(); conf = float(it[2])
        else: txt = str(it[1]).strip() if len(it)>1 else str(it[0]); conf = float(it[2]) if len(it)>2 else 0.0
        if conf < 0.3: continue
        digs = re.findall(r"\d+", txt)
        if not digs: continue
        for d in digs:
            if d in regset:
                if conf > best_conf: best_conf = conf; best = d; best_txt = txt
            else:
                if conf > best_conf and best is None: best_conf = conf; best = f"UNREG:{d}"; best_txt = txt
    if best is None: return None, None, 0.0
    return best, best_txt, best_conf

def export_csv(wv, uv, last_seen, path="violations_summary.csv"):
    if not wv and not uv: print("no data"); return
    import csv
    with open(path,"w",newline="",encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["id","last_seen_frame","violations"])
        for k,v in wv.items(): w.writerow([k, last_seen.get(k,""), ";".join(sorted(v))])
        for k,v in uv.items(): w.writerow([k, last_seen.get(k,""), ";".join(sorted(v))])
    print("saved", path)

def draw_ppe_boxes(frame, boxes_by_class):
    for cid, bl in boxes_by_class.items():
        col = (0,255,0)
        if cid==0: col=(220,100,30)
        if cid==1: col=(0,215,255)
        if cid==3: col=(255,140,30)
        if cid==4: col=(30,200,30)
        for (x1,y1,x2,y2,c) in bl:
            cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), col, 2)
            cv2.putText(frame, f"{c:.2f}", (int(x1), max(12,int(y1)-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230,230,230), 1, cv2.LINE_AA)

def main():
    reg = ask_ids()
    cap = open_cap(INPUT_PATH)
    if not cap.isOpened(): print("cannot open", INPUT_PATH); return
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    paused=False; scaled=SCALED
    frame_idx=0; last=time.time(); fps_s=0.0; alpha=0.9
    worker_v = defaultdict(set); unreg_v = defaultdict(set); last_seen = {}
    with mp_pose.Pose(**pose_conf) as pose:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    info = np.zeros((200,600,3),dtype=np.uint8)
                    cv2.putText(info, "End of video or read error", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200),2)
                    cv2.putText(info, "Press 'r' to restart, 'q' to quit", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200),2)
                    cv2.imshow(WINDOW, info)
                    k = cv2.waitKey(0) & 0xFF
                    if k == ord('r'):
                        cap.release(); cap = open_cap(INPUT_PATH); frame_idx = 0; last = time.time(); continue
                    else:
                        break
                H,W = frame.shape[:2]
                results = yolo.predict(source=frame, conf=DETECT_CONF, iou=DETECT_IOU, classes=[0], verbose=False)
                dets = []
                if len(results)>0:
                    r = results[0]; boxes = getattr(r,"boxes",None)
                    if boxes is not None and len(boxes)>0:
                        try:
                            xyxy = boxes.xyxy.cpu().numpy(); confs = boxes.conf.cpu().numpy(); cls = boxes.cls.cpu().numpy()
                        except Exception:
                            xyxy = np.array(boxes.xyxy); confs = np.array(boxes.conf); cls = np.array(boxes.cls)
                        for (x1,y1,x2,y2),c,cl in zip(xyxy,confs,cls):
                            if int(cl)!=0: continue
                            dets.append(((float(x1),float(y1),float(x2),float(y2)), float(c)))
                if len(dets) > MAX_PEOPLE: dets = sorted(dets, key=lambda x:x[1], reverse=True)[:MAX_PEOPLE]
                annotated = frame.copy()
                try:
                    ppe_res = yolo_ppe.predict(source=frame, conf=DETECT_CONF, iou=DETECT_IOU, classes=PPE_CLASS_IDS, verbose=False)
                    if len(ppe_res)>0: boxes_by_class = combine_boxes_by_class(ppe_res[0], H, W)
                    else: boxes_by_class = {cid:[] for cid in PPE_CLASS_IDS}
                except Exception:
                    boxes_by_class = {cid:[] for cid in PPE_CLASS_IDS}
                draw_ppe_boxes(annotated, boxes_by_class)
                for pid, (bbox, pconf) in enumerate(dets):
                    x1,y1,x2,y2 = bbox
                    x1i,y1i,x2i,y2i = expand_bbox((x1,y1,x2,y2), W, H)
                    crop = frame[y1i:y2i, x1i:x2i]
                    if crop.size==0: continue
                    res_pose = pose.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    person_bbox = (x1i,y1i,x2i,y2i)
                    violations = []
                    matched_id = None
                    if res_pose and res_pose.pose_landmarks:
                        draw_pose(annotated, res_pose.pose_landmarks, x1i, y1i, x2i-x1i, y2i-y1i)
                        flags = check_ppe(boxes_by_class, person_bbox, res_pose.pose_landmarks, x1i, y1i, x2i-x1i, y2i-y1i)
                        if not flags["helmet"]: violations.append("NO HELMET")
                        if not flags["vest"]: violations.append("NO VEST")
                        if not flags["left_glove"]: violations.append("NO LEFT GLOVE")
                        if not flags["right_glove"]: violations.append("NO RIGHT GLOVE")
                        if not flags["left_shoe"]: violations.append("NO LEFT SHOE")
                        if not flags["right_shoe"]: violations.append("NO RIGHT SHOE")
                        if frame_idx % OCR_INTERVAL == 0:
                            LM = mp_pose.PoseLandmark
                            try:
                                def abs_lm(idx):
                                    lm = res_pose.pose_landmarks.landmark[idx]
                                    return x1i + lm.x * (x2i - x1i), y1i + lm.y * (y2i - y1i), getattr(lm,"visibility",0.0)
                                lsx, lsy, lsv = abs_lm(LM.LEFT_SHOULDER.value)
                                rsx, rsy, rsv = abs_lm(LM.RIGHT_SHOULDER.value)
                                lhx, lhy, lhv = abs_lm(LM.LEFT_HIP.value)
                                rhx, rhy, rhv = abs_lm(LM.RIGHT_HIP.value)
                                xs = [p for p in [lsx,rsx,lhx,rhx] if p is not None]; ys = [p for p in [lsy,rsy,lhy,rhy] if p is not None]
                                if xs and ys:
                                    tx1 = int(max(0, min(xs) - 0.12*(x2i-x1i)))
                                    tx2 = int(min(W-1, max(xs) + 0.12*(x2i-x1i)))
                                    ty1 = int(max(0, min(ys) - 0.15*(y2i-y1i)))
                                    ty2 = int(min(H-1, max(ys) + 0.12*(y2i-y1i)))
                                    torso = frame[ty1:ty2, tx1:tx2]
                                else:
                                    tx1 = int(x1i + 0.15*(x2i-x1i)); tx2 = int(x2i - 0.15*(x2i-x1i))
                                    ty1 = int(y1i + 0.25*(y2i-y1i)); ty2 = int(y2i - 0.15*(y2i-y1i))
                                    torso = frame[ty1:ty2, tx1:tx2]
                            except Exception:
                                torso = None
                            if torso is not None and torso.size>0:
                                mid, txt, conf = detect_torso(ocr_reader, torso, reg)
                                if mid is not None: matched_id = mid
                    else:
                        vest_boxes = boxes_by_class.get(4, [])
                        helmet_boxes = boxes_by_class.get(1, [])
                        if not any(iou(person_bbox,(bx1,by1,bx2,by2))>0.03 for (bx1,by1,bx2,by2,_) in vest_boxes): violations.append("NO VEST")
                        if not any(iou(person_bbox,(bx1,by1,bx2,by2))>0.02 for (bx1,by1,bx2,by2,_) in helmet_boxes): violations.append("NO HELMET")
                        if frame_idx % OCR_INTERVAL == 0:
                            pc = frame[int(person_bbox[1]):int(person_bbox[3]), int(person_bbox[0]):int(person_bbox[2])]
                            if pc.size>0:
                                mid,txt,conf = detect_torso(ocr_reader, pc, reg)
                                if mid is not None: matched_id = mid
                    if matched_id is not None:
                        if matched_id.startswith("UNREG:"): unreg_v[matched_id].update(violations)
                        else: worker_v[matched_id].update(violations); last_seen[matched_id]=frame_idx
                    else:
                        if violations: worker_v["UNKNOWN"].update(violations)
                    cv2.rectangle(annotated, (x1i,y1i), (x2i,y2i), (80,200,80), 2)
                    id_label = matched_id if matched_id is not None else "UNID"
                    cv2.putText(annotated, f"ID:{id_label}", (x1i, max(12,y1i-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240,240,240), 2, cv2.LINE_AA)
                    ty = y2i + 18
                    for v in violations:
                        cv2.putText(annotated, v, (x1i+4, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20,20,220), 2, cv2.LINE_AA); ty += 18
                now = time.time(); dt = now - last; fps = 1.0/dt if dt>0 else 0.0
                fps_s = alpha*fps_s + (1-alpha)*fps if fps_s>0 else fps; last = now
                cv2.putText(annotated, f"Frame:{frame_idx} FPS:{fps_s:.1f}", (12,28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220,220,220), 2, cv2.LINE_AA)
                disp = scale_to_fit(annotated) if scaled else annotated
                cv2.imshow(WINDOW, disp)
                frame_idx += 1
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'): break
            if k == ord('p'): paused = not paused
            if k == ord('s'): scaled = not scaled
            if k == ord('e'): export_csv(worker_v, unreg_v, last_seen)
        cap.release(); cv2.destroyAllWindows()
    export_csv(worker_v, unreg_v, last_seen)

if __name__ == "__main__":
    main()
