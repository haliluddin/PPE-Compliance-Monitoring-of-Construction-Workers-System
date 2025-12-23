# app/decision_logic.py
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

import cv2
import re
import time
import numpy as np
import threading
from collections import defaultdict
from threading import Lock
import logging

log = logging.getLogger(__name__)

# Triton client helpers may not be installed in all environments.
try:
    from tritonclient.http import InferInput, InferRequestedOutput
except Exception:
    InferInput = None
    InferRequestedOutput = None

PPE_CLASS_IDS = [0, 1, 2, 3]
VIS_THRESH = 0.3
CROP_PAD = 0.12

IMPROPER_GLOVE_OVERLAP_THRESH = 0.45
IMPROPER_HELMET_HEAD_IOU_THRESH = 0.40
HELMET_EYE_COVER_MARGIN = 5
HELMET_NOSE_DOWN_FACTOR = 0.30

_mp_lock = Lock()
_mp_pose = None
_left_indices = None
_right_indices = None

_pose_local = threading.local()

PPE_LABELS = {0: "GLOVE", 1: "HELMET", 2: "SHOE", 3: "VEST"}
PPE_COLORS = {0: (200, 100, 200), 1: (10, 200, 200), 2: (180, 120, 40), 3: (240, 180, 30)}

if set(PPE_CLASS_IDS) != set(PPE_LABELS.keys()):
    raise RuntimeError(f"PPE_CLASS_IDS {PPE_CLASS_IDS} and PPE_LABELS keys {list(PPE_LABELS.keys())} mismatch")

def _init_mp_pose():
    global _mp_pose, _left_indices, _right_indices
    if _mp_pose is not None:
        return _mp_pose
    try:
        import mediapipe as mp
    except Exception:
        return None
    with _mp_lock:
        if _mp_pose is not None:
            return _mp_pose
        try:
            mp_pose = mp.solutions.pose
            _mp_pose = mp_pose
            _left_indices = {lm.value for lm in mp_pose.PoseLandmark if "LEFT" in lm.name}
            _right_indices = {lm.value for lm in mp_pose.PoseLandmark if "RIGHT" in lm.name}
            return _mp_pose
        except Exception:
            return None

def init_pose():
    mp = _init_mp_pose()
    if mp is None:
        return None
    inst = getattr(_pose_local, "pose", None)
    if inst is not None:
        return inst
    try:
        inst = mp.Pose(static_image_mode=False, model_complexity=0, smooth_landmarks=True, enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        _pose_local.pose = inst
        return inst
    except Exception:
        try:
            inst = mp.Pose(static_image_mode=False, model_complexity=0, smooth_landmarks=True, enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
            _pose_local.pose = inst
            return inst
        except Exception:
            return None

def get_pose_instance():
    return init_pose()

def iou(A, B):
    xA = max(A[0], B[0]); yA = max(A[1], B[1]); xB = min(A[2], B[2]); yB = min(A[3], B[3])
    interW = max(0.0, xB - xA); interH = max(0.0, yB - yA); inter = interW * interH
    aA = max(0.0, A[2] - A[0]) * max(0.0, A[3] - A[1]); aB = max(0.0, B[2] - B[0]) * max(0.0, B[3] - B[1])
    u = aA + aB - inter
    return inter / u if u > 0 else 0.0

def expand_bbox(b, w, h, pad=CROP_PAD):
    x1, y1, x2, y2 = b
    ww = x2 - x1; hh = y2 - y1
    pad_px = int(max(ww, hh) * pad)
    return max(0, int(x1) - pad_px), max(0, int(y1) - pad_px), min(w - 1, int(x2) + pad_px), min(h - 1, int(y2) + pad_px)

def parse_triton_outputs(outputs, H, W):
    out = {cid: [] for cid in PPE_CLASS_IDS}
    people = []
    if outputs is None:
        return out, people
    arrays = []
    for v in outputs.values():
        if v is None:
            continue
        a = np.array(v)
        if a.ndim == 2 and a.shape[0] > 0:
            arrays.append(a)
    if not arrays:
        return out, people
    arr = arrays[0]
    def _to_box_coords(x1, y1, x2, y2):
        if max(x1, x2, y1, y2) <= 1.01:
            return x1 * W, y1 * H, x2 * W, y2 * H
        return x1, y1, x2, y2
    if arr.ndim == 2 and arr.shape[1] >= 6:
        if arr.shape[1] == 6:
            for row in arr:
                x1, y1, x2, y2, conf, cls = row[:6]
                cls = int(cls)
                x1c, y1c, x2c, y2c = _to_box_coords(x1, y1, x2, y2)
                x1c = max(0.0, float(x1c)); y1c = max(0.0, float(y1c)); x2c = min(W - 1.0, float(x2c)); y2c = min(H - 1.0, float(y2c))
                if cls in PPE_CLASS_IDS:
                    out[cls].append((x1c, y1c, x2c, y2c, float(conf)))
        else:
            for row in arr:
                x, y, w, h, conf = float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])
                probs = row[5:]
                if probs.size > 0:
                    cls = int(np.argmax(probs))
                    cls_conf = float(probs[cls])
                else:
                    cls = int(row[-1]) if row.size > 5 else -1
                    cls_conf = 1.0
                final_conf = float(conf * cls_conf)
                if max(x, y, w, h) <= 1.01:
                    cx = x * W; cy = y * H; ww = w * W; hh = h * H
                else:
                    cx = x; cy = y; ww = w; hh = h
                x1 = cx - ww / 2; y1 = cy - hh / 2; x2 = cx + ww / 2; y2 = cy + hh / 2
                x1c = max(0.0, x1); y1c = max(0.0, y1); x2c = min(W - 1.0, x2); y2c = min(H - 1.0, y2)
                if cls in PPE_CLASS_IDS:
                    out[cls].append((x1c, y1c, x2c, y2c, final_conf))
    return out, people

def abs_lm(landmarks, idx, xoff, yoff, cw, ch):
    lm = landmarks.landmark[idx]
    return (xoff + lm.x * cw, yoff + lm.y * ch, getattr(lm, "visibility", 0.0))

def _area(box):
    x1, y1, x2, y2 = box
    w = max(0.0, x2 - x1); h = max(0.0, y2 - y1)
    return w * h

def _intersection_area(A, B):
    xA = max(A[0], B[0]); yA = max(A[1], B[1]); xB = min(A[2], B[2]); yB = min(A[3], B[3])
    w = max(0.0, xB - xA); h = max(0.0, yB - yA)
    return w * h

def check_ppe(boxes_by_class, person_bbox, landmarks, xoff, yoff, cw, ch):
    mp_pose = _init_mp_pose()
    LM = mp_pose.PoseLandmark if mp_pose is not None else None
    flags = {
        "helmet": False, "vest": False,
        "left_glove": False, "right_glove": False,
        "left_shoe": False, "right_shoe": False,
        "improper_helmet": False,
        "improper_left_glove": False,
        "improper_right_glove": False
    }
    head_pts = []
    if LM is not None:
        candidates = [LM.NOSE.value, LM.LEFT_EYE.value, LM.RIGHT_EYE.value, LM.LEFT_EAR.value, LM.RIGHT_EAR.value]
    else:
        candidates = list(range(5))
    for idx in candidates:
        try:
            lm = landmarks.landmark[idx]
        except Exception:
            continue
        if getattr(lm, "visibility", 0.0) >= VIS_THRESH:
            head_pts.append((xoff + lm.x * cw, yoff + lm.y * ch))
    if head_pts:
        hx = int(np.mean([p[0] for p in head_pts])); hy = int(np.mean([p[1] for p in head_pts]))
    else:
        hx = int((person_bbox[0] + person_bbox[2]) / 2); hy = int(person_bbox[1] + (person_bbox[3] - person_bbox[1]) * 0.12)
    try:
        lwx, lwy, lwv = abs_lm(landmarks, LM.LEFT_WRIST.value, xoff, yoff, cw, ch) if LM is not None else (0, 0, 0)
        rwx, rwy, rwv = abs_lm(landmarks, LM.RIGHT_WRIST.value, xoff, yoff, cw, ch) if LM is not None else (0, 0, 0)
        lax, lay, lav = abs_lm(landmarks, LM.LEFT_ANKLE.value, xoff, yoff, cw, ch) if LM is not None else (0, 0, 0)
        rax, ray, rav = abs_lm(landmarks, LM.RIGHT_ANKLE.value, xoff, yoff, cw, ch) if LM is not None else (0, 0, 0)
    except Exception:
        lwx = lwy = lwv = rwx = rwy = rwv = lax = lay = lav = rax = ray = rav = 0.0
    try:
        if LM is not None:
            lsx, lsy, lsv = abs_lm(landmarks, LM.LEFT_SHOULDER.value, xoff, yoff, cw, ch)
            rsx, rsy, rsv = abs_lm(landmarks, LM.RIGHT_SHOULDER.value, xoff, yoff, cw, ch)
            lhx, lhy, lhv = abs_lm(landmarks, LM.LEFT_HIP.value, xoff, yoff, cw, ch)
            rhx, rhy, rhv = abs_lm(landmarks, LM.RIGHT_HIP.value, xoff, yoff, cw, ch)
            try:
                nose_x, nose_y, nose_v = abs_lm(landmarks, LM.NOSE.value, xoff, yoff, cw, ch)
            except Exception:
                nose_x = nose_y = nose_v = None
            try:
                leye_x, leye_y, leye_v = abs_lm(landmarks, LM.LEFT_EYE.value, xoff, yoff, cw, ch)
            except Exception:
                leye_x = leye_y = leye_v = None
            try:
                reye_x, reye_y, reye_v = abs_lm(landmarks, LM.RIGHT_EYE.value, xoff, yoff, cw, ch)
            except Exception:
                reye_x = reye_y = reye_v = None
            try:
                lear_x, lear_y, lear_v = abs_lm(landmarks, LM.LEFT_EAR.value, xoff, yoff, cw, ch)
            except Exception:
                lear_x = lear_y = lear_v = None
            try:
                rear_x, rear_y, rear_v = abs_lm(landmarks, LM.RIGHT_EAR.value, xoff, yoff, cw, ch)
            except Exception:
                rear_x = rear_y = rear_v = None
        else:
            lsx = rsx = lhx = rhx = lsy = rsy = lhy = rhy = None
            lsv = rsv = lhv = rhv = 0.0
            nose_x = nose_y = nose_v = None
            leye_x = leye_y = leye_v = None
            reye_x = reye_y = reye_v = None
            lear_x = lear_y = lear_v = None
            rear_x = rear_y = rear_v = None
    except Exception:
        lsx = rsx = lhx = rhx = lsy = rsy = lhy = rhy = None; lsv = rsv = lhv = rhv = 0.0
        nose_x = nose_y = nose_v = None
        leye_x = leye_y = leye_v = None
        reye_x = reye_y = reye_v = None
        lear_x = lear_y = lear_v = None
        rear_x = rear_y = rear_v = None
    glove_boxes = boxes_by_class.get(0, [])
    helmet_boxes = boxes_by_class.get(1, [])
    shoe_boxes = boxes_by_class.get(2, [])
    vest_boxes = boxes_by_class.get(3, [])
    wrist_radius = max(16, int((person_bbox[2] - person_bbox[0]) * 0.04))
    matched_left_glove_box = None
    matched_right_glove_box = None
    if lwv >= VIS_THRESH:
        wb = (lwx - wrist_radius, lwy - wrist_radius, lwx + wrist_radius, lwy + wrist_radius)
        for bx in glove_boxes:
            if iou(wb, (bx[0], bx[1], bx[2], bx[3])) > 0.02:
                flags["left_glove"] = True
                matched_left_glove_box = (bx[0], bx[1], bx[2], bx[3])
                break
    if rwv >= VIS_THRESH:
        wb = (rwx - wrist_radius, rwy - wrist_radius, rwx + wrist_radius, rwy + wrist_radius)
        for bx in glove_boxes:
            if iou(wb, (bx[0], bx[1], bx[2], bx[3])) > 0.02:
                flags["right_glove"] = True
                matched_right_glove_box = (bx[0], bx[1], bx[2], bx[3])
                break
    ankle_radius = max(20, int((person_bbox[3] - person_bbox[1]) * 0.05))
    if lav >= 0.05:
        ab = (lax - ankle_radius, lay - ankle_radius, lax + ankle_radius, lay + ankle_radius)
        for bx in shoe_boxes:
            if iou(ab, (bx[0], bx[1], bx[2], bx[3])) > 0.02:
                flags["left_shoe"] = True
                break
    if rav >= 0.05:
        ab = (rax - ankle_radius, ray - ankle_radius, rax + ankle_radius, ray + ankle_radius)
        for bx in shoe_boxes:
            if iou(ab, (bx[0], bx[1], bx[2], bx[3])) > 0.02:
                flags["right_shoe"] = True
                break
    head_w = max(20, (person_bbox[2] - person_bbox[0]) * 0.25); head_h = max(20, (person_bbox[3] - person_bbox[1]) * 0.18)
    head_box = (hx - head_w, hy - head_h, hx + head_w, hy + head_h)
    matched_helmet_box = None
    for bx in helmet_boxes:
        if iou(head_box, (bx[0], bx[1], bx[2], bx[3])) > 0.05:
            flags["helmet"] = True
            matched_helmet_box = (bx[0], bx[1], bx[2], bx[3])
            break
    if lsx is not None:
        xs = [p for p in [lsx, rsx, lhx, rhx] if p is not None]
        ys = [p for p in [lsy, rsy, lhy, rhy] if p is not None]
        if xs and ys:
            tx1 = min(xs) - 10
            tx2 = max(xs) + 10
            ty1 = min(ys) - 10
            ty2 = max(ys) + 10
            for bx in vest_boxes:
                if iou((tx1, ty1, tx2, ty2), (bx[0], bx[1], bx[2], bx[3])) > 0.05:
                    flags["vest"] = True
                    break
    else:
        cx = int((person_bbox[0] + person_bbox[2]) / 2); cy = int((person_bbox[1] + person_bbox[3]) / 2)
        cb = (cx - 30, cy - 30, cx + 30, cy + 30)
        for bx in vest_boxes:
            if iou(cb, (bx[0], bx[1], bx[2], bx[3])) > 0.05:
                flags["vest"] = True
                break
    def _make_wrist_region(wx, wy, ex=None, ey=None):
        if ex is not None and ey is not None:
            x1 = min(wx, ex) - int(wrist_radius * 1.0)
            x2 = max(wx, ex) + int(wrist_radius * 1.0)
            y1 = min(wy, ey) - int(wrist_radius * 1.0)
            y2 = max(wy, ey) + int(wrist_radius * 1.0)
        else:
            x1 = wx - wrist_radius * 2
            x2 = wx + wrist_radius * 2
            y1 = wy - wrist_radius * 2
            y2 = wy + wrist_radius * 2
        return (x1, y1, x2, y2)
    if matched_left_glove_box is not None and LM is not None:
        try:
            lex, ley, lev = abs_lm(landmarks, LM.LEFT_ELBOW.value, xoff, yoff, cw, ch)
            expected = _make_wrist_region(lwx, lwy, lex if lev >= VIS_THRESH else None, ley if lev >= VIS_THRESH else None)
        except Exception:
            expected = _make_wrist_region(lwx, lwy)
        glue_area = _area(matched_left_glove_box)
        if glue_area > 0:
            inter = _intersection_area(matched_left_glove_box, expected)
            frac = inter / glue_area
            if frac < IMPROPER_GLOVE_OVERLAP_THRESH:
                flags["improper_left_glove"] = True
    if matched_right_glove_box is not None and LM is not None:
        try:
            rex, rey, rev = abs_lm(landmarks, LM.RIGHT_ELBOW.value, xoff, yoff, cw, ch)
            expected = _make_wrist_region(rwx, rwy, rex if rev >= VIS_THRESH else None, rey if rev >= VIS_THRESH else None)
        except Exception:
            expected = _make_wrist_region(rwx, rwy)
        glue_area = _area(matched_right_glove_box)
        if glue_area > 0:
            inter = _intersection_area(matched_right_glove_box, expected)
            frac = inter / glue_area
            if frac < IMPROPER_GLOVE_OVERLAP_THRESH:
                flags["improper_right_glove"] = True
    if matched_helmet_box is not None:
        hb = matched_helmet_box
        helmet_area = _area(hb)
        if helmet_area > 0:
            head_inter = _intersection_area(hb, head_box)
            head_overlap_frac = head_inter / helmet_area
            if head_overlap_frac < IMPROPER_HELMET_HEAD_IOU_THRESH:
                flags["improper_helmet"] = True
        eye_y = None
        leye_v = locals().get("leye_v", None)
        reye_v = locals().get("reye_v", None)
        leye_y = locals().get("leye_y", None)
        reye_y = locals().get("reye_y", None)
        nose_v = locals().get("nose_v", None)
        nose_y = locals().get("nose_y", None)
        if leye_v is not None and leye_v >= VIS_THRESH and reye_v is not None and reye_v >= VIS_THRESH:
            eye_y = min(leye_y, reye_y)
        elif leye_v is not None and leye_v >= VIS_THRESH:
            eye_y = leye_y
        elif reye_v is not None and reye_v >= VIS_THRESH:
            eye_y = reye_y
        if eye_y is not None:
            if hb[1] > eye_y + HELMET_EYE_COVER_MARGIN:
                flags["improper_helmet"] = True
        if nose_v is not None and nose_v >= VIS_THRESH:
            if hb[3] > nose_y + (head_h * HELMET_NOSE_DOWN_FACTOR):
                flags["improper_helmet"] = True
    return flags

def draw_pose(frame, landmarks, xoff, yoff, cw, ch):
    mp_pose = _init_mp_pose()
    if mp_pose is None:
        return
    for connection in mp_pose.POSE_CONNECTIONS:
        s, e = connection
        lm_s = landmarks.landmark[s]; lm_e = landmarks.landmark[e]
        if getattr(lm_s, "visibility", 1.0) < VIS_THRESH or getattr(lm_e, "visibility", 1.0) < VIS_THRESH:
            continue
        x1 = int(xoff + lm_s.x * cw); y1 = int(yoff + lm_s.y * ch)
        x2 = int(xoff + lm_e.x * cw); y2 = int(yoff + lm_e.y * ch)
        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 3, lineType=cv2.LINE_AA)
        cv2.line(frame, (x1, y1), (x2, y2), (200, 200, 200), 1, lineType=cv2.LINE_AA)
    global _left_indices, _right_indices
    for idx, lm in enumerate(landmarks.landmark):
        if getattr(lm, "visibility", 0.0) < VIS_THRESH: continue
        cx = int(xoff + lm.x * cw); cy = int(yoff + lm.y * ch)
        if _left_indices is None or _right_indices is None:
            _init_mp_pose()
        if _left_indices and idx in _left_indices:
            col = (230, 100, 50)
        elif _right_indices and idx in _right_indices:
            col = (10, 140, 255)
        else:
            col = (200, 200, 200)
        cv2.circle(frame, (cx, cy), 6, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 4, col, -1, lineType=cv2.LINE_AA)

def detect_torso(ocr_reader, crop, regset):
    if crop is None or crop.size == 0:
        return None, None, 0.0
    try:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    except Exception:
        rgb = crop
    try:
        res = ocr_reader.readtext(rgb, detail=1)
    except Exception:
        try:
            res = ocr_reader.readtext(rgb)
        except Exception:
            return None, None, 0.0
    best_conf = 0.0; best = None; best_txt = None
    for it in res:
        if len(it) == 3:
            txt = str(it[1]).strip(); conf = float(it[2])
        else:
            txt = str(it[1]).strip() if len(it) > 1 else str(it[0]); conf = float(it[2]) if len(it) > 2 else 0.0
        if conf < 0.3: continue
        digs = re.findall(r"\d+", txt)
        if not digs: continue
        for d in digs:
            if d in regset:
                if conf > best_conf:
                    best_conf = conf; best = d; best_txt = txt
            else:
                if conf > best_conf and best is None:
                    best_conf = conf; best = f"UNREG:{d}"; best_txt = txt
    if best is None:
        return None, None, 0.0
    return best, best_txt, best_conf

def process_frame(frame, triton_client=None, triton_model_name=None, input_name=None, output_names=None, triton_outputs=None, ocr_reader=None, regset=None, pose_instance=None, person_boxes=None, draw_labels=True):
    H, W = frame.shape[:2]

    # If caller didn't provide triton_outputs but we have a triton client and model metadata,
    # attempt Triton inference — but only if InferInput helpers are available.
    if (triton_outputs is None and triton_client is not None and triton_model_name is not None and input_name is not None and output_names is not None):
        if InferInput is None or InferRequestedOutput is None:
            log.warning("Triton helpers not available — skipping Triton inference in decision_logic.process_frame")
            triton_outputs = None
        else:
            try:
                img = cv2.resize(frame, (416, 416))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
                img = np.transpose(img, (2, 0, 1))[None, ...]
                inp = InferInput(input_name, img.shape, "FP32")
                inp.set_data_from_numpy(img)
                outputs = [InferRequestedOutput(n) for n in output_names]
                res = triton_client.infer(triton_model_name, inputs=[inp], outputs=outputs)
                triton_outputs = {n: res.as_numpy(n) for n in output_names}
            except Exception:
                log.exception("Triton inference failed in process_frame; falling back to local parsing")
                triton_outputs = None

    boxes_by_class, parsed_people = parse_triton_outputs(triton_outputs, H, W)
    if person_boxes is None:
        person_boxes = parsed_people
    annotated = frame.copy()
    people_results = []
    if pose_instance is None:
        pose_instance = get_pose_instance()
    for idx, pb in enumerate(person_boxes):
        if isinstance(pb, tuple) and len(pb) >= 4:
            x1, y1, x2, y2 = int(pb[0]), int(pb[1]), int(pb[2]), int(pb[3])
        else:
            continue
        x1i, y1i, x2i, y2i = expand_bbox((x1, y1, x2, y2), W, H)
        crop = frame[y1i:y2i, x1i:x2i]
        if crop is None or crop.size == 0:
            continue
        res_pose = None
        if pose_instance is not None:
            try:
                res_pose = pose_instance.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            except Exception:
                res_pose = None
        person_bbox = (x1i, y1i, x2i, y2i)
        violations = []
        matched_id = None
        matched_id_conf = 0.0
        if res_pose and getattr(res_pose, "pose_landmarks", None):
            draw_pose(annotated, res_pose.pose_landmarks, x1i, y1i, x2i - x1i, y2i - y1i)
            try:
                flags = check_ppe(boxes_by_class, person_bbox, res_pose.pose_landmarks, x1i, y1i, x2i - x1i, y2i - y1i)
            except Exception:
                log.exception("check_ppe failed")
                flags = {}
            if not flags.get("helmet", False):
                violations.append("NO HELMET")
            else:
                if flags.get("improper_helmet"):
                    violations.append("IMPROPER HELMET")
            if not flags.get("vest", False):
                violations.append("NO VEST")
            if not flags.get("left_glove", False):
                violations.append("NO LEFT GLOVE")
            else:
                if flags.get("improper_left_glove"):
                    violations.append("IMPROPER LEFT GLOVE")
            if not flags.get("right_glove", False):
                violations.append("NO RIGHT GLOVE")
            else:
                if flags.get("improper_right_glove"):
                    violations.append("IMPROPER RIGHT GLOVE")
            if not flags.get("left_shoe", False):
                violations.append("NO LEFT SHOE")
            if not flags.get("right_shoe", False):
                violations.append("NO RIGHT SHOE")
            if ocr_reader is not None:
                mp_pose = _init_mp_pose()
                LM = mp_pose.PoseLandmark if mp_pose is not None else None
                try:
                    def _abs_lm(idx):
                        lm = res_pose.pose_landmarks.landmark[idx]
                        return x1i + lm.x * (x2i - x1i), y1i + lm.y * (y2i - y1i), getattr(lm, "visibility", 0.0)
                    if LM is not None:
                        lsx, lsy, lsv = _abs_lm(LM.LEFT_SHOULDER.value)
                        rsx, rsy, rsv = _abs_lm(LM.RIGHT_SHOULDER.value)
                        lhx, lhy, lhv = _abs_lm(LM.LEFT_HIP.value)
                        rhx, rhy, rhv = _abs_lm(LM.RIGHT_HIP.value)
                    else:
                        lsx = rsx = lhx = rhx = None
                    xs = [p for p in [lsx, rsx, lhx, rhx] if p is not None]; ys = [p for p in [lsy, rsy, lhy, rhy] if p is not None]
                    if xs and ys:
                        tx1 = int(max(0, min(xs) - 0.12 * (x2i - x1i)))
                        tx2 = int(min(W - 1, max(xs) + 0.12 * (x2i - x1i)))
                        ty1 = int(max(0, min(ys) - 0.15 * (y2i - y1i)))
                        ty2 = int(min(H - 1, max(ys) + 0.12 * (y2i - y1i)))
                        torso = frame[ty1:ty2, tx1:tx2]
                    else:
                        tx1 = int(x1i + 0.15 * (x2i - x1i)); tx2 = int(x2i - 0.15 * (x2i - x1i))
                        ty1 = int(y1i + 0.25 * (y2i - y1i)); ty2 = int(y2i - 0.15 * (y2i - y1i))
                        torso = frame[ty1:ty2, tx1:tx2]
                except Exception:
                    torso = None
                if torso is not None and torso.size > 0:
                    try:
                        mid, txt, conf = detect_torso(ocr_reader, torso, regset or set())
                        if mid is not None:
                            matched_id = mid
                            try:
                                matched_id_conf = float(conf) if conf is not None else 0.0
                            except Exception:
                                matched_id_conf = 0.0
                    except Exception:
                        log.exception("detect_torso failed")
        else:
            vest_boxes = boxes_by_class.get(3, [])
            helmet_boxes = boxes_by_class.get(1, [])
            try:
                if not any(iou(person_bbox, (bx1, by1, bx2, by2)) > 0.03 for (bx1, by1, bx2, by2, _) in vest_boxes):
                    violations.append("NO VEST")
                if not any(iou(person_bbox, (bx1, by1, bx2, by2)) > 0.02 for (bx1, by1, bx2, by2, _) in helmet_boxes):
                    violations.append("NO HELMET")
            except Exception:
                log.exception("fallback bbox checks failed")
            if ocr_reader is not None:
                pc = frame[int(person_bbox[1]):int(person_bbox[3]), int(person_bbox[0]):int(person_bbox[2])]
                if pc.size > 0:
                    try:
                        mid, txt, conf = detect_torso(ocr_reader, pc, regset or set())
                        if mid is not None:
                            matched_id = mid
                            try:
                                matched_id_conf = float(conf) if conf is not None else 0.0
                            except Exception:
                                matched_id_conf = 0.0
                    except Exception:
                        log.exception("detect_torso fallback failed")

        if matched_id is not None:
            id_label = matched_id
        else:
            id_label = "UNID"
        severity_color = (80, 200, 80)
        vio_set = set([v.upper() for v in violations])
        if not vio_set:
            severity_color = (80, 200, 80)
        else:
            if len(vio_set) >= 2:
                severity_color = (0, 0, 255)
            elif vio_set == {"NO HELMET"}:
                severity_color = (0, 140, 255)
            elif vio_set == {"NO VEST"}:
                severity_color = (0, 215, 255)
            else:
                severity_color = (0, 140, 255)
        cv2.rectangle(annotated, (x1i, y1i), (x2i, y2i), severity_color, 2)
        if draw_labels:
            cv2.putText(annotated, f"ID:{id_label}", (x1i, max(12, y1i - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2, cv2.LINE_AA)
            ty = y2i + 18
            for v in violations:
                cv2.putText(annotated, v, (x1i + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 220), 2, cv2.LINE_AA); ty += 18
        people_results.append({"bbox": person_bbox, "id": id_label, "id_conf": matched_id_conf, "violations": violations})
    for cls_id, boxes in boxes_by_class.items():
        if not boxes:
            continue
        label = PPE_LABELS.get(cls_id, f"CLS{cls_id}")
        col = PPE_COLORS.get(cls_id, (200, 200, 200))
        for bx in boxes:
            try:
                x1b, y1b, x2b, y2b = int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3])
                conf = float(bx[4]) if len(bx) > 4 else None
                x1b = max(0, min(x1b, W-1)); y1b = max(0, min(y1b, H-1))
                x2b = max(0, min(x2b, W-1)); y2b = max(0, min(y2b, H-1))
                cv2.rectangle(annotated, (x1b, y1b), (x2b, y2b), col, 2)
                if draw_labels:
                    txt = f"{label}" + (f" {conf:.2f}" if conf is not None else "")
                    cv2.putText(annotated, txt, (x1b + 3, max(12, y1b + 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240,240,240), 1, cv2.LINE_AA)
            except Exception:
                continue
    result = {"frame_h": H, "frame_w": W, "boxes_by_class": boxes_by_class, "people": people_results, "annotated_bgr": annotated}
    return result
