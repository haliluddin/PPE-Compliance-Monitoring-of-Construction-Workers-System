# ppe_pose_integration.py
import cv2
import mediapipe as mp
import numpy as np
import sys
import time
from pathlib import Path
from ultralytics import YOLO

# ----------------- USER CONFIG -----------------
INPUT_PATH = "4271760-hd_1920_1080_30fps.mp4"
PERSON_MODEL = "yolov8n.pt"  # your person detector (same as your original)
PPE_MODEL = "runs/segment/ppe_yolov8n_seg_run1/weights/best.pt"  # your trained instance-segmentation model
DETECT_CONF = 0.35
DETECT_IOU = 0.45
MAX_PEOPLE = 30
CROP_PAD = 0.12
VISIBILITY_THRESHOLD = 0.3

# IoU thresholds for assigning PPE -> human regions
TH_HELMET = 0.35
TH_VEST = 0.30
TH_GLOVE = 0.25
TH_SHOE = 0.25

# class mapping for your PPE model: ['glove','helmet','person','shoe','vest']
PPE_CLASS_NAMES = {0: "glove", 1: "helmet", 2: "person", 3: "shoe", 4: "vest"}

# display
WINDOW_NAME = "PPE + Pose"
SCALED_DISPLAY = True
SCALE_MAX_WIDTH = 1280
SCALE_MAX_HEIGHT = 720

# colors
WHITE = (255, 255, 255)
VIOLATION_COLOR = (0, 0, 255)
OK_COLOR = (0, 200, 0)
BOX_COLOR = (80, 200, 80)

# ------------------------------------------------

# load models
print("Loading YOLO models...")
person_yolo = YOLO(PERSON_MODEL)
ppe_yolo = YOLO(PPE_MODEL)

mp_pose = mp.solutions.pose
pose_conf = dict(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# left/right sets for drawing
left_indices = {lm.value for lm in mp_pose.PoseLandmark if "LEFT" in lm.name}
right_indices = {lm.value for lm in mp_pose.PoseLandmark if "RIGHT" in lm.name}

def expand_bbox(bbox, img_w, img_h, pad=CROP_PAD):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    pad_pixels = int(max(w, h) * pad)
    x1e = max(0, int(x1) - pad_pixels)
    y1e = max(0, int(y1) - pad_pixels)
    x2e = min(img_w - 1, int(x2) + pad_pixels)
    y2e = min(img_h - 1, int(y2) + pad_pixels)
    return x1e, y1e, x2e, y2e

def bbox_iou(boxA, boxB):
    # boxes are [x1,y1,x2,y2] floats
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    inter = interW * interH
    areaA = max(1e-6, (boxA[2]-boxA[0])*(boxA[3]-boxA[1]))
    areaB = max(1e-6, (boxB[2]-boxB[0])*(boxB[3]-boxB[1]))
    union = areaA + areaB - inter
    return inter/union if union>0 else 0.0

def landmarks_to_bbox(landmarks, crop_shape, landmark_indices, pad_ratio=0.2):
    # landmarks: mediapipe landmark list (normalized relative to crop)
    h, w = crop_shape[:2]
    xs, ys = [], []
    for idx in landmark_indices:
        lm = landmarks.landmark[idx]
        xs.append(lm.x * w)
        ys.append(lm.y * h)
    if not xs:
        return None
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    pad_w = (x2 - x1) * pad_ratio
    pad_h = (y2 - y1) * pad_ratio
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(w, x2 + pad_w)
    y2 = min(h, y2 + pad_h)
    return [x1, y1, x2, y2]

def draw_person_pose_on_frame(frame, landmarks, x_offset, y_offset, crop_w, crop_h, visibility_threshold=VISIBILITY_THRESHOLD):
    for connection in mp_pose.POSE_CONNECTIONS:
        s = connection[0]
        e = connection[1]
        lm_s = landmarks.landmark[s]
        lm_e = landmarks.landmark[e]
        if getattr(lm_s, "visibility", 1.0) < visibility_threshold or getattr(lm_e, "visibility", 1.0) < visibility_threshold:
            continue
        x1 = int(x_offset + lm_s.x * crop_w)
        y1 = int(y_offset + lm_s.y * crop_h)
        x2 = int(x_offset + lm_e.x * crop_w)
        y2 = int(y_offset + lm_e.y * crop_h)
        cv2.line(frame, (x1, y1), (x2, y2), WHITE, thickness=7, lineType=cv2.LINE_AA)
        cv2.line(frame, (x1, y1), (x2, y2), (220, 220, 220), thickness=2, lineType=cv2.LINE_AA)

    for idx, lm in enumerate(landmarks.landmark):
        if lm.visibility < visibility_threshold:
            continue
        cx = int(x_offset + lm.x * crop_w)
        cy = int(y_offset + lm.y * crop_h)
        if idx in left_indices:
            color = (230, 100, 50)
        elif idx in right_indices:
            color = (10, 140, 255)
        else:
            color = (200, 200, 200)
        cv2.circle(frame, (cx, cy), 8, WHITE, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 5, color, thickness=-1, lineType=cv2.LINE_AA)

def scale_to_fit(img, max_w=SCALE_MAX_WIDTH, max_h=SCALE_MAX_HEIGHT):
    h, w = img.shape[:2]
    scale = min(1.0, max_w / w, max_h / h)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img

def open_source(src):
    try:
        idx = int(src)
        return cv2.VideoCapture(idx)
    except Exception:
        return cv2.VideoCapture(src)

def main():
    cap = open_source(INPUT_PATH)
    if not cap.isOpened():
        print(f"ERROR: cannot open source '{INPUT_PATH}'")
        sys.exit(1)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    paused = False
    scaled = SCALED_DISPLAY
    frame_idx = 0

    last_time = time.time()
    fps_smooth = 0.0
    alpha = 0.9

    with mp_pose.Pose(**pose_conf) as pose:
        print("PPE+Pose monitor started. Controls: q=quit, p=pause/resume, s=toggle scaled display")
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("End of source.")
                    break
                img_h, img_w = frame.shape[:2]

                # 1) detect people using person_yolo (same as before)
                person_results = person_yolo.predict(source=frame, conf=DETECT_CONF, iou=DETECT_IOU, classes=[0], verbose=False)
                detections = []
                if len(person_results) > 0:
                    r = person_results[0]
                    boxes = getattr(r, "boxes", None)
                    if boxes is not None and len(boxes) > 0:
                        xyxy = boxes.xyxy.cpu().numpy()
                        confs = boxes.conf.cpu().numpy()
                        cls = boxes.cls.cpu().numpy()
                        for (x1, y1, x2, y2), c, cl in zip(xyxy, confs, cls):
                            # class filtering already via classes=[0], but keep safe
                            detections.append(((float(x1), float(y1), float(x2), float(y2)), float(c)))

                # cap people
                if len(detections) > MAX_PEOPLE:
                    detections = sorted(detections, key=lambda x: x[1], reverse=True)[:MAX_PEOPLE]

                # 2) detect PPE items using PPE model (full-frame)
                ppe_results = ppe_yolo.predict(source=frame, conf=DETECT_CONF, iou=DETECT_IOU, verbose=False)
                ppe_dets = []  # list of dicts: {bbox, class_id, name, conf}
                if len(ppe_results) > 0:
                    rr = ppe_results[0]
                    boxes = getattr(rr, "boxes", None)
                    if boxes is not None and len(boxes) > 0:
                        xyxy = boxes.xyxy.cpu().numpy()
                        confs = boxes.conf.cpu().numpy()
                        cls = boxes.cls.cpu().numpy().astype(int)
                        for (x1, y1, x2, y2), conf, cid in zip(xyxy, confs, cls):
                            # ignore 'person' from ppe model (we use separate person detector)
                            if PPE_CLASS_NAMES.get(cid, None) == "person":
                                continue
                            ppe_dets.append({
                                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                                "class_id": int(cid),
                                "name": PPE_CLASS_NAMES.get(int(cid), str(cid)),
                                "conf": float(conf)
                            })

                annotated = frame.copy()

                # For each person: crop, run MediaPipe, derive regions, map PPE detections
                summary = []
                for (bbox, pconf) in detections:
                    x1, y1, x2, y2 = bbox
                    x1i, y1i, x2i, y2i = expand_bbox((x1, y1, x2, y2), img_w, img_h)
                    crop = frame[y1i:y2i, x1i:x2i]
                    if crop.size == 0:
                        continue
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    res_pose = pose.process(crop_rgb)

                    # default: all violations true -> we will clear them if found
                    violations = {
                        "helmet": True,
                        "left_glove": True,
                        "right_glove": True,
                        "left_shoe": True,
                        "right_shoe": True,
                        "vest": True
                    }

                    # gather person regions (in frame coords)
                    person_regions = {"head": None, "torso": None, "left_hand": None, "right_hand": None, "left_foot": None, "right_foot": None}
                    if res_pose and res_pose.pose_landmarks:
                        lm = res_pose.pose_landmarks
                        from mediapipe.python.solutions.pose import PoseLandmark as PL
                        # head: nose, eyes, ears
                        head_bb = landmarks_to_bbox(lm, crop.shape, [PL.NOSE.value, PL.LEFT_EYE.value, PL.RIGHT_EYE.value, PL.LEFT_EAR.value, PL.RIGHT_EAR.value], pad_ratio=0.5)
                        torso_bb = landmarks_to_bbox(lm, crop.shape, [PL.LEFT_SHOULDER.value, PL.RIGHT_SHOULDER.value, PL.LEFT_HIP.value, PL.RIGHT_HIP.value], pad_ratio=0.3)
                        left_hand_bb = landmarks_to_bbox(lm, crop.shape, [PL.LEFT_WRIST.value, PL.LEFT_INDEX.value], pad_ratio=0.3)
                        right_hand_bb = landmarks_to_bbox(lm, crop.shape, [PL.RIGHT_WRIST.value, PL.RIGHT_INDEX.value], pad_ratio=0.3)
                        # feet: use ankle and heel or foot index
                        left_foot_bb = landmarks_to_bbox(lm, crop.shape, [PL.LEFT_ANKLE.value, PL.LEFT_HEEL.value, PL.LEFT_FOOT_INDEX.value], pad_ratio=0.3)
                        right_foot_bb = landmarks_to_bbox(lm, crop.shape, [PL.RIGHT_ANKLE.value, PL.RIGHT_HEEL.value, PL.RIGHT_FOOT_INDEX.value], pad_ratio=0.3)

                        # convert to frame coords
                        def to_frame(bb_crop):
                            if bb_crop is None:
                                return None
                            return [bb_crop[0] + x1i, bb_crop[1] + y1i, bb_crop[2] + x1i, bb_crop[3] + y1i]

                        person_regions["head"] = to_frame(head_bb)
                        person_regions["torso"] = to_frame(torso_bb)
                        person_regions["left_hand"] = to_frame(left_hand_bb)
                        person_regions["right_hand"] = to_frame(right_hand_bb)
                        person_regions["left_foot"] = to_frame(left_foot_bb)
                        person_regions["right_foot"] = to_frame(right_foot_bb)

                        # draw pose on annotated image
                        draw_person_pose_on_frame(annotated, lm, x1i, y1i, x2i - x1i, y2i - y1i)
                    else:
                        # if no landmarks, we can still try to detect PPE items that overlap person bbox
                        person_regions["torso"] = [x1i, y1i + (y2i - y1i) * 0.25, x2i, y2i - (y2i - y1i) * 0.1]  # rough torso
                        person_regions["head"] = [x1i + (x2i-x1i)*0.25, y1i, x2i - (x2i-x1i)*0.25, y1i + (y2i-y1i)*0.25]
                        person_regions["left_hand"] = None
                        person_regions["right_hand"] = None
                        person_regions["left_foot"] = None
                        person_regions["right_foot"] = None

                    # find PPE detections that overlap this person bbox somewhat (filter)
                    assigned_ppe = [d for d in ppe_dets if bbox_iou(d["bbox"], [x1, y1, x2, y2]) > 0.02]

                    # Evaluate each PPE type
                    # Helmet
                    helmets = [d for d in assigned_ppe if d["name"] == "helmet"]
                    if person_regions["head"] is not None and len(helmets) > 0:
                        # treat helmet present if any helmet IoU with head bbox > TH_HELMET
                        if any(bbox_iou(h["bbox"], person_regions["head"]) > TH_HELMET for h in helmets):
                            violations["helmet"] = False

                    # Vest (torso)
                    vests = [d for d in assigned_ppe if d["name"] == "vest"]
                    if person_regions["torso"] is not None and len(vests) > 0:
                        if any(bbox_iou(v["bbox"], person_regions["torso"]) > TH_VEST for v in vests):
                            violations["vest"] = False

                    # Gloves: match by left/right hand region (if not available fall back to any glove overlapping person bbox)
                    gloves = [d for d in assigned_ppe if d["name"] == "glove"]
                    # left glove
                    if person_regions["left_hand"] is not None:
                        if any(bbox_iou(g["bbox"], person_regions["left_hand"]) > TH_GLOVE for g in gloves):
                            violations["left_glove"] = False
                    else:
                        # fallback: any glove on left half of person bbox
                        cx = (x1 + x2) / 2.0
                        if any((g["bbox"][0] + g["bbox"][2]) / 2.0 < cx for g in gloves):
                            violations["left_glove"] = False
                    # right glove
                    if person_regions["right_hand"] is not None:
                        if any(bbox_iou(g["bbox"], person_regions["right_hand"]) > TH_GLOVE for g in gloves):
                            violations["right_glove"] = False
                    else:
                        cx = (x1 + x2) / 2.0
                        if any((g["bbox"][0] + g["bbox"][2]) / 2.0 > cx for g in gloves):
                            violations["right_glove"] = False

                    # Shoes: similar logic on feet
                    shoes = [d for d in assigned_ppe if d["name"] == "shoe"]
                    if person_regions["left_foot"] is not None:
                        if any(bbox_iou(s["bbox"], person_regions["left_foot"]) > TH_SHOE for s in shoes):
                            violations["left_shoe"] = False
                    else:
                        # fallback by y position (shoe boxes near bottom of person bbox AND on left half)
                        cy_threshold = y2i - (y2i - y1i) * 0.15
                        cx_mid = (x1 + x2) / 2.0
                        if any(((s["bbox"][1] + s["bbox"][3]) / 2.0) > cy_threshold and ((s["bbox"][0] + s["bbox"][2]) / 2.0) < cx_mid for s in shoes):
                            violations["left_shoe"] = False

                    if person_regions["right_foot"] is not None:
                        if any(bbox_iou(s["bbox"], person_regions["right_foot"]) > TH_SHOE for s in shoes):
                            violations["right_shoe"] = False
                    else:
                        cy_threshold = y2i - (y2i - y1i) * 0.15
                        cx_mid = (x1 + x2) / 2.0
                        if any(((s["bbox"][1] + s["bbox"][3]) / 2.0) > cy_threshold and ((s["bbox"][0] + s["bbox"][2]) / 2.0) > cx_mid for s in shoes):
                            violations["right_shoe"] = False

                    # If no PPE detections at all assigned, we keep violations true (missing)
                    # else any of the above clearing sets them false.

                    # Draw person bbox and violations
                    cv2.rectangle(annotated, (x1i, y1i), (x2i, y2i), BOX_COLOR, 2)
                    # print violations near bbox
                    txt_y = y1i - 6
                    txt_x = x1i
                    violation_texts = []
                    if violations["helmet"]:
                        violation_texts.append("NO HELMET")
                    if violations["vest"]:
                        violation_texts.append("NO VEST")
                    if violations["left_glove"]:
                        violation_texts.append("NO L-GLOVE")
                    if violations["right_glove"]:
                        violation_texts.append("NO R-GLOVE")
                    if violations["left_shoe"]:
                        violation_texts.append("NO L-SHOE")
                    if violations["right_shoe"]:
                        violation_texts.append("NO R-SHOE")

                    # choose color: red if any violation, green otherwise
                    color = VIOLATION_COLOR if any(violation_texts) else OK_COLOR
                    # draw label background
                    label = ", ".join(violation_texts) if violation_texts else "OK"
                    cv2.putText(annotated, label, (txt_x, max(12, txt_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

                    summary.append({"bbox":[x1i,y1i,x2i,y2i], "violations": violation_texts})

                # compute FPS
                now = time.time()
                dt = now - last_time
                fps = 1.0 / dt if dt > 0 else 0.0
                fps_smooth = alpha * fps_smooth + (1 - alpha) * fps if fps_smooth > 0 else fps
                last_time = now

                # overlay status
                cv2.putText(annotated, f"Frame:{frame_idx}  People:{len(detections)}  PPE:{len(ppe_dets)}  FPS:{fps_smooth:.1f}", (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)

                display = scale_to_fit(annotated) if scaled else annotated
                cv2.imshow(WINDOW_NAME, display)
                frame_idx += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Quit requested.")
                break
            elif key == ord("p"):
                paused = not paused
                print("Paused." if paused else "Resumed.")
            elif key == ord("s"):
                scaled = not scaled
                print("Scaled display:" , scaled)

            if paused:
                cv2.waitKey(100)

    cap.release()
    cv2.destroyAllWindows()
    print("Stopped.")

if __name__ == "__main__":
    main()
