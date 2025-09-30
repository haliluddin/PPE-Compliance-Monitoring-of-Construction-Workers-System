import cv2
import numpy as np
import math
import time
from pathlib import Path

# ultralytics YOLO
from ultralytics import YOLO
import mediapipe as mp

INPUT_PATH = "2048246-hd_1920_1080_24fps.mp4"   # can be "0" for webcam
MODEL_PATH = "runs/segment/ppe_yolov8n_seg_run1/weights/best.pt"
CONF = 0.35
IOU = 0.45
MAX_PEOPLE = 30
VISIBILITY_THRESHOLD = 0.3
WINDOW_NAME = "PPE Compliance (LIVE)"

# Colors (BGR)
COLOR_OK = (60, 180, 75)
COLOR_VIOL = (0, 0, 255)
COLOR_BOX = (200, 200, 200)
COLOR_TEXT = (230, 230, 230)
PPE_COLORS = {
    "helmet": (0, 210, 210),
    "glove": (180, 120, 40),
    "shoe": (200, 80, 200),
    "vest": (80, 200, 120),
}

# load model
print("Loading YOLO model...")
model = YOLO(MODEL_PATH)
names = {int(k): v for k, v in model.names.items()}
# map class names to ids (safe lookup)
class_name_to_id = {v: int(k) for k, v in names.items()}
# ensure required classes exist
required = ["glove", "helmet", "person", "shoe", "vest"]
for r in required:
    if r not in class_name_to_id:
        raise SystemExit(f"Model at {MODEL_PATH} does not contain class '{r}' in model.names. Found: {names}")

ID_PERSON = class_name_to_id["person"]
ID_GLOVE = class_name_to_id["glove"]
ID_HELMET = class_name_to_id["helmet"]
ID_SHOE = class_name_to_id["shoe"]
ID_VEST = class_name_to_id["vest"]

# setup MediaPipe Pose (we'll run per-person crop)
mp_pose = mp.solutions.pose
pose_conf = dict(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def iou_xyxy(boxA, boxB):
    # boxes = (x1,y1,x2,y2)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union = areaA + areaB - interArea
    return interArea / union if union > 0 else 0.0


def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def expand_box(box, img_w, img_h, pad=0.12):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    p = int(max(w, h) * pad)
    return (max(0, int(x1) - p), max(0, int(y1) - p), min(img_w - 1, int(x2) + p), min(img_h - 1, int(y2) + p))


def landmark_to_full(lm, crop_box):
    x1, y1, x2, y2 = crop_box
    cw = x2 - x1
    ch = y2 - y1
    return (int(x1 + lm.x * cw), int(y1 + lm.y * ch))


def assign_ppe_to_persons(person_boxes, ppe_boxes):
    mapping = {i: [] for i in range(len(person_boxes))}
    for p_idx, pbox in enumerate(person_boxes):
        pass
    for idx_ppe, ppe in enumerate(ppe_boxes):
        best_iou = 0.0
        best_person = None
        for i, pbox in enumerate(person_boxes):
            val = iou_xyxy(pbox, ppe["box"])
            if val > best_iou:
                best_iou = val
                best_person = i
        # require small overlap to count (0.05)
        if best_person is not None and best_iou > 0.03:
            mapping[best_person].append(idx_ppe)
    return mapping


def decide_left_right_by_landmarks(ppe_box_center_x, left_coord, right_coord, person_center_x):
    if left_coord is not None and right_coord is not None:
        d_left = abs(ppe_box_center_x - left_coord)
        d_right = abs(ppe_box_center_x - right_coord)
        return "left" if d_left < d_right else "right"
    elif left_coord is not None:
        return "left" if ppe_box_center_x < person_center_x else "right"
    elif right_coord is not None:
        return "right" if ppe_box_center_x > person_center_x else "left"
    else:
        # fallback purely by center comparison
        return "left" if ppe_box_center_x < person_center_x else "right"


def run_live():
    # open source
    try:
        src_idx = int(INPUT_PATH)
        cap = cv2.VideoCapture(src_idx)
    except Exception:
        cap = cv2.VideoCapture(INPUT_PATH)

    if not cap.isOpened():
        raise SystemExit(f"Cannot open input {INPUT_PATH}")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    paused = False
    frame_idx = 0
    fps_t0 = time.time()
    fps_smooth = 0.0
    alpha = 0.9

    with mp_pose.Pose(**pose_conf) as pose:
        print("Started. Controls: q=quit, p=pause/resume")
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("End of source.")
                    break
                H, W = frame.shape[:2]

                # run YOLO (single inference)
                results = model.predict(source=frame, conf=CONF, iou=IOU, verbose=False)
                if len(results) == 0:
                    detections_boxes = []
                else:
                    r = results[0]
                    boxes = getattr(r, "boxes", None)
                    detections_boxes = []
                    if boxes is not None and len(boxes) > 0:
                        xyxy = boxes.xyxy.cpu().numpy()
                        confs = boxes.conf.cpu().numpy()
                        cls = boxes.cls.cpu().numpy()
                        for (x1, y1, x2, y2), c, cl in zip(xyxy, confs, cls):
                            detections_boxes.append({
                                "box": (float(x1), float(y1), float(x2), float(y2)),
                                "conf": float(c),
                                "cls": int(cl)
                            })

                # separate person boxes and ppe boxes
                person_boxes = [d["box"] for d in detections_boxes if d["cls"] == ID_PERSON]
                ppe_boxes = [d for d in detections_boxes if d["cls"] in (ID_GLOVE, ID_HELMET, ID_SHOE, ID_VEST)]

                # safety cap
                if len(person_boxes) > MAX_PEOPLE:
                    person_boxes = sorted(person_boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)[:MAX_PEOPLE]

                # map ppe -> person indices by IoU
                ppe_to_person = assign_ppe_to_persons(person_boxes, ppe_boxes)

                # prepare per-person status and pose
                persons_info = []
                for p_idx, pbox in enumerate(person_boxes):
                    x1, y1, x2, y2 = map(int, pbox)
                    crop_box = expand_box((x1, y1, x2, y2), W, H, pad=0.08)
                    cx1, cy1, cx2, cy2 = crop_box
                    crop = frame[cy1:cy2, cx1:cx2]
                    persons_info.append({
                        "box": (x1, y1, x2, y2),
                        "crop_box": crop_box,
                        "landmarks": None,
                        "assigned_ppe_idxs": ppe_to_person.get(p_idx, []),
                        "violations": {
                            "helmet": True,
                            "left_glove": True,
                            "right_glove": True,
                            "left_shoe": True,
                            "right_shoe": True,
                            "vest": True
                        }
                    })

                    # run pose on crop (if crop valid)
                    if crop.size > 0:
                        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        res_pose = pose.process(crop_rgb)
                        if res_pose and res_pose.pose_landmarks:
                            # convert landmarks to full-frame pixel coords
                            lm_full = {}
                            for i, lm in enumerate(res_pose.pose_landmarks.landmark):
                                px, py = landmark_to_full(lm, crop_box)
                                lm_full[i] = {"x": px, "y": py, "vis": getattr(lm, "visibility", 1.0)}
                            persons_info[-1]["landmarks"] = lm_full

                # Evaluate PPE presence per person
                for p_idx, pinfo in enumerate(persons_info):
                    pbox = pinfo["box"]
                    px1, py1, px2, py2 = pbox
                    p_cx, p_cy = box_center(pbox)
                    lm = pinfo["landmarks"]

                    # convenience x positions for wrists / ankles if available (pixel coords)
                    left_wrist_x = None
                    right_wrist_x = None
                    left_ankle_x = None
                    right_ankle_x = None
                    left_shoulder_x = None
                    right_shoulder_x = None
                    nose_y = None
                    if lm:
                        def get_lm_x(idx):
                            v = lm.get(idx)
                            return v["x"] if (v and v["vis"] >= VISIBILITY_THRESHOLD) else None
                        def get_lm_y(idx):
                            v = lm.get(idx)
                            return v["y"] if (v and v["vis"] >= VISIBILITY_THRESHOLD) else None

                        left_wrist_x = get_lm_x(mp_pose.PoseLandmark.LEFT_WRIST.value)
                        right_wrist_x = get_lm_x(mp_pose.PoseLandmark.RIGHT_WRIST.value)
                        left_ankle_x = get_lm_x(mp_pose.PoseLandmark.LEFT_ANKLE.value)
                        right_ankle_x = get_lm_x(mp_pose.PoseLandmark.RIGHT_ANKLE.value)
                        left_shoulder_x = get_lm_x(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
                        right_shoulder_x = get_lm_x(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
                        nose_y = get_lm_y(mp_pose.PoseLandmark.NOSE.value)

                    # iterate assigned ppe indices
                    for ppe_idx in pinfo["assigned_ppe_idxs"]:
                        ppe = ppe_boxes[ppe_idx]
                        cls = ppe["cls"]
                        box = ppe["box"]
                        bx1, by1, bx2, by2 = box
                        b_cx, b_cy = box_center(box)

                        # only consider if reasonably overlapping with person's vertical span
                        if not (by2 > py1 + 0.05 * (py2-py1) and by1 < py2 - 0.05 * (py2-py1)):
                            # little vertical overlap -> skip
                            pass

                        # helmet detection: should be near head (upper part)
                        if cls == ID_HELMET:
                            # if nose_y exists, helmet center y should be above nose or near top of bbox
                            helmet_on_head = False
                            if nose_y is not None:
                                helmet_on_head = b_cy < nose_y + (py1 - 0) * 0  # basically above nose
                                # simpler: check helmet center is in upper 40% of person bbox
                            if b_cy < py1 + 0.45 * (py2 - py1):
                                helmet_on_head = True
                            if helmet_on_head:
                                pinfo["violations"]["helmet"] = False

                        # vest detection: overlap with torso (between shoulders and hips)
                        elif cls == ID_VEST:
                            # torso vertical band approx between shoulders and hips
                            if lm and left_shoulder_x and right_shoulder_x:
                                # torso y-range: shoulders y to hips y
                                sh_y = lm.get(mp_pose.PoseLandmark.LEFT_SHOULDER.value, {}).get("y", None)
                                if sh_y is None:
                                    sh_y = lm.get(mp_pose.PoseLandmark.RIGHT_SHOULDER.value, {}).get("y", None)
                                hip_y = lm.get(mp_pose.PoseLandmark.LEFT_HIP.value, {}).get("y", None) or lm.get(mp_pose.PoseLandmark.RIGHT_HIP.value, {}).get("y", None)
                                if sh_y and hip_y:
                                    # check overlap
                                    if by1 < hip_y and by2 > sh_y:
                                        pinfo["violations"]["vest"] = False
                            else:
                                # fallback: if vest bbox center is in central 60% of person bbox
                                if px1 + 0.2 * (px2-px1) < b_cx < px1 + 0.8 * (px2-px1) and py1 + 0.25*(py2-py1) < b_cy < py1 + 0.75*(py2-py1):
                                    pinfo["violations"]["vest"] = False

                        # glove detection: decide left/right by comparing to wrists
                        elif cls == ID_GLOVE:
                            side = decide_left_right_by_landmarks(b_cx, left_wrist_x, right_wrist_x, p_cx)
                            if side == "left":
                                pinfo["violations"]["left_glove"] = False
                            else:
                                pinfo["violations"]["right_glove"] = False

                        # shoe detection: decide left/right by ankles
                        elif cls == ID_SHOE:
                            side = decide_left_right_by_landmarks(b_cx, left_ankle_x, right_ankle_x, p_cx)
                            if side == "left":
                                pinfo["violations"]["left_shoe"] = False
                            else:
                                pinfo["violations"]["right_shoe"] = False

                # Drawing & visualization
                annotated = frame.copy()
                # draw PPE boxes
                for ppe in ppe_boxes:
                    x1, y1, x2, y2 = map(int, ppe["box"])
                    cname = names[ppe["cls"]]
                    color = PPE_COLORS.get(cname, COLOR_BOX)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated, f"{cname}", (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)

                # draw persons + violations
                for pinfo in persons_info:
                    x1, y1, x2, y2 = map(int, pinfo["box"])
                    v = pinfo["violations"]
                    missing = []
                    if v["helmet"]:
                        missing.append("no helmet")
                    if v["left_glove"]:
                        missing.append("no left glove")
                    if v["right_glove"]:
                        missing.append("no right glove")
                    if v["left_shoe"]:
                        missing.append("no left shoe")
                    if v["right_shoe"]:
                        missing.append("no right shoe")
                    if v["vest"]:
                        missing.append("no vest")

                    ok = len(missing) == 0
                    box_color = COLOR_OK if ok else COLOR_VIOL
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 3)
                    label = "OK" if ok else ", ".join(missing)
                    # clip label length
                    if len(label) > 80:
                        label = label[:77] + "..."
                    cv2.putText(annotated, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)

                    # optionally draw landmarks
                    if pinfo["landmarks"]:
                        for idx, lm in pinfo["landmarks"].items():
                            if lm["vis"] >= VISIBILITY_THRESHOLD:
                                cv2.circle(annotated, (lm["x"], lm["y"]), 3, (220, 220, 220), -1)

                # FPS
                tnow = time.time()
                fps = 1.0 / (tnow - fps_t0) if (tnow - fps_t0) > 0 else 0.0
                fps_smooth = alpha * fps_smooth + (1 - alpha) * fps if fps_smooth > 0 else fps
                fps_t0 = tnow
                cv2.putText(annotated, f"Frame:{frame_idx}  Persons:{len(person_boxes)}  FPS:{fps_smooth:.1f}", (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2, cv2.LINE_AA)

                cv2.imshow(WINDOW_NAME, annotated)
                frame_idx += 1

            # keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Quit requested.")
                break
            elif key == ord("p"):
                paused = not paused
                print("Paused." if paused else "Resumed.")

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_live()
