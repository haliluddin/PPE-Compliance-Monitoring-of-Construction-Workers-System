import cv2
import mediapipe as mp
import numpy as np
import sys
import time
from pathlib import Path

try:
    from ultralytics import YOLO
except Exception:
    print("ERROR: ultralytics not found. Install with: pip install ultralytics")
    raise

INPUT_PATH = "2048246-hd_1920_1080_24fps.mp4"
DETECT_CONF = 0.35
DETECT_IOU = 0.45
MAX_PEOPLE = 30
CROP_PAD = 0.12
VISIBILITY_THRESHOLD = 0.3
WINDOW_NAME = "Multi-person Pose (LIVE)"
YOLO_MODEL = "yolov8n.pt"
PPE_MODEL_PATH = "runs/segment/ppe_yolov8n_seg_run1/weights/best.pt"
SCALED_DISPLAY = True
SCALE_MAX_WIDTH = 1280
SCALE_MAX_HEIGHT = 720

# Colors (BGR)
WHITE = (255, 255, 255)
LEFT_COLOR = (230, 100, 50)
RIGHT_COLOR = (10, 140, 255)
CENTER_COLOR = (200, 200, 200)
JOINT_RADIUS = 5

PPE_COLORS = {
    0: (220, 100, 30),  # glove
    1: (0, 215, 255),   # helmet
    3: (255, 140, 30),  # shoe
    4: (30, 200, 30),   # vest
}
PPE_NAME = {0: "glove", 1: "helmet", 3: "shoe", 4: "vest"}
PPE_CLASS_IDS = [0, 1, 3, 4]  # glove, helmet, shoe, vest

print("Loading YOLO (person) model...")
yolo = YOLO(YOLO_MODEL)
print("Loading PPE segmentation model (we'll only use its boxes)...")
yolo_ppe = YOLO(PPE_MODEL_PATH)

mp_pose = mp.solutions.pose
pose_conf = dict(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
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

def draw_person_pose_on_frame(frame, landmarks, x_offset, y_offset, crop_w, crop_h, visibility_threshold=VISIBILITY_THRESHOLD):
    for connection in mp_pose.POSE_CONNECTIONS:
        s = connection[0]; e = connection[1]
        lm_s = landmarks.landmark[s]; lm_e = landmarks.landmark[e]
        if getattr(lm_s, "visibility", 1.0) < visibility_threshold or getattr(lm_e, "visibility", 1.0) < visibility_threshold:
            continue
        x1 = int(x_offset + lm_s.x * crop_w); y1 = int(y_offset + lm_s.y * crop_h)
        x2 = int(x_offset + lm_e.x * crop_w); y2 = int(y_offset + lm_e.y * crop_h)
        cv2.line(frame, (x1, y1), (x2, y2), WHITE, thickness=7, lineType=cv2.LINE_AA)
        cv2.line(frame, (x1, y1), (x2, y2), (220, 220, 220), thickness=2, lineType=cv2.LINE_AA)
    for idx, lm in enumerate(landmarks.landmark):
        if lm.visibility < visibility_threshold:
            continue
        cx = int(x_offset + lm.x * crop_w); cy = int(y_offset + lm.y * crop_h)
        if idx in left_indices:
            color = LEFT_COLOR
        elif idx in right_indices:
            color = RIGHT_COLOR
        else:
            color = CENTER_COLOR
        cv2.circle(frame, (cx, cy), JOINT_RADIUS + 3, WHITE, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), JOINT_RADIUS, color, thickness=-1, lineType=cv2.LINE_AA)

def scale_to_fit(img, max_w=SCALE_MAX_WIDTH, max_h=SCALE_MAX_HEIGHT):
    h, w = img.shape[:2]
    scale = min(1.0, max_w / w, max_h / h)
    if scale < 1.0:
        new_w = int(w * scale); new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img

def open_source(src):
    try:
        idx = int(src)
        return cv2.VideoCapture(idx)
    except Exception:
        return cv2.VideoCapture(src)

def iou(boxA, boxB):
    # boxes are (x1,y1,x2,y2)
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH
    areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    union = areaA + areaB - interArea
    return interArea / union if union > 0 else 0.0

def point_to_box(px, py, radius):
    return (px - radius, py - radius, px + radius, py + radius)

def combine_boxes_by_class(r, img_h, img_w):
    boxes_by_class = {cid: [] for cid in PPE_CLASS_IDS}
    boxes = getattr(r, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return boxes_by_class
    try:
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
    except Exception:
        # fallback if CPU tensors not available
        xyxy = np.array(boxes.xyxy).astype(float)
        confs = np.array(boxes.conf).astype(float)
        cls = np.array(boxes.cls).astype(int)
    for (x1, y1, x2, y2), c, cl in zip(xyxy, confs, cls):
        if int(cl) not in PPE_CLASS_IDS:
            continue
        # clamp
        x1c = max(0, float(x1)); y1c = max(0, float(y1))
        x2c = min(img_w - 1, float(x2)); y2c = min(img_h - 1, float(y2))
        boxes_by_class[int(cl)].append((x1c, y1c, x2c, y2c, float(c)))
    return boxes_by_class

def check_ppe_for_person_from_boxes(boxes_by_class, person_bbox, landmarks, x_offset, y_offset, crop_w, crop_h):
    flags = {
        "helmet": False,
        "vest": False,
        "left_glove": False,
        "right_glove": False,
        "left_shoe": False,
        "right_shoe": False
    }
    LM = mp_pose.PoseLandmark

    def get_abs(idx):
        lm = landmarks.landmark[idx]
        return (x_offset + lm.x * crop_w, y_offset + lm.y * crop_h, getattr(lm, "visibility", 0.0))

    # head center
    head_pts = []
    for idx in [LM.NOSE.value, LM.LEFT_EYE.value, LM.RIGHT_EYE.value, LM.LEFT_EAR.value, LM.RIGHT_EAR.value]:
        lm = landmarks.landmark[idx]
        if getattr(lm, "visibility", 0.0) >= VISIBILITY_THRESHOLD:
            head_pts.append((x_offset + lm.x * crop_w, y_offset + lm.y * crop_h))
    if len(head_pts) > 0:
        hx = int(np.mean([p[0] for p in head_pts])); hy = int(np.mean([p[1] for p in head_pts]))
    else:
        hx = int((person_bbox[0] + person_bbox[2]) / 2)
        hy = int(person_bbox[1] + (person_bbox[3] - person_bbox[1]) * 0.12)

    # wrists and ankles
    lw_x, lw_y, lw_v = get_abs(LM.LEFT_WRIST.value)
    rw_x, rw_y, rw_v = get_abs(LM.RIGHT_WRIST.value)
    la_x, la_y, la_v = get_abs(LM.LEFT_ANKLE.value)
    ra_x, ra_y, ra_v = get_abs(LM.RIGHT_ANKLE.value)

    # shoulders & hips for torso box
    try:
        ls_x, ls_y, ls_v = get_abs(LM.LEFT_SHOULDER.value)
        rs_x, rs_y, rs_v = get_abs(LM.RIGHT_SHOULDER.value)
        lh_x, lh_y, lh_v = get_abs(LM.LEFT_HIP.value)
        rh_x, rh_y, rh_v = get_abs(LM.RIGHT_HIP.value)
    except Exception:
        ls_v = rs_v = lh_v = rh_v = 0.0

    # get relevant PPE boxes
    glove_boxes = boxes_by_class.get(0, [])
    helmet_boxes = boxes_by_class.get(1, [])
    shoe_boxes = boxes_by_class.get(3, [])
    vest_boxes = boxes_by_class.get(4, [])

    # glove detection: form a small box around wrist and check IoU/overlap with glove boxes
    wrist_radius = max(16, int((person_bbox[2] - person_bbox[0]) * 0.04))
    if lw_v >= VISIBILITY_THRESHOLD:
        wrist_box = point_to_box(lw_x, lw_y, wrist_radius)
        for (bx1, by1, bx2, by2, conf) in glove_boxes:
            if iou(wrist_box, (bx1, by1, bx2, by2)) > 0.02:
                flags["left_glove"] = True
                break
    if rw_v >= VISIBILITY_THRESHOLD:
        wrist_box = point_to_box(rw_x, rw_y, wrist_radius)
        for (bx1, by1, bx2, by2, conf) in glove_boxes:
            if iou(wrist_box, (bx1, by1, bx2, by2)) > 0.02:
                flags["right_glove"] = True
                break

    # shoe detection: small box around ankle
    ankle_radius = max(20, int((person_bbox[3] - person_bbox[1]) * 0.05))
    if la_v >= 0.05:
        ank_box = point_to_box(la_x, la_y, ankle_radius)
        for (bx1, by1, bx2, by2, conf) in shoe_boxes:
            if iou(ank_box, (bx1, by1, bx2, by2)) > 0.02:
                flags["left_shoe"] = True
                break
    if ra_v >= 0.05:
        ank_box = point_to_box(ra_x, ra_y, ankle_radius)
        for (bx1, by1, bx2, by2, conf) in shoe_boxes:
            if iou(ank_box, (bx1, by1, bx2, by2)) > 0.02:
                flags["right_shoe"] = True
                break

    # helmet: compute head bbox and check IoU with helmet boxes
    head_w = max(20, (person_bbox[2] - person_bbox[0]) * 0.25)
    head_h = max(20, (person_bbox[3] - person_bbox[1]) * 0.18)
    head_box = (hx - head_w, hy - head_h, hx + head_w, hy + head_h)
    for (bx1, by1, bx2, by2, conf) in helmet_boxes:
        if iou(head_box, (bx1, by1, bx2, by2)) > 0.05:
            flags["helmet"] = True
            break

    # vest: torso box from shoulders & hips if available
    if ls_v + rs_v + lh_v + rh_v > 0.0:
        xs = [p for p in [ls_x, rs_x, lh_x, rh_x] if p is not None]
        ys = [p for p in [ls_y, rs_y, lh_y, rh_y] if p is not None]
        if xs and ys:
            tx1 = min(xs) - 10; tx2 = max(xs) + 10
            ty1 = min(ys) - 10; ty2 = max(ys) + 10
            torso_box = (tx1, ty1, tx2, ty2)
            for (bx1, by1, bx2, by2, conf) in vest_boxes:
                if iou(torso_box, (bx1, by1, bx2, by2)) > 0.05:
                    flags["vest"] = True
                    break
    else:
        # fallback to center-of-person check
        cx = int((person_bbox[0] + person_bbox[2]) / 2); cy = int((person_bbox[1] + person_bbox[3]) / 2)
        center_box = point_to_box(cx, cy, max(30, int((person_bbox[3] - person_bbox[1]) * 0.12)))
        for (bx1, by1, bx2, by2, conf) in vest_boxes:
            if iou(center_box, (bx1, by1, bx2, by2)) > 0.05:
                flags["vest"] = True
                break

    return flags

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
        print("Live monitor started. Controls: q=quit, p=pause/resume, s=toggle scaled display")
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("End of source.")
                    break
                img_h, img_w = frame.shape[:2]

                # person detection
                results = yolo.predict(source=frame, conf=DETECT_CONF, iou=DETECT_IOU, classes=[0], verbose=False)
                detections = []
                if len(results) > 0:
                    r = results[0]
                    boxes = getattr(r, "boxes", None)
                    if boxes is not None and len(boxes) > 0:
                        xyxy = boxes.xyxy.cpu().numpy()
                        confs = boxes.conf.cpu().numpy()
                        cls = boxes.cls.cpu().numpy()
                        for (x1, y1, x2, y2), c, cl in zip(xyxy, confs, cls):
                            if int(cl) != 0:
                                continue
                            detections.append(((float(x1), float(y1), float(x2), float(y2)), float(c)))

                if len(detections) > MAX_PEOPLE:
                    detections = sorted(detections, key=lambda x: x[1], reverse=True)[:MAX_PEOPLE]

                annotated = frame.copy()

                try:
                    ppe_results = yolo_ppe.predict(source=frame, conf=DETECT_CONF, iou=DETECT_IOU, classes=PPE_CLASS_IDS, verbose=False)
                    if len(ppe_results) > 0:
                        r_ppe = ppe_results[0]
                        boxes_by_class = combine_boxes_by_class(r_ppe, img_h, img_w)
                    else:
                        boxes_by_class = {cid: [] for cid in PPE_CLASS_IDS}
                except Exception as e:
                    boxes_by_class = {cid: [] for cid in PPE_CLASS_IDS}

                # draw PPE boxes for visualization
                for cid, boxlist in boxes_by_class.items():
                    color = PPE_COLORS.get(cid, (0,255,0))
                    for (bx1, by1, bx2, by2, conf) in boxlist:
                        cv2.rectangle(annotated, (int(bx1), int(by1)), (int(bx2), int(by2)), color, 2)
                        cv2.putText(annotated, f"{PPE_NAME.get(cid,'ppe')}:{conf:.2f}", (int(bx1), max(12, int(by1)-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220,220,220), 1, cv2.LINE_AA)

                # per person -> pose -> check PPE using boxes
                for (bbox, conf) in detections:
                    x1, y1, x2, y2 = bbox
                    x1i, y1i, x2i, y2i = expand_bbox((x1, y1, x2, y2), img_w, img_h)
                    crop = frame[y1i:y2i, x1i:x2i]
                    if crop.size == 0:
                        continue
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    res_pose = pose.process(crop_rgb)
                    person_bbox = (x1i, y1i, x2i, y2i)
                    violations = []
                    if res_pose and res_pose.pose_landmarks:
                        draw_person_pose_on_frame(annotated, res_pose.pose_landmarks, x1i, y1i, x2i - x1i, y2i - y1i)
                        flags = check_ppe_for_person_from_boxes(boxes_by_class, person_bbox, res_pose.pose_landmarks, x1i, y1i, x2i - x1i, y2i - y1i)
                        if not flags.get("helmet", False):
                            violations.append("NO HELMET")
                        if not flags.get("vest", False):
                            violations.append("NO VEST")
                        if not flags.get("left_glove", False):
                            violations.append("NO LEFT GLOVE")
                        if not flags.get("right_glove", False):
                            violations.append("NO RIGHT GLOVE")
                        if not flags.get("left_shoe", False):
                            violations.append("NO LEFT SHOE")
                        if not flags.get("right_shoe", False):
                            violations.append("NO RIGHT SHOE")
                    else:
                        # if no landmarks, check simple bbox overlap for vest/helmet
                        vest_boxes = boxes_by_class.get(4, [])
                        helmet_boxes = boxes_by_class.get(1, [])
                        vest_found = any(iou(person_bbox, (bx1,by1,bx2,by2)) > 0.03 for (bx1,by1,bx2,by2,_) in vest_boxes)
                        helmet_found = any(iou(person_bbox, (bx1,by1,bx2,by2)) > 0.02 for (bx1,by1,bx2,by2,_) in helmet_boxes)
                        if not helmet_found:
                            violations.append("NO HELMET")
                        if not vest_found:
                            violations.append("NO VEST")
                        violations.append("NO LEFT GLOVE?"); violations.append("NO RIGHT GLOVE?")
                        violations.append("NO LEFT SHOE?"); violations.append("NO RIGHT SHOE?")

                    cv2.rectangle(annotated, (x1i, y1i), (x2i, y2i), (80, 200, 80), 2)
                    cv2.putText(annotated, f"{conf:.2f}", (x1i, max(12, y1i - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

                    ty = y2i + 18
                    for v in violations:
                        cv2.putText(annotated, v, (x1i + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 220), 2, cv2.LINE_AA)
                        ty += 18

                # FPS
                now = time.time()
                dt = now - last_time
                fps = 1.0 / dt if dt > 0 else 0.0
                fps_smooth = alpha * fps_smooth + (1 - alpha) * fps if fps_smooth > 0 else fps
                last_time = now

                cv2.putText(annotated, f"Frame:{frame_idx}  People:{len(detections)}  FPS:{fps_smooth:.1f}", (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)

                display = scale_to_fit(annotated) if scaled else annotated
                cv2.imshow(WINDOW_NAME, display)
                frame_idx += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Quit requested."); break
            elif key == ord("p"):
                paused = not paused; print("Paused." if paused else "Resumed.")
            elif key == ord("s"):
                scaled = not scaled; print("Scaled display:", scaled)
            if paused:
                cv2.waitKey(100)

    cap.release()
    cv2.destroyAllWindows()
    print("Stopped.")

if __name__ == "__main__":
    main()
