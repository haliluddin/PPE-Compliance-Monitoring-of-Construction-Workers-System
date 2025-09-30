import cv2
import mediapipe as mp
import numpy as np
import platform
import subprocess
import shutil
import os
from pathlib import Path
import sys
import time

# Try to import ultralytics YOLO
try:
    from ultralytics import YOLO
except Exception:
    print("ERROR: ultralytics not found. Install with: pip install ultralytics")
    raise

INPUT_PATH = "2048246-hd_1920_1080_24fps.mp4"  # input file (can be a webcam "0" or path)
DETECT_CONF = 0.35        # YOLO detection confidence threshold
DETECT_IOU = 0.45         # YOLO NMS IoU threshold
MAX_PEOPLE = 30           # safety cap
CROP_PAD = 0.12           # padding fraction added to bbox
VISIBILITY_THRESHOLD = 0.3
WINDOW_NAME = "Multi-person Pose (LIVE)"
YOLO_MODEL = "yolov8n.pt"  # person detector
PPE_MODEL_PATH = "runs/segment/ppe_yolov8n_seg_run1/weights/best.pt"  # instance segmentation PPE model
SCALED_DISPLAY = True     # start with scaled display to fit screen
SCALE_MAX_WIDTH = 1280
SCALE_MAX_HEIGHT = 720

# Colors (BGR)
WHITE = (255, 255, 255)
LEFT_COLOR = (230, 100, 50)
RIGHT_COLOR = (10, 140, 255)
CENTER_COLOR = (200, 200, 200)
JOINT_RADIUS = 5

# PPE overlay colors (BGR)
PPE_COLORS = {
    0: (220, 100, 30),   # glove (example)
    1: (0, 215, 255),    # helmet (yellow)
    3: (255, 140, 30),   # shoe (orange)
    4: (30, 200, 30),    # vest (green)
}
PPE_NAME = {
    0: "glove",
    1: "helmet",
    3: "shoe",
    4: "vest"
}
# indexes in your model (from your message): ['glove', 'helmet', 'person', 'shoe', 'vest']
PPE_CLASS_IDS = [0, 1, 3, 4]  # exclude person class (2)

# load models
print("Loading YOLO (person) model...")
yolo = YOLO(YOLO_MODEL)
print("Loading PPE segmentation model...")
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

# left/right sets
left_indices = {lm.value for lm in mp_pose.PoseLandmark if "LEFT" in lm.name}
right_indices = {lm.value for lm in mp_pose.PoseLandmark if "RIGHT" in lm.name}

# helper functions
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
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img

def open_source(src):
    # accept either integer index for webcam or path
    try:
        idx = int(src)
        return cv2.VideoCapture(idx)
    except Exception:
        return cv2.VideoCapture(src)

# mask utilities
def point_in_mask(mask, x, y, radius=20):
    if mask is None:
        return False
    x = int(round(x)); y = int(round(y))
    h, w = mask.shape[:2]
    x1 = max(0, x - radius); x2 = min(w, x + radius)
    y1 = max(0, y - radius); y2 = min(h, y + radius)
    if x2 <= x1 or y2 <= y1:
        return False
    return bool(mask[y1:y2, x1:x2].sum())

def mask_overlap_pixels(mask, bbox):
    if mask is None:
        return 0
    x1, y1, x2, y2 = bbox
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(mask.shape[1], int(x2)); y2 = min(mask.shape[0], int(y2))
    if x2 <= x1 or y2 <= y1:
        return 0
    return int(mask[y1:y2, x1:x2].sum())

def draw_mask_outline(frame, mask, color, thickness=2):
    if mask is None:
        return
    m = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cnt.shape[0] < 6:
            continue
        cv2.polylines(frame, [cnt], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)

def landmarks_to_image_coords(landmarks, x_offset, y_offset, crop_w, crop_h):
    coords = {}
    for idx, lm in enumerate(landmarks.landmark):
        x_img = x_offset + lm.x * crop_w
        y_img = y_offset + lm.y * crop_h
        coords[idx] = (x_img, y_img, getattr(lm, "visibility", 1.0))
    return coords

def safe_get_landmark(landmarks, idx):
    lm = landmarks.landmark[idx]
    return getattr(lm, "x", None), getattr(lm, "y", None), getattr(lm, "visibility", 0.0)

def to_abs_point(lm_x, lm_y, x_offset, y_offset, crop_w, crop_h):
    return x_offset + lm_x * crop_w, y_offset + lm_y * crop_h

def ensure_mask_shape(mask, h, w):
    if mask is None:
        return None
    if mask.shape[0] != h or mask.shape[1] != w:
        # resize to full frame
        return cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return mask

def combine_masks_by_class(r, img_h, img_w):
    class_masks = {cid: np.zeros((img_h, img_w), dtype=np.uint8) for cid in PPE_CLASS_IDS}
    if getattr(r, "masks", None) is None:
        return class_masks

    # try raster masks if available
    masks_data = getattr(r.masks, "data", None)
    cls_arr = []
    boxes = getattr(r, "boxes", None)
    if boxes is not None:
        cls_arr = boxes.cls.cpu().numpy().astype(int)  # np array

    if masks_data is not None:
        # masks_data is typically torch tensor (n, h, w)
        try:
            masks_np = masks_data.cpu().numpy()
        except Exception:
            masks_np = np.array(masks_data)  # fallback
        for i in range(masks_np.shape[0]):
            cls_id = int(cls_arr[i]) if i < len(cls_arr) else None
            if cls_id is None or cls_id not in PPE_CLASS_IDS:
                continue
            mask_i = (masks_np[i] > 0.5).astype(np.uint8)
            # masks might be smaller than full frame (but usually same)
            if mask_i.shape != (img_h, img_w):
                mask_i = cv2.resize(mask_i, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            class_masks[cls_id] = np.bitwise_or(class_masks[cls_id], mask_i)

        return class_masks

    # else attempt polygon representation: r.masks.xy (list of polygon points per instance)
    polys = getattr(r.masks, "xy", None)
    if polys is not None:
        for i, poly in enumerate(polys):
            cls_id = int(cls_arr[i]) if i < len(cls_arr) else None
            if cls_id is None or cls_id not in PPE_CLASS_IDS:
                continue
            # poly is list/array of points [[x,y], [x,y], ...] in absolute pixel coords
            try:
                poly_np = np.array(poly, dtype=np.int32).reshape(-1, 2)
                blank = np.zeros((img_h, img_w), dtype=np.uint8)
                if poly_np.size == 0:
                    continue
                cv2.fillPoly(blank, [poly_np], 1)
                class_masks[cls_id] = np.bitwise_or(class_masks[cls_id], blank)
            except Exception:
                # If polygon format unexpected, skip
                continue
        return class_masks

    return class_masks

def label_text_on_frame(frame, text, x, y, bgcolor=(0,0,0), fgcolor=(220, 30, 30)):
    cv2.putText(frame, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fgcolor, 2, cv2.LINE_AA)

def check_ppe_for_person(class_masks, person_bbox, landmarks, x_offset, y_offset, crop_w, crop_h, img_w, img_h):
    flags = {
        "helmet": False,
        "vest": False,
        "left_glove": False,
        "right_glove": False,
        "left_shoe": False,
        "right_shoe": False
    }

    # Convert some landmark indices to absolute image coords
    # helpful MP indices
    LM = mp_pose.PoseLandmark
    # helper to get absolute point if visible
    def get_abs(idx):
        lm = landmarks.landmark[idx]
        return (x_offset + lm.x * crop_w, y_offset + lm.y * crop_h, getattr(lm,"visibility",0.0))

    # compute torso bbox from shoulders & hips
    try:
        ls_x, ls_y, ls_v = get_abs(LM.LEFT_SHOULDER.value)
        rs_x, rs_y, rs_v = get_abs(LM.RIGHT_SHOULDER.value)
        lh_x, lh_y, lh_v = get_abs(LM.LEFT_HIP.value)
        rh_x, rh_y, rh_v = get_abs(LM.RIGHT_HIP.value)
    except Exception:
        ls_v = rs_v = lh_v = rh_v = 0.0

    # compute head center using nose & eyes & ears if visible
    head_pts = []
    for idx in [LM.NOSE.value, LM.LEFT_EYE.value, LM.RIGHT_EYE.value, LM.LEFT_EAR.value, LM.RIGHT_EAR.value]:
        lm = landmarks.landmark[idx]
        if getattr(lm, "visibility", 0.0) >= VISIBILITY_THRESHOLD:
            head_pts.append((x_offset + lm.x * crop_w, y_offset + lm.y * crop_h))
    if len(head_pts) > 0:
        hx = int(np.mean([p[0] for p in head_pts]))
        hy = int(np.mean([p[1] for p in head_pts]))
    else:
        # fallback to top of bbox
        hx = int((person_bbox[0] + person_bbox[2]) / 2)
        hy = int(person_bbox[1] + (person_bbox[3] - person_bbox[1]) * 0.12)

    # glove checks using wrists
    lw_x, lw_y, lw_v = get_abs(LM.LEFT_WRIST.value)
    rw_x, rw_y, rw_v = get_abs(LM.RIGHT_WRIST.value)
    # shoe checks using ankles / foot index
    la_x, la_y, la_v = get_abs(LM.LEFT_ANKLE.value)
    ra_x, ra_y, ra_v = get_abs(LM.RIGHT_ANKLE.value)

    # Masks from class_masks dict
    glove_mask = class_masks.get(0, None)
    helmet_mask = class_masks.get(1, None)
    shoe_mask = class_masks.get(3, None)
    vest_mask = class_masks.get(4, None)

    # gloves: require wrist visibility and check small radius around wrist
    if lw_v >= VISIBILITY_THRESHOLD:
        if point_in_mask(glove_mask, lw_x, lw_y, radius=25):
            flags["left_glove"] = True
    if rw_v >= VISIBILITY_THRESHOLD:
        if point_in_mask(glove_mask, rw_x, rw_y, radius=25):
            flags["right_glove"] = True

    # shoes: check ankle/foot area
    if la_v >= 0.1:
        if point_in_mask(shoe_mask, la_x, la_y, radius=30):
            flags["left_shoe"] = True
    if ra_v >= 0.1:
        if point_in_mask(shoe_mask, ra_x, ra_y, radius=30):
            flags["right_shoe"] = True

    # helmet: check overlap near head center
    if helmet_mask is not None:
        # check a small head bbox
        head_w = int(max(20, (person_bbox[2] - person_bbox[0]) * 0.25))
        head_h = int(max(20, (person_bbox[3] - person_bbox[1]) * 0.18))
        hbbox = (hx - head_w, hy - head_h, hx + head_w, hy + head_h)
        inter = mask_overlap_pixels(helmet_mask, hbbox)
        # Also check direct point
        helmet_present = inter > 50 or point_in_mask(helmet_mask, hx, hy, radius=25)
        flags["helmet"] = bool(helmet_present)

    # vest: check torso overlap
    if vest_mask is not None:
        # if shoulders/hips visible build torso box
        if ls_v + rs_v + lh_v + rh_v > 0.0:
            xs = [p for p in [ls_x, rs_x, lh_x, rh_x] if p is not None]
            ys = [p for p in [ls_y, rs_y, lh_y, rh_y] if p is not None]
            if xs and ys:
                tx1 = min(xs) - 10; tx2 = max(xs) + 10
                ty1 = min(ys) - 10; ty2 = max(ys) + 10
                torso_bbox = (tx1, ty1, tx2, ty2)
                inter = mask_overlap_pixels(vest_mask, torso_bbox)
                flags["vest"] = inter > 80
        else:
            # fallback: check center of bbox region
            cx = int((person_bbox[0] + person_bbox[2]) / 2)
            cy = int((person_bbox[1] + person_bbox[3]) / 2)
            flags["vest"] = point_in_mask(vest_mask, cx, cy, radius=50)

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

    # fps tracker
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

                # YOLO person detection
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
                        class_masks = combine_masks_by_class(r_ppe, img_h, img_w)
                    else:
                        class_masks = {cid: np.zeros((img_h, img_w), dtype=np.uint8) for cid in PPE_CLASS_IDS}
                except Exception as e:
                    # If PPE model fails for any reason, default to empty masks and continue
                    # print("PPE model error:", e)
                    class_masks = {cid: np.zeros((img_h, img_w), dtype=np.uint8) for cid in PPE_CLASS_IDS}

                # optional: draw PPE outlines for visualization
                for cid, mask in class_masks.items():
                    if mask is not None and mask.sum() > 50:
                        color = PPE_COLORS.get(cid, (0,255,0))
                        draw_mask_outline(annotated, mask, color=color, thickness=2)

                # per-detection crop -> MediaPipe -> draw and check PPE
                for (bbox, conf) in detections:
                    x1, y1, x2, y2 = bbox
                    x1i, y1i, x2i, y2i = expand_bbox((x1, y1, x2, y2), img_w, img_h)
                    crop = frame[y1i:y2i, x1i:x2i]
                    if crop.size == 0:
                        continue
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    res_pose = pose.process(crop_rgb)
                    person_bbox = (x1i, y1i, x2i, y2i)
                    # default violations empty
                    violations = []
                    if res_pose and res_pose.pose_landmarks:
                        # draw pose
                        draw_person_pose_on_frame(annotated, res_pose.pose_landmarks, x1i, y1i, x2i - x1i, y2i - y1i)
                        # check PPE flags using masks and pose landmarks
                        flags = check_ppe_for_person(class_masks, person_bbox, res_pose.pose_landmarks, x1i, y1i, x2i - x1i, y2i - y1i, img_w, img_h)
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
                        # No pose landmarks; we can still check mask overlap with whole bbox for vest/helmet
                        # vest
                        vest_mask = class_masks.get(4, None)
                        if vest_mask is not None:
                            inter = mask_overlap_pixels(vest_mask, person_bbox)
                            if inter < 60:
                                violations.append("NO VEST")
                        # helmet
                        helmet_mask = class_masks.get(1, None)
                        if helmet_mask is not None:
                            inter = mask_overlap_pixels(helmet_mask, person_bbox)
                            if inter < 40:
                                violations.append("NO HELMET")
                        # gloves & shoes are hard to infer without landmarks; mark unknown
                        violations.append("NO LEFT GLOVE?")
                        violations.append("NO RIGHT GLOVE?")
                        violations.append("NO LEFT SHOE?")
                        violations.append("NO RIGHT SHOE?")

                    # draw bbox and confidence
                    cv2.rectangle(annotated, (x1i, y1i), (x2i, y2i), (80, 200, 80), 2)
                    cv2.putText(annotated, f"{conf:.2f}", (x1i, max(12, y1i - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

                    # draw violations text under bbox (stacked)
                    ty = y2i + 18
                    for v in violations:
                        cv2.putText(annotated, v, (x1i + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 220), 2, cv2.LINE_AA)
                        ty += 18

                # compute FPS
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
