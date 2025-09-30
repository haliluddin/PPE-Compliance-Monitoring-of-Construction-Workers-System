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
YOLO_MODEL = "yolov8n.pt"  # change to yolov8s/m for better accuracy (slower)
SCALED_DISPLAY = True     # start with scaled display to fit screen
SCALE_MAX_WIDTH = 1280
SCALE_MAX_HEIGHT = 720

# Colors (BGR)
WHITE = (255, 255, 255)
LEFT_COLOR = (230, 100, 50)
RIGHT_COLOR = (10, 140, 255)
CENTER_COLOR = (200, 200, 200)
JOINT_RADIUS = 5

# load models
print("Loading YOLO model...")
yolo = YOLO(YOLO_MODEL)

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

                # per-detection crop -> MediaPipe -> draw
                for (bbox, conf) in detections:
                    x1, y1, x2, y2 = bbox
                    x1i, y1i, x2i, y2i = expand_bbox((x1, y1, x2, y2), img_w, img_h)
                    crop = frame[y1i:y2i, x1i:x2i]
                    if crop.size == 0:
                        continue
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    res_pose = pose.process(crop_rgb)
                    if res_pose and res_pose.pose_landmarks:
                        draw_person_pose_on_frame(annotated, res_pose.pose_landmarks, x1i, y1i, x2i - x1i, y2i - y1i)
                    # optional: draw bbox confidence
                    cv2.rectangle(annotated, (x1i, y1i), (x2i, y2i), (80, 200, 80), 2)
                    cv2.putText(annotated, f"{conf:.2f}", (x1i, max(12, y1i - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

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
