import argparse
import time
from pathlib import Path

import cv2
import numpy as np

# ultralytics YOLO
try:
    from ultralytics import YOLO
except Exception as e:
    print("Install ultralytics: pip install ultralytics")
    raise

# ---------- Config / defaults ----------
DEFAULT_MODEL = "runs/segment/ppe_yolov8n_seg_run1/weights/best.pt"
DEFAULT_SOURCE = "2048246-hd_1920_1080_24fps.mp4"  # change or use "0" for webcam
CONF_THRESH = 0.35
IOU_THRESH = 0.45
# classes to show (from your trained model): glove=0, helmet=1, person=2, shoe=3, vest=4
SHOW_CLASSES = [0, 1, 3, 4]  # exclude 'person' class
# colors per class id (BGR)
CLASS_COLORS = {
    0: (220, 100, 30),   # glove
    1: (0, 215, 255),    # helmet
    3: (255, 140, 30),   # shoe
    4: (30, 200, 30),    # vest
}
ALPHA_MASK = 0.35  # transparency for filled masks

# ---------- helpers ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", "-m", default=DEFAULT_MODEL, help="Path to ppe segmentation model .pt")
    p.add_argument("--source", "-s", default=DEFAULT_SOURCE, help='Video file path or "0" for webcam')
    p.add_argument("--conf", "-c", type=float, default=CONF_THRESH, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=IOU_THRESH, help="NMS IoU threshold")
    p.add_argument("--classes", nargs="*", type=int, default=SHOW_CLASSES, help="Class ids to display")
    p.add_argument("--save", "-o", default=None, help="Optional: path to save annotated video (mp4)")
    return p.parse_args()

def ensure_int_coords(poly):
    # poly might be list of floats: convert to Nx2 int coords
    arr = np.array(poly).reshape(-1, 2)
    arr = np.round(arr).astype(np.int32)
    return arr

def draw_polygons_and_boxes(frame, result, class_filter, names, colors, alpha_mask=0.35):
    """
    Draw polygons (if available) or raster-contour masks + boxes + labels
    - result: one ultralytics result object for the frame
    - class_filter: list of class ids to visualize
    """
    h, w = frame.shape[:2]
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)

    # Prepare overlay for semi-transparent fills
    overlay = frame.copy()

    # 1) Try polygon representation r.masks.xy (list of polygons per instance)
    polys = None
    if masks is not None and hasattr(masks, "xy") and masks.xy is not None:
        polys = masks.xy  # list of lists: each item is polygon points (float coords)
    # 2) raster masks fallback
    raster_masks = None
    if masks is not None and hasattr(masks, "data") and masks.data is not None:
        try:
            # masks.data may be a torch tensor (n, H, W)
            m = masks.data.cpu().numpy()
        except Exception:
            m = np.array(masks.data)
        # m shape might be (n, h, w). Sometimes smaller; we will resize per instance later if needed
        raster_masks = m

    # gather box info arrays if boxes exist
    box_xyxy = []
    box_conf = []
    box_cls = []
    if boxes is not None and len(boxes) > 0:
        try:
            box_xyxy = boxes.xyxy.cpu().numpy()
            box_conf = boxes.conf.cpu().numpy()
            box_cls = boxes.cls.cpu().numpy().astype(int)
        except Exception:
            # fallback if not tensors
            box_xyxy = np.array(boxes.xyxy)
            box_conf = np.array(boxes.conf)
            box_cls = np.array(boxes.cls).astype(int)

    n_instances = len(box_xyxy)
    # Process each detection
    for i in range(n_instances):
        cls_id = int(box_cls[i])
        if cls_id not in class_filter:
            continue
        (x1, y1, x2, y2) = box_xyxy[i].astype(int)
        conf = float(box_conf[i])

        color = colors.get(cls_id, (0, 255, 0))
        label = f"{names.get(cls_id, str(cls_id))}:{conf:.2f}"

        # draw bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230,230,230), 1, cv2.LINE_AA)

        # draw polygon if available
        drawn_poly = False
        if polys is not None and i < len(polys):
            # polys[i] may be a list of one or more polygons (list-of-arrays)
            # ultralytics often returns a list like [[x1,y1,x2,y2,...]] or list per instance
            poly_list = polys[i]
            if poly_list is not None:
                try:
                    # poly_list could be list-of-lists; try to iterate
                    if isinstance(poly_list[0], (list, tuple, np.ndarray)):
                        for p in poly_list:
                            pts = ensure_int_coords(p)
                            cv2.fillPoly(overlay, [pts], color)
                            cv2.polylines(frame, [pts], True, color, 2, lineType=cv2.LINE_AA)
                            drawn_poly = True
                    else:
                        pts = ensure_int_coords(poly_list)
                        cv2.fillPoly(overlay, [pts], color)
                        cv2.polylines(frame, [pts], True, color, 2, lineType=cv2.LINE_AA)
                        drawn_poly = True
                except Exception:
                    drawn_poly = False

        # draw raster mask if polygons not available
        if not drawn_poly and raster_masks is not None and i < raster_masks.shape[0]:
            mask_i = raster_masks[i]
            # mask_i should align with frame; if not, resize
            if mask_i.shape[0] != h or mask_i.shape[1] != w:
                mask_i = cv2.resize((mask_i > 0.5).astype("uint8"), (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                mask_i = (mask_i > 0.5).astype("uint8")
            # find contours
            contours, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(frame, contours, -1, color, 2, lineType=cv2.LINE_AA)          # outlines
                cv2.drawContours(overlay, contours, -1, color, thickness=cv2.FILLED, lineType=cv2.LINE_AA)  # filled overlay

    # Blend overlay (filled masks) with original
    cv2.addWeighted(overlay, alpha_mask, frame, 1 - alpha_mask, 0, frame)
    return frame

# ---------- main ----------
def main():
    args = parse_args()
    model_path = args.model
    src = args.source
    conf = args.conf
    iou = args.iou
    class_filter = args.classes
    save_path = args.save

    if src.isdigit():
        src = int(src)

    print("Loading model:", model_path)
    model = YOLO(model_path)

    # Get model class names if available
    try:
        names = model.model.names
    except Exception:
        # fallback: try model.names
        names = getattr(model, "names", {})
    # ensure names maps ints to strings
    names = {int(k): str(v) for k, v in dict(names).items()}

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("ERROR: cannot open source:", src)
        return

    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
        print("Saving annotated video to:", save_path)

    win = "PPE Only Viewer"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    frame_idx = 0
    last_time = time.time()
    fps_smooth = 0.0
    alpha = 0.9

    print("Starting. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img_h, img_w = frame.shape[:2]

        # run PPE model on the frame (segmentation)
        # pass classes filter to model: only detect PPE classes
        results = model.predict(source=frame, conf=conf, iou=iou, classes=class_filter, verbose=False)

        annotated = frame.copy()
        if len(results) > 0:
            r = results[0]
            annotated = draw_polygons_and_boxes(annotated, r, class_filter, names, CLASS_COLORS, alpha_mask=ALPHA_MASK)

        # fps
        now = time.time()
        dt = now - last_time
        fps = 1.0 / dt if dt > 0 else 0.0
        fps_smooth = alpha * fps_smooth + (1 - alpha) * fps if fps_smooth > 0 else fps
        last_time = now

        cv2.putText(annotated, f"Frame:{frame_idx}  FPS:{fps_smooth:.1f}", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)

        cv2.imshow(win, annotated)
        if writer is not None:
            writer.write(annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print("Finished.")

if __name__ == "__main__":
    main()
