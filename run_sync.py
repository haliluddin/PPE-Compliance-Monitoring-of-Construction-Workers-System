#!/usr/bin/env python3
# run_sync.py - synchronous runner using onnxruntime-gpu + MediaPipe + EasyOCR
# Usage: python run_sync.py --video /path/to/video.mp4 --out out.mp4 --skip 3

import argparse, time, os
import cv2
import numpy as np
import onnxruntime as ort
from app.decision_logic import process_frame, init_pose, parse_triton_outputs  # reuse parsing
import easyocr

def make_session(onnx_path):
    # prefer CUDA provider, fallback to CPU
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    sess = ort.InferenceSession(onnx_path, providers=providers)
    return sess

def prepare_input(img, target_size=(416,416)):
    h, w = target_size
    im = cv2.resize(img, (w, h))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    im = np.transpose(im, (2,0,1))[None, ...]
    return im

def run_model(sess, inp):
    name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    res = sess.run(out_names, {name: inp})
    # convert list outputs to dict name->np.array (mimic triton outputs)
    outd = {n: (np.array(r) if r is not None else None) for n, r in zip(out_names, res)}
    return outd

def parse_person_boxes_from_outputs(outputs, H, W):
    # mimic tasks._parse_person... simple parser
    arrays = []
    for v in outputs.values():
        if v is None: continue
        a = np.array(v)
        if a.ndim == 2 and a.shape[0] > 0:
            arrays.append(a)
    if not arrays:
        return []
    arr = arrays[0]
    boxes = []
    if arr.ndim == 2 and arr.shape[1] >= 6:
        if arr.shape[1] == 6:
            for row in arr:
                x1,y1,x2,y2,conf,cls = row[:6]
                # if normalized coords
                if max(x1,x2,y1,y2) <= 1.01:
                    x1c = float(x1) * W; y1c = float(y1) * H
                    x2c = float(x2) * W; y2c = float(y2) * H
                else:
                    x1c = float(x1); y1c = float(y1); x2c = float(x2); y2c = float(y2)
                if int(cls) == 0:
                    boxes.append((max(0.0,x1c), max(0.0,y1c), min(W-1.0,x2c), min(H-1.0,y2c), float(conf)))
        else:
            # center format
            for row in arr:
                x,y,w,h,conf = float(row[0]),float(row[1]),float(row[2]),float(row[3]),float(row[4])
                probs = row[5:]
                if probs.size>0:
                    cls = int(np.argmax(probs)); cls_conf = float(probs[cls])
                else:
                    cls = int(row[-1]) if row.size>5 else -1; cls_conf = 1.0
                final_conf = float(conf * cls_conf)
                if max(x,y,w,h) <= 1.01:
                    cx = x * W; cy = y * H; ww = w * W; hh = h * H
                else:
                    cx = x; cy = y; ww = w; hh = h
                x1 = cx - ww/2; y1 = cy - hh/2; x2 = cx + ww/2; y2 = cy + hh/2
                x1c = max(0.0, x1); y1c = max(0.0, y1); x2c = min(W-1.0, x2); y2c = min(H-1.0, y2)
                if cls == 0:
                    boxes.append((x1c, y1c, x2c, y2c, final_conf))
    return boxes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="annotated_out.mp4")
    ap.add_argument("--person_onnx", default="triton_model_repo/person_yolo/1/model.onnx")
    ap.add_argument("--ppe_onnx", default="triton_model_repo/ppe_yolo/1/model.onnx")
    ap.add_argument("--skip", type=int, default=3)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    # create sessions
    print("Loading ONNX models...")
    person_sess = make_session(args.person_onnx)
    ppe_sess = make_session(args.ppe_onnx)

    # output names (we'll keep output name lists for parse_triton_outputs)
    ppe_out_names = [o.name for o in ppe_sess.get_outputs()]

    # EasyOCR reader (GPU)
    reader = easyocr.Reader(['en'], gpu=True)  # allow GPU

    # mediapipe pose
    pose = init_pose()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video: " + args.video)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.out, fourcc, fps / max(1,args.skip), (W, H))

    frame_idx = 0
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret: break
        if args.skip > 1 and (frame_idx % args.skip) != 0:
            frame_idx += 1
            continue

        # Person detection
        inp_person = prepare_input(frame, (416,416))
        person_out = run_model(person_sess, inp_person)
        person_boxes = parse_person_boxes_from_outputs(person_out, H, W)

        # PPE detection (run on full frame for now)
        inp_ppe = prepare_input(frame, (416,416))
        ppe_out = run_model(ppe_sess, inp_ppe)

        # process_frame expects triton-like outputs dict and person_boxes list
        # re-use existing process_frame. use triton_outputs = ppe_out
        try:
            result = process_frame(frame, triton_client=None, triton_model_name=None,
                                   input_name=None, output_names=ppe_out_names,
                                   triton_outputs=ppe_out, ocr_reader=reader, regset=set(),
                                   pose_instance=pose, person_boxes=person_boxes)
        except Exception as e:
            print("process_frame error:", e)
            result = {"annotated_bgr": frame, "people": []}

        annotated = result.get("annotated_bgr") or frame
        writer.write(annotated)
        if args.show:
            cv2.imshow("annotated", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_idx += 1
        if frame_idx % 10 == 0:
            elapsed = time.time() - t0
            print(f"frame {frame_idx} processed, time/frames: {elapsed:.2f}s total")

    cap.release()
    writer.release()
    print("Saved annotated video to:", args.out)

if __name__ == "__main__":
    main()
