# run_local_pipeline.py
import os
import time
import argparse
import numpy as np
import cv2
import onnxruntime as ort
import easyocr
from app.decision_logic import process_frame, init_pose

def get_model_input_size(sess, default=416):
    try:
        shape = sess.get_inputs()[0].shape
        parsed = []
        for s in shape:
            if isinstance(s, int):
                parsed.append(s)
            else:
                try:
                    parsed.append(int(s))
                except Exception:
                    parsed.append(None)
        if len(parsed) >= 4 and parsed[2] is not None and parsed[3] is not None:
            return int(parsed[2])
        ints = [x for x in parsed if isinstance(x, int)]
        if len(ints) >= 2:
            return int(ints[-2])
    except Exception:
        pass
    return int(default)

def make_session(onnx_path, providers=None, intra_threads=1, inter_threads=1):
    if providers is None:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    so = ort.SessionOptions()
    so.intra_op_num_threads = intra_threads
    so.inter_op_num_threads = inter_threads
    return ort.InferenceSession(onnx_path, sess_options=so, providers=providers)

def prepare_image_for_onnx(img_bgr, size=(416,416)):
    im = cv2.resize(img_bgr, size)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype('float32') / 255.0
    im = np.transpose(im, (2,0,1))[None, ...]
    return im.astype('float32')

def parse_person_outputs_from_onnx(arr, H, W):
    boxes = []
    if arr is None:
        return boxes
    if isinstance(arr, list):
        if len(arr) > 0:
            arr = np.array(arr[0])
    arr = np.array(arr)
    if arr.ndim != 2:
        return boxes
    if arr.shape[1] == 6:
        for row in arr:
            x1,y1,x2,y2,conf,cls = row[:6]
            cls = int(cls)
            if max(x1, x2, y1, y2) <= 1.01:
                x1c = float(x1) * W
                y1c = float(y1) * H
                x2c = float(x2) * W
                y2c = float(y2) * H
            else:
                x1c, y1c, x2c, y2c = float(x1), float(y1), float(x2), float(y2)
            if cls == 0:
                boxes.append((max(0.0, x1c), max(0.0, y1c), min(W - 1.0, x2c), min(H - 1.0, y2c), float(conf)))
    else:
        for row in arr:
            x,y,w,h,conf = float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])
            probs = row[5:]
            if probs.size > 0:
                cls = int(np.argmax(probs))
                cls_conf = float(probs[cls])
            else:
                cls = int(row[-1]) if row.size > 5 else -1
                cls_conf = 1.0
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

def run_infer(sess, inp_name, out_names, img_tensor):
    res = sess.run(out_names, {inp_name: img_tensor})
    return {n: np.array(v) for n, v in zip(out_names, res)}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--person-onnx", default="triton_model_repo/person_yolo/1/model.onnx")
    p.add_argument("--ppe-onnx", default="triton_model_repo/ppe_yolo/1/model.onnx")
    p.add_argument("--out", default="annotated_out.mp4")
    p.add_argument("--size", type=int, default=416)
    p.add_argument("--frame-skip", type=int, default=3)
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    try:
        person_sess = make_session(args.person_onnx, providers=providers)
    except Exception:
        person_sess = make_session(args.person_onnx, providers=['CPUExecutionProvider'])
    p_input = person_sess.get_inputs()[0].name
    p_outs = [o.name for o in person_sess.get_outputs()]
    person_input_size = get_model_input_size(person_sess, default=args.size)
    try:
        ppe_sess = make_session(args.ppe_onnx, providers=providers)
    except Exception:
        ppe_sess = make_session(args.ppe_onnx, providers=['CPUExecutionProvider'])
    pe_input = ppe_sess.get_inputs()[0].name
    pe_outs = [o.name for o in ppe_sess.get_outputs()]
    ppe_input_size = get_model_input_size(ppe_sess, default=args.size)

    pose = init_pose()
    try:
        ocr = easyocr.Reader(['en'], gpu=True)
    except Exception:
        ocr = easyocr.Reader(['en'], gpu=False)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit("Cannot open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    outv = cv2.VideoWriter(args.out, fourcc, max(1.0, fps/args.frame_skip), (W, H))
    frame_idx = 0
    t0 = time.time()
    processed = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if args.frame_skip <= 1 or (frame_idx % args.frame_skip) == 0:
                inp_person = prepare_image_for_onnx(frame, size=(person_input_size, person_input_size))
                try:
                    p_res = person_sess.run(p_outs, {p_input: inp_person})
                except Exception:
                    p_res = [np.array([])]
                arr = None
                for a in p_res:
                    a = np.array(a)
                    if a.ndim == 2 and a.shape[0] > 0 and a.shape[1] >= 5:
                        arr = a
                        break
                person_boxes = parse_person_outputs_from_onnx(arr if arr is not None else np.array([]), H, W)
                inp_ppe = prepare_image_for_onnx(frame, size=(ppe_input_size, ppe_input_size))
                try:
                    pe_res_list = ppe_sess.run(pe_outs, {pe_input: inp_ppe})
                    triton_like = {n: np.array(v) for n, v in zip(pe_outs, pe_res_list)}
                except Exception:
                    triton_like = {}
                result = process_frame(frame, triton_client=None, triton_model_name=None, input_name=None, output_names=None, triton_outputs=triton_like, ocr_reader=ocr, regset=set(), pose_instance=pose, person_boxes=person_boxes)
                annotated = result.get("annotated_bgr")
                if annotated is None:
                    annotated = frame
                outv.write(annotated)
                if args.show:
                    cv2.imshow("annot", annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                processed += 1
            frame_idx += 1
    finally:
        cap.release()
        outv.release()
        if args.show:
            cv2.destroyAllWindows()
    t1 = time.time()
    print(f"Processed {processed} frames in {t1-t0:.2f}s — approx {processed/(t1-t0):.2f} FPS")
    print("Output file:", args.out)

if __name__ == "__main__":
    main()
