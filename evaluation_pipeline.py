import os
import sys
import json
import cv2
import math
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict, namedtuple
import matplotlib.pyplot as plt
from datetime import timedelta, datetime

# ensure project root import works
# run from repo root with `export PYTHONPATH=.`
try:
    from app.tasks import _init_local_yolo
    from app.decision_logic import process_frame, init_pose
except Exception as e:
    print("ERROR importing app modules. Ensure you run from repo root with PYTHONPATH=. and that dependencies are installed.")
    raise

# ---------------------------
# Helper: Levenshtein distance (pure python)
# ---------------------------
def levenshtein(a: str, b: str) -> int:
    a = a or ""
    b = b or ""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    # allocate matrix 2 rows
    prev = list(range(m + 1))
    cur = [0] * (m + 1)
    for i in range(1, n + 1):
        cur[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev, cur = cur, prev
    return prev[m]

def cer(pred: str, gt: str) -> float:
    if gt is None:
        return 1.0
    if len(gt) == 0:
        return 1.0 if pred else 0.0
    return levenshtein(pred or "", gt or "") / max(1, len(gt))

# ---------------------------
# Build ground truth from your provided site summary
# ---------------------------
def build_default_ground_truth():
    # based on the user-provided info in the message
    # Each worker entry: map worker_code -> list of violations expected
    # Violation taxonomy: NO HELMET, NO VEST, NO LEFT GLOVE, NO RIGHT GLOVE, NO LEFT SHOE, NO RIGHT SHOE, IMPROPER HELMET, IMPROPER LEFT GLOVE, IMPROPER RIGHT GLOVE
    sites = {}
    # Site A
    sites['siteA'] = {
        'angles': ['near','between','far'],
        'workers': {
            "2":    {"helmet": True, "vest": True,  "left_glove": True, "right_glove": True, "left_shoe": True, "right_shoe": True},
            "751":  {"helmet": True, "vest": True,  "left_glove": False,"right_glove": False,"left_shoe": False,"right_shoe": False},
            "81":   {"helmet": False,"vest": False, "left_glove": True, "right_glove": True, "left_shoe": True, "right_shoe": True},
            "65":   {"helmet": True, "vest": False, "left_glove": True, "right_glove": False,"left_shoe": False,"right_shoe": False},
            "76":   {"helmet": False,"vest": True,  "left_glove": False,"right_glove": False,"left_shoe": True, "right_shoe": True},
        }
    }
    # Site B (same as A)
    sites['siteB'] = sites['siteA'].copy()
    sites['siteB'] = dict(sites['siteA'])
    # Site C
    sites['siteC'] = {
        'angles': ['all'],
        'workers': {
            "23":  {"helmet": True, "vest": True,  "left_glove": True, "right_glove": True, "left_shoe": True, "right_shoe": True},
            "6":   {"helmet": True, "vest": False, "left_glove": True, "right_glove": True, "left_shoe": True, "right_shoe": True},
            "3":   {"helmet": True, "vest": True,  "left_glove": False,"right_glove": False,"left_shoe": True, "right_shoe": True},
            "698": {"helmet": True, "vest": True,  "left_glove": True, "right_glove": True, "left_shoe": False,"right_shoe": False},
        }
    }
    # Site D, E, F (same per user)
    sDEF_workers = {
            "325": {"helmet": True, "vest": True, "left_glove": True, "right_glove": True, "left_shoe": True, "right_shoe": True},
            "23":  {"helmet": True, "vest": False,"left_glove": False,"right_glove": False,"left_shoe": True, "right_shoe": True},
            "6":   {"helmet": False,"vest": True, "left_glove": True, "right_glove": True, "left_shoe": False,"right_shoe": False},
            "201": {"helmet": False,"vest": False,"left_glove": False,"right_glove": False,"left_shoe": False,"right_shoe": False},
            "3":   {"helmet": True, "vest": False,"left_glove": False,"right_glove": False,"left_shoe": False,"right_shoe": False, "improper_helmet": True, "improper_left_glove": True},
    }
    for s in ['siteD','siteE','siteF']:
        sites[s] = {'angles':['near','between','far'], 'workers': sDEF_workers.copy()}

    # Create per-worker expected violation sets (frame-level)
    for sname, sdata in sites.items():
        for wcode, ppe in sdata['workers'].items():
            vset = set()
            if not ppe.get('helmet', False):
                vset.add("NO HELMET")
            else:
                if ppe.get('improper_helmet', False):
                    vset.add("IMPROPER HELMET")
            if not ppe.get('vest', False):
                vset.add("NO VEST")
            if not ppe.get('left_glove', False):
                vset.add("NO LEFT GLOVE")
            else:
                if ppe.get('improper_left_glove', False):
                    vset.add("IMPROPER LEFT GLOVE")
            if not ppe.get('right_glove', False):
                vset.add("NO RIGHT GLOVE")
            else:
                if ppe.get('improper_right_glove', False):
                    vset.add("IMPROPER RIGHT GLOVE")
            if not ppe.get('left_shoe', False):
                vset.add("NO LEFT SHOE")
            if not ppe.get('right_shoe', False):
                vset.add("NO RIGHT SHOE")
            sdata['workers'][wcode]['violations'] = vset
    return sites

# ---------------------------
# Run inference on video frames and collect per-frame predictions
# ---------------------------
def run_inference_on_video(video_path, site_name, angle_name, ly, pose_instance, frame_step=30, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(frame_step))
    results = []
    frame_idx = 0
    pbar = tqdm(total=frame_count, desc=f"{site_name}/{angle_name}", unit="f")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if (frame_idx % step) == 0:
            # get person boxes and ppe outputs via local yolo wrapper
            try:
                person_boxes = ly.predict_person_boxes(frame)
            except Exception:
                person_boxes = []
            try:
                ppe_out = ly.predict_ppe_outputs(frame)
            except Exception:
                ppe_out = {}
            # ppe_out is a dict like {"ppe_boxes": np.array([...])} -> feed into process_frame as triton_outputs
            # NOTE: process_frame expects triton_outputs dict compatible with parse_triton_outputs
            triton_outputs = ppe_out
            res = process_frame(frame.copy(), triton_client=None, triton_model_name=None, input_name=None,
                                output_names=None, triton_outputs=triton_outputs, ocr_reader=None,
                                regset=set(), pose_instance=pose_instance, person_boxes=person_boxes)
            # result contains 'people' list of dicts: {"bbox", "id", "violations"}
            meta = {"site": site_name, "angle": angle_name, "frame_idx": frame_idx, "timestamp": time.time()}
            results.append({"meta": meta, "people": res.get("people", []), "boxes_by_class": res.get("boxes_by_class", {})})
        frame_idx += 1
        pbar.update(1)
        if max_frames and frame_idx >= max_frames:
            break
    pbar.close()
    cap.release()
    return results

# ---------------------------
# Evaluation metric calculations
# ---------------------------
def compute_frame_level_metrics(predictions, ground_truth_sites):
    # predictions: list of dicts with meta and people
    # ground_truth_sites: output of build_default_ground_truth()
    violation_types = ["NO HELMET","NO VEST","NO LEFT GLOVE","NO RIGHT GLOVE","NO LEFT SHOE","NO RIGHT SHOE",
                       "IMPROPER HELMET","IMPROPER LEFT GLOVE","IMPROPER RIGHT GLOVE"]
    counts = {vt: {"tp":0,"fp":0,"fn":0} for vt in violation_types}
    id_stats = {"tp":0,"fp":0,"fn":0,"total_present":0}
    cer_list = []
    for rec in predictions:
        site = rec["meta"]["site"]
        angle = rec["meta"]["angle"]
        # for each GT worker in this site assume present (per assumptions)
        gt_workers = ground_truth_sites[site]['workers']
        # build mapping predicted id -> predicted violations
        predicted_by_worker = {}  # worker_code or UNID -> set(violations)
        for p in rec["people"]:
            pred_id = p.get("id")
            if pred_id is None:
                pred_id = "UNID"
            predicted_by_worker[pred_id] = set(p.get("violations") or [])
        # Evaluate for each gt worker (frame-level)
        for wcode, winfo in gt_workers.items():
            gt_v = winfo.get("violations", set())
            id_stats["total_present"] += 1
            # find predicted entry for that worker (by exact id match)
            pred_v = predicted_by_worker.get(wcode)
            if pred_v is None:
                # no id match - consider UNID predictions as not matched
                # count all GT violations as FN
                for vt in violation_types:
                    if vt in gt_v:
                        counts[vt]["fn"] += 1
                # ID false negative
                id_stats["fn"] += 1
            else:
                # id matched
                id_stats["tp"] += 1
                # per-violation tp/fp/fn
                for vt in violation_types:
                    p_has = vt in pred_v
                    g_has = vt in gt_v
                    if p_has and g_has:
                        counts[vt]["tp"] += 1
                    elif p_has and not g_has:
                        counts[vt]["fp"] += 1
                    elif not p_has and g_has:
                        counts[vt]["fn"] += 1
                # OCR CER: if predicted id string exists but not exact, compute CER
                if isinstance(pred_id, str) and pred_id != wcode:
                    cer_list.append(cer(pred_id, wcode))
    # compute per-violation precision/recall/f1
    metrics = {}
    for vt, c in counts.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
        metrics[vt] = {"tp":tp,"fp":fp,"fn":fn,"precision":prec,"recall":rec,"f1":f1}
    # id matching
    id_prec = id_stats["tp"] / (id_stats["tp"] + id_stats["fp"]) if (id_stats["tp"]+id_stats["fp"])>0 else 0.0
    id_rec = id_stats["tp"] / (id_stats["total_present"]) if id_stats["total_present"]>0 else 0.0
    metrics["ID_MATCH"] = {"tp":id_stats["tp"], "fn":id_stats["fn"], "precision": id_prec, "recall": id_rec}
    metrics["CER_mean"] = (sum(cer_list)/len(cer_list)) if cer_list else 0.0
    return metrics, counts

# ---------------------------
# Event-level grouping and metrics
# ---------------------------
def frames_to_events(predictions, ground_truth_sites, gap_tolerance=2):
    # returns events as dict: site->worker->violation->list of (start_frame,end_frame)
    pred_events = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    gt_events = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # We assume GT: every frame a worker is present (as per assumptions), so GT events for a worker/violation are entire frame span
    # For more realistic scenario you would parse a GT timeline.
    # Build predictions timeline
    for rec in predictions:
        site = rec["meta"]["site"]
        frame = rec["meta"]["frame_idx"]
        # index predicted violations per worker id
        for p in rec["people"]:
            pid = p.get("id") or "UNID"
            for vt in p.get("violations") or []:
                pred_events[site][pid][vt].append(frame)
    # collapse frame lists into time intervals per pid/vt
    def collapse(frames_list):
        if not frames_list:
            return []
        frames_list = sorted(set(frames_list))
        intervals = []
        start = frames_list[0]
        prev = start
        for f in frames_list[1:]:
            if f - prev <= gap_tolerance:
                prev = f
                continue
            else:
                intervals.append((start, prev))
                start = f
                prev = f
        intervals.append((start, prev))
        return intervals
    pred_intervals = defaultdict(lambda: defaultdict(dict))
    for site, workers in pred_events.items():
        for pid, vmap in workers.items():
            for vt, frames_list in vmap.items():
                pred_intervals[site][pid][vt] = collapse(frames_list)
    # GT events: assume workers present in entire video range -> we need min/max frames per site from predictions
    frames_per_site = defaultdict(list)
    for rec in predictions:
        frames_per_site[rec["meta"]["site"]].append(rec["meta"]["frame_idx"])
    for site, sdata in ground_truth_sites.items():
        all_frames = sorted(set(frames_per_site.get(site, [])))
        if not all_frames:
            continue
        gs = min(all_frames); ge = max(all_frames)
        for wcode in sdata['workers'].keys():
            for vt in sdata['workers'][wcode]['violations']:
                gt_events[site][wcode][vt].append((gs, ge))
    return pred_intervals, gt_events

def match_events(pred_intervals, gt_events, iou_threshold=0.5):
    # For each gt event, see if any predicted event overlaps by temporal IoU >= threshold.
    metrics = {}
    for site, workers in gt_events.items():
        site_metrics = defaultdict(lambda: {"tp":0,"fp":0,"fn":0})
        # gather preds
        preds_for_site = pred_intervals.get(site, {})
        # Evaluate per worker/violation
        for wcode, vmap in workers.items():
            for vt, gt_list in vmap.items():
                for gt_int in gt_list:
                    gs, ge = gt_int
                    # find predicted intervals for the same worker id (exact match) and the same vt
                    matched = False
                    for pred_id, pvmap in preds_for_site.items():
                        pred_list = pvmap.get(vt, [])
                        # for event-level matching we require pred_id == wcode (ID matching)
                        if pred_id != wcode:
                            continue
                        for pi in pred_list:
                            ps, pe = pi
                            inter = max(0, min(ge, pe) - max(gs, ps) + 1)
                            union = (ge - gs + 1) + (pe - ps + 1) - inter
                            tiou = inter / union if union>0 else 0.0
                            if tiou >= iou_threshold:
                                matched = True
                                break
                        if matched:
                            break
                    if matched:
                        site_metrics[vt]["tp"] += 1
                    else:
                        site_metrics[vt]["fn"] += 1
        # false positives: predicted events that do not match any GT event (for same worker/violation)
        for pred_id, vmap in preds_for_site.items():
            for vt, pred_list in vmap.items():
                for pi in pred_list:
                    ps, pe = pi
                    matched_any = False
                    # check against GT events for same worker
                    gt_for_worker = gt_events.get(site, {}).get(pred_id, {}).get(vt, [])
                    for gs, ge in gt_for_worker:
                        inter = max(0, min(ge, pe) - max(gs, ps) + 1)
                        union = (ge - gs + 1) + (pe - ps + 1) - inter
                        if union > 0 and inter / union >= iou_threshold:
                            matched_any = True
                            break
                    if not matched_any:
                        site_metrics[vt]["fp"] += 1
        metrics[site] = site_metrics
    # convert counts to precision/recall/f1 per site/violation
    site_scores = {}
    for site, vtmap in metrics.items():
        site_scores[site] = {}
        for vt, c in vtmap.items():
            tp, fp, fn = c['tp'], c['fp'], c['fn']
            prec = tp / (tp+fp) if (tp+fp)>0 else 0.0
            rec = tp / (tp+fn) if (tp+fn)>0 else 0.0
            f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
            site_scores[site][vt] = {"tp":tp,"fp":fp,"fn":fn,"precision":prec,"recall":rec,"f1":f1}
    return site_scores

# ---------------------------
# Plotting helpers
# ---------------------------
def bar_plot_metrics(metrics_dict, title, out_png):
    # metrics_dict: {label: {"precision":..,"recall":..,"f1":..}}
    labels = list(metrics_dict.keys())
    precisions = [metrics_dict[l]["precision"] for l in labels]
    recalls = [metrics_dict[l]["recall"] for l in labels]
    f1s = [metrics_dict[l]["f1"] for l in labels]
    x = np.arange(len(labels))
    w = 0.25
    plt.figure(figsize=(max(8, len(labels)*0.6), 5))
    plt.bar(x - w, precisions, width=w, label="Precision")
    plt.bar(x, recalls, width=w, label="Recall")
    plt.bar(x + w, f1s, width=w, label="F1")
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.ylim(0,1)
    plt.ylabel("score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

# ---------------------------
# Main CLI
# ---------------------------
def main(args):
    # build default GT based on user's site summary
    ground_truth = build_default_ground_truth()

    # initialize local yolo
    ly = _init_local_yolo()
    if ly is None:
        print("Local YOLO wrapper not available (models not found). Exiting.")
        return
    pose_instance = init_pose()

    # find videos
    videos = {}
    for root, _, files in os.walk(args.videos_dir):
        for f in files:
            if f.lower().endswith(('.mp4','.mov','.mkv','.avi')):
                videos[f] = os.path.join(root, f)

    # mapping expected names -> site/angle
    expected_names = []
    # siteA..F mapping
    mapping = {
        'siteA_near.mp4': ('siteA','near'),
        'siteA_between.mp4': ('siteA','between'),
        'siteA_far.mp4': ('siteA','far'),
        'siteB_near.mp4': ('siteB','near'),
        'siteB_between.mp4': ('siteB','between'),
        'siteB_far.mp4': ('siteB','far'),
        'siteC_all.mp4': ('siteC','all'),
        'siteD_near.mp4': ('siteD','near'),
        'siteD_between.mp4': ('siteD','between'),
        'siteD_far.mp4': ('siteD','far'),
        'siteE_near.mp4': ('siteE','near'),
        'siteE_between.mp4': ('siteE','between'),
        'siteE_far.mp4': ('siteE','far'),
        'siteF_near.mp4': ('siteF','near'),
        'siteF_between.mp4': ('siteF','between'),
        'siteF_far.mp4': ('siteF','far'),
    }

    # ensure out dir exists
    os.makedirs(args.out_dir, exist_ok=True)

    all_predictions = []
    # iterate mapping and run inference if video present
    for fname, (site, angle) in mapping.items():
        path = videos.get(fname)
        if path is None:
            print(f"WARNING: expected file {fname} not found in {args.videos_dir}, skipping.")
            continue
        print(f"Processing {fname} -> site={site} angle={angle}")
        preds = run_inference_on_video(path, site, angle, ly, pose_instance, frame_step=args.frame_step, max_frames=args.max_frames)
        # persist per-video predictions
        outjson = os.path.join(args.out_dir, f"predictions_{site}_{angle}.json")
        with open(outjson, "w") as fh:
            json.dump(preds, fh, default=str)
        all_predictions.extend(preds)

    # Save the generated ground-truth (so you can later replace it with real GT CSV)
    gt_json = os.path.join(args.out_dir, "ground_truth_generated.json")
    with open(gt_json, "w") as fh:
        json.dump(ground_truth, fh, default=list)

    # compute frame-level metrics
    frame_metrics, counts = compute_frame_level_metrics(all_predictions, ground_truth)
    with open(os.path.join(args.out_dir, "frame_level_metrics.json"), "w") as fh:
        json.dump(frame_metrics, fh, indent=2, default=str)

    # compute event-level metrics
    pred_intervals, gt_events = frames_to_events(all_predictions, ground_truth, gap_tolerance=args.event_gap)
    event_scores = match_events(pred_intervals, gt_events, iou_threshold=args.event_iou)
    with open(os.path.join(args.out_dir, "event_level_metrics.json"), "w") as fh:
        json.dump(event_scores, fh, indent=2, default=str)

    # plots
    # frame-level per-violation
    vt_metrics = {k: v for k, v in frame_metrics.items() if k != "ID_MATCH" and k != "CER_mean"}
    bar_plot_metrics(vt_metrics, "Frame-level violation metrics", os.path.join(args.out_dir, "frame_level_violations.png"))

    # event-level - aggregate across sites (compute average f1 per violation)
    agg_event = {}
    for site, vtmap in event_scores.items():
        for vt, m in vtmap.items():
            agg_event.setdefault(vt, {"precision":[],"recall":[],"f1":[]})
            agg_event[vt]["precision"].append(m["precision"])
            agg_event[vt]["recall"].append(m["recall"])
            agg_event[vt]["f1"].append(m["f1"])
    agg_event_avg = {vt: {"precision": float(np.mean(vals["precision"])) if vals["precision"] else 0.0,
                          "recall": float(np.mean(vals["recall"])) if vals["recall"] else 0.0,
                          "f1": float(np.mean(vals["f1"])) if vals["f1"] else 0.0}
                     for vt, vals in agg_event.items()}
    bar_plot_metrics(agg_event_avg, "Event-level violation metrics (avg across sites)", os.path.join(args.out_dir, "event_level_violations.png"))

    # ID metrics bar
    id_metrics = {"ID_MATCH": frame_metrics.get("ID_MATCH", {"precision":0,"recall":0})}
    # show as plot
    plt.figure(figsize=(4,4))
    plt.bar(['precision','recall'], [id_metrics["ID_MATCH"]["precision"], id_metrics["ID_MATCH"]["recall"]])
    plt.ylim(0,1)
    plt.title("ID matching (frame-level)")
    plt.savefig(os.path.join(args.out_dir, "id_matching.png"))
    plt.close()

    # CER
    cer_val = frame_metrics.get("CER_mean", 0.0)
    with open(os.path.join(args.out_dir, "cer.txt"), "w") as fh:
        fh.write(f"mean_CER:{cer_val}\n")

    # Save summary CSV
    rows = []
    for vt, m in vt_metrics.items():
        rows.append({"violation":vt, "precision":m["precision"], "recall":m["recall"], "f1":m["f1"], "tp":m["tp"], "fp":m["fp"], "fn":m["fn"]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "frame_violation_summary.csv"), index=False)
    print("Evaluation results saved to", args.out_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_dir", default="./videos", help="directory with site videos")
    ap.add_argument("--out_dir", default="./eval_results", help="output directory")
    ap.add_argument("--frame_step", type=int, default=30, help="process every Nth frame (default=30)")
    ap.add_argument("--max_frames", type=int, default=None, help="max frames per video (for debug)")
    ap.add_argument("--event_gap", type=int, default=2, help="max gap (frames) to merge events")
    ap.add_argument("--event_iou", type=float, default=0.5, help="temporal IoU threshold for event match")
    args = ap.parse_args()
    main(args)
