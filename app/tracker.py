import time
from threading import Lock
from collections import defaultdict

class SimpleTracker:
    def __init__(self, iou_thresh=0.3, lost_frames_thresh=30, persist_K=3, persist_M=5):
        self.iou_thresh = float(iou_thresh)
        self.lost_frames_thresh = int(lost_frames_thresh)
        self.persist_K = int(persist_K)
        self.persist_M = int(persist_M)
        self.tracks = {}
        self.next_id = 1
        self.lock = Lock()

    def iou(self, A, B):
        xA = max(A[0], B[0]); yA = max(A[1], B[1]); xB = min(A[2], B[2]); yB = min(A[3], B[3])
        interW = max(0.0, xB - xA); interH = max(0.0, yB - yA); inter = interW * interH
        aA = max(0.0, A[2] - A[0]) * max(0.0, A[3] - A[1]); aB = max(0.0, B[2] - B[0]) * max(0.0, B[3] - B[1])
        u = aA + aB - inter
        return inter / u if u > 0 else 0.0

    def match_boxes(self, boxes, frame_idx):
        with self.lock:
            assignments = {}
            for bi, box in enumerate(boxes):
                best_tid = None
                best_iou = 0.0
                for tid, t in self.tracks.items():
                    if frame_idx - t["last_seen_frame"] > self.lost_frames_thresh:
                        continue
                    i = self.iou(box, t["bbox"])
                    if i > best_iou and i >= self.iou_thresh:
                        best_iou = i
                        best_tid = tid
                if best_tid is not None:
                    assignments[bi] = best_tid
                    self.tracks[best_tid]["bbox"] = box
                    self.tracks[best_tid]["last_seen_frame"] = frame_idx
                    self.tracks[best_tid]["age"] += 1
                else:
                    tid = self.next_id
                    self.next_id += 1
                    self.tracks[tid] = {
                        "bbox": box,
                        "first_seen_frame": frame_idx,
                        "last_seen_frame": frame_idx,
                        "age": 1,
                        "viol_buf": [],
                        "id_buf": [],
                        "started_events": set(),
                        "agg_violation_id": None,
                        "finalized": False
                    }
                    assignments[bi] = tid
            return assignments

    def _should_start_violation(self, track, vtype):
        buf = track["viol_buf"][-self.persist_M:]
        count = sum(1 for vset in buf if vtype in vset)
        return count >= self.persist_K

    def _resolve_best_id_from_buf(self, track):
        entries = [i for i,c in track["id_buf"] if i is not None]
        if not entries:
            return None
        candidates = {}
        for e in entries:
            candidates[e] = candidates.get(e, 0) + 1
        best = max(candidates.items(), key=lambda x: x[1])[0]
        return best

    def finalize_stale(self, frame_idx, sess):
        to_remove = []
        with self.lock:
            for tid, t in list(self.tracks.items()):
                if frame_idx - t["last_seen_frame"] > self.lost_frames_thresh and not t.get("finalized"):
                    t["finalized"] = True
                    to_remove.append(tid)
            for tid in to_remove:
                try:
                    self.tracks.pop(tid, None)
                except Exception:
                    pass

    def update_tracks(self, people, frame_idx, frame_ts, annotated_snapshot, sess, job_id=None, camera_id=None, job_user_id=None):
        boxes = []
        for p in people:
            b = p.get("bbox")
            if b is None:
                boxes.append((0,0,0,0))
            else:
                boxes.append(tuple([float(x) for x in b]))
        assignments = self.match_boxes(boxes, frame_idx)
        results = []
        with self.lock:
            for bi, p in enumerate(people):
                tid = assignments.get(bi)
                if tid is None:
                    continue
                track = self.tracks.get(tid)
                viols = set(p.get("violations", []))
                id_label = p.get("id")
                id_conf = p.get("id_conf", 0.0) if isinstance(p.get("id_conf", None), (int,float)) else 0.0
                track["viol_buf"].append(viols)
                track["id_buf"].append((id_label, id_conf))
                if len(track["viol_buf"]) > 60:
                    track["viol_buf"] = track["viol_buf"][-60:]
                if len(track["id_buf"]) > 60:
                    track["id_buf"] = track["id_buf"][-60:]
                best_id = self._resolve_best_id_from_buf(track)
                worker_code = None
                worker_obj = None
                if best_id is not None:
                    s = str(best_id)
                    if s.startswith("UNREG:"):
                        worker_code = s.split(":",1)[1]
                    else:
                        worker_code = s
                    try:
                        from app.models import Worker
                        worker_obj = sess.query(Worker).filter(Worker.worker_code == worker_code).first() if worker_code else None
                    except Exception:
                        worker_obj = None
                current_started = set()
                for vtype in viols:
                    if vtype in track["started_events"]:
                        current_started.add(vtype)
                    else:
                        if self._should_start_violation(track, vtype):
                            current_started.add(vtype)
                if current_started:
                    if track.get("agg_violation_id") is None:
                        try:
                            from app.models import Violation, Notification
                            snap_bytes = annotated_snapshot
                            inference_json = {"person": p}
                            should_save = True if worker_obj is not None and getattr(worker_obj, "registered", True) else False
                            if should_save:
                                vio = Violation(
                                    job_id=job_id,
                                    camera_id=camera_id,
                                    worker_id=worker_obj.id if worker_obj is not None else None,
                                    worker_code=worker_code,
                                    violation_types=",".join(sorted(current_started)),
                                    frame_index=frame_idx,
                                    frame_ts=frame_ts,
                                    snapshot=snap_bytes,
                                    inference=inference_json,
                                    status="pending",
                                    user_id=job_user_id
                                )
                                sess.add(vio)
                                sess.commit()
                                sess.refresh(vio)
                                track["agg_violation_id"] = vio.id
                                try:
                                    notif = Notification(
                                        message=f"New violation: {vio.violation_types} ({worker_code})",
                                        is_read=False,
                                        created_at=frame_ts if frame_ts else None,
                                        user_id=vio.user_id,
                                        violation_id=vio.id
                                    )
                                    sess.add(notif)
                                    sess.commit()
                                except Exception:
                                    sess.rollback()
                            track["started_events"].update(current_started)
                        except Exception:
                            sess.rollback()
                    else:
                        try:
                            from app.models import Violation
                            vio = sess.query(Violation).filter(Violation.id == track["agg_violation_id"]).first()
                            if vio:
                                existing = set([s.strip() for s in (vio.violation_types or "").split(",") if s.strip()])
                                newset = existing.union(current_started)
                                vio.violation_types = ",".join(sorted(newset))
                                if annotated_snapshot is not None:
                                    vio.snapshot = annotated_snapshot
                                vio.frame_index = frame_idx
                                vio.frame_ts = frame_ts
                                vio.inference = {"person": p}
                                sess.commit()
                                track["started_events"].update(current_started)
                        except Exception:
                            sess.rollback()
                results.append({"track_id": tid, "best_id": best_id, "violations": list(viols), "agg_violation_id": track.get("agg_violation_id")})
        self.finalize_stale(frame_idx, sess)
        return results

_tracker_singleton = None
_tracker_lock = Lock()

def get_global_tracker():
    global _tracker_singleton
    with _tracker_lock:
        if _tracker_singleton is None:
            _tracker_singleton = SimpleTracker()
        return _tracker_singleton
