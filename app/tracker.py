import time
from threading import Lock
from collections import defaultdict
from datetime import datetime, timezone
import logging

log = logging.getLogger(__name__)

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
            used = set()
            for bi, box in enumerate(boxes):
                if box is None:
                    assignments[bi] = None
                    continue
                best_tid = None
                best_iou = 0.0
                for tid, t in self.tracks.items():
                    if tid in used:
                        continue
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
                    used.add(best_tid)
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
                        "db_events": {},
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
        to_finalize = []
        with self.lock:
            for tid, t in list(self.tracks.items()):
                if frame_idx - t["last_seen_frame"] > self.lost_frames_thresh and not t.get("finalized"):
                    self.tracks[tid]["finalized"] = True
                    to_finalize.append(tid)
        for tid in to_finalize:
            t = self.tracks.get(tid)
            if not t:
                continue
            try:
                for vtype, vio_id in t.get("db_events", {}).items():
                    try:
                        from app.models import Violation
                        vio = sess.query(Violation).filter(Violation.id == vio_id).first()
                        if vio:
                            sess.commit()
                    except Exception:
                        sess.rollback()
            except Exception:
                pass
        with self.lock:
            for tid in to_finalize:
                try:
                    self.tracks.pop(tid, None)
                except Exception:
                    pass

    def update_tracks(self, people, frame_idx, frame_ts, annotated_snapshot, sess, job_id=None, camera_id=None, job_user_id=None):
        boxes = []
        for p in people:
            b = p.get("bbox")
            if not b or len(b) < 4:
                boxes.append(None)
            else:
                try:
                    x1, y1, x2, y2 = [float(x) for x in b[:4]]
                    if x2 <= x1 or y2 <= y1:
                        boxes.append(None)
                    else:
                        boxes.append((x1, y1, x2, y2))
                except Exception:
                    boxes.append(None)
        assignments = self.match_boxes(boxes, frame_idx)
        results = []
        with self.lock:
            for bi, p in enumerate(people):
                tid = assignments.get(bi)
                if tid is None:
                    continue
                track = self.tracks.get(tid)
                if track is None:
                    continue
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
                display_name = None
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
                persistent_new = set(v for v in viols if v not in track["started_events"] and self._should_start_violation(track, v))
                current_viols = set(viols)
                existing_ids = set(track["db_events"].get(v) for v in current_viols if track["db_events"].get(v) is not None)
                existing_id = next(iter(existing_ids), None)
                def _update_violation_row(vio_id, types_list):
                    try:
                        from app.models import Violation
                        vio = sess.query(Violation).filter(Violation.id == vio_id).first()
                        if not vio:
                            return None
                        vio.violation_types = ", ".join(sorted(types_list))
                        if annotated_snapshot is not None:
                            vio.snapshot = annotated_snapshot
                        vio.frame_index = frame_idx
                        vio.frame_ts = frame_ts
                        vio.inference = {"person": p}
                        sess.commit()
                        return vio.id
                    except Exception:
                        sess.rollback()
                        return None
                if persistent_new:
                    try:
                        from app.models import Violation, Notification
                        snap_bytes = annotated_snapshot
                        inference_json = {"person": p}
                        should_save = True if worker_obj is not None and getattr(worker_obj, "registered", True) else False
                        full_types = sorted(list(current_viols)) if current_viols else sorted(list(persistent_new))
                        if existing_id:
                            updated_id = _update_violation_row(existing_id, full_types)
                            if updated_id:
                                for vt in full_types:
                                    track["db_events"][vt] = updated_id
                        else:
                            if should_save:
                                vio = Violation(
                                    job_id=job_id,
                                    camera_id=camera_id,
                                    worker_id=worker_obj.id if worker_obj is not None else None,
                                    worker_code=worker_code,
                                    violation_types=", ".join(full_types),
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
                                for vt in full_types:
                                    track["db_events"][vt] = vio.id
                                try:
                                    created_at_val = frame_ts if frame_ts is not None else datetime.now(timezone.utc)
                                    notif = Notification(
                                        message=f"New violation: {', '.join(full_types)} ({worker_code})",
                                        is_read=False,
                                        created_at=created_at_val,
                                        user_id=vio.user_id,
                                        violation_id=vio.id
                                    )
                                    sess.add(notif)
                                    sess.commit()
                                except Exception:
                                    sess.rollback()
                        track["started_events"].update(current_viols)
                    except Exception:
                        sess.rollback()
                else:
                    if current_viols:
                        if existing_id:
                            _update_violation_row(existing_id, sorted(list(current_viols)))
                            for vt in current_viols:
                                track["db_events"][vt] = existing_id
                        else:
                            for vtype in current_viols:
                                vio_id = track["db_events"].get(vtype)
                                if vio_id:
                                    try:
                                        from app.models import Violation
                                        vio = sess.query(Violation).filter(Violation.id == vio_id).first()
                                        if vio:
                                            if annotated_snapshot is not None:
                                                vio.snapshot = annotated_snapshot
                                            vio.frame_index = frame_idx
                                            vio.frame_ts = frame_ts
                                            vio.inference = {"person": p}
                                            sess.commit()
                                    except Exception:
                                        sess.rollback()
                results.append({"track_id": tid, "best_id": best_id, "violations": list(viols)})
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
