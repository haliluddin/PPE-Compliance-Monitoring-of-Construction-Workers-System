from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, case
from datetime import datetime, timedelta, timezone
from app.database import get_db
from app.models import Violation, Worker, Camera
from app.router.auth import get_current_user
import csv
from fastapi.responses import StreamingResponse
from io import StringIO

router = APIRouter(prefix="/reports", tags=["Reports"])
PH_TZ = timezone(timedelta(hours=8))

def _period_bounds(period: str):
    now_ph = datetime.now(PH_TZ)
    today_start_ph = now_ph.replace(hour=0, minute=0, second=0, microsecond=0)
    if period == "last_week":
        end_ph = today_start_ph
        start_ph = end_ph - timedelta(days=7)
        days = 7
    elif period == "last_month":
        end_ph = today_start_ph
        start_ph = end_ph - timedelta(days=30)
        days = 30
    else:
        start_ph = today_start_ph
        end_ph = start_ph + timedelta(days=1)
        days = 1
    start_utc = start_ph.astimezone(timezone.utc)
    end_utc = end_ph.astimezone(timezone.utc)
    return start_ph, start_utc, end_utc, days

def _to_iso_ph(dt):
    if dt is None:
        return None
    try:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(PH_TZ).isoformat()
    except Exception:
        try:
            return dt.isoformat()
        except Exception:
            return None

@router.get("/")
def get_reports_summary(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
    period: str = Query("today", description="Period filter: today | last_week | last_month")
):
    user_id = current_user.id
    start_ph, start_utc, end_utc, days = _period_bounds(period)

    total_incidents = (
        db.query(func.count(Violation.id))
        .filter(Violation.user_id == user_id, Violation.created_at >= start_utc, Violation.created_at < end_utc)
        .scalar() or 0
    )

    total_workers_involved = (
        db.query(func.count(func.distinct(Violation.worker_id)))
        .filter(Violation.user_id == user_id, Violation.created_at >= start_utc, Violation.created_at < end_utc)
        .scalar() or 0
    )

    resolved_count = (
        db.query(func.count(Violation.id))
        .filter(Violation.user_id == user_id, Violation.created_at >= start_utc, Violation.created_at < end_utc, func.lower(func.coalesce(Violation.status, "")) == "resolved")
        .scalar() or 0
    )

    total_violations = total_incidents

    violation_resolution_rate = round((resolved_count / total_violations) * 100, 2) if total_violations > 0 else 0

    high_risk_locations = (
        db.query(Violation.camera_id)
        .filter(Violation.user_id == user_id, Violation.created_at >= start_utc, Violation.created_at < end_utc)
        .group_by(Violation.camera_id)
        .having(func.count(Violation.id) > 10)
        .count()
    )

    most_violations_raw = (
        db.query(Violation.violation_types, func.count(Violation.id).label("count"))
        .filter(Violation.user_id == user_id, Violation.created_at >= start_utc, Violation.created_at < end_utc)
        .group_by(Violation.violation_types)
        .order_by(func.count(Violation.id).desc())
        .limit(5)
        .all()
    )
    most_violations = [{"name": mv[0], "violations": mv[1]} for mv in most_violations_raw]

    top_offenders_raw = (
        db.query(Worker.fullName, func.count(Violation.id).label("violations"))
        .join(Violation, Worker.id == Violation.worker_id)
        .filter(Violation.user_id == user_id, Violation.created_at >= start_utc, Violation.created_at < end_utc)
        .group_by(Worker.id)
        .order_by(func.count(Violation.id).desc())
        .limit(5)
        .all()
    )
    top_offenders = [{"name": t[0], "value": t[1]} for t in top_offenders_raw]

    camera_stats = (
        db.query(
            Camera.name.label("location"),
            func.count(Violation.id).label("violations")
        )
        .outerjoin(Violation, and_(
            Camera.id == Violation.camera_id,
            Violation.user_id == user_id,
            Violation.created_at >= start_utc,
            Violation.created_at < end_utc
        ))
        .filter(Camera.user_id == user_id)
        .group_by(Camera.id)
        .all()
    )

    camera_data = []
    for c in camera_stats:
        if c.violations > 10:
            risk = "High"
        elif c.violations >= 5:
            risk = "Medium"
        else:
            risk = "Low"
        camera_data.append({
            "location": c.location,
            "violations": c.violations,
            "risk": risk
        })

    worker_resolution_stats = (
        db.query(
            Worker.id,
            Worker.fullName.label("name"),
            func.count(Violation.id).label("total_violations"),
            func.sum(case((func.lower(func.coalesce(Violation.status, "")) == "resolved", 1), else_=0)).label("resolved_violations")
        )
        .outerjoin(Violation, and_(
            Worker.id == Violation.worker_id,
            Violation.user_id == user_id,
            Violation.created_at >= start_utc,
            Violation.created_at < end_utc
        ))
        .filter(Worker.user_id == user_id)
        .group_by(Worker.id)
        .all()
    )

    worker_data = []
    for rank, w in enumerate(worker_resolution_stats, start=1):
        total = int(w.total_violations or 0)
        resolved = int(w.resolved_violations or 0)
        resolution_rate = round((resolved / total) * 100, 2) if total > 0 else 0
        worker_data.append({
            "rank": rank,
            "name": w.name,
            "violations": total,
            "resolved": resolved,
            "resolution_rate": resolution_rate
        })

    return {
        "total_incidents": total_incidents,
        "total_workers_involved": total_workers_involved,
        "violation_resolution_rate": violation_resolution_rate,
        "high_risk_locations": high_risk_locations,
        "most_violations": most_violations,
        "top_offenders": top_offenders,
        "camera_data": camera_data,
        "worker_data": worker_data
    }

@router.get("/performance")
def get_performance_data(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
    period: str = Query("today", description="Period filter: today | last_week | last_month")
):
    user_id = current_user.id
    start_ph, start_utc, end_utc, days = _period_bounds(period)
    rows = db.query(Violation).filter(Violation.user_id == user_id, Violation.created_at >= start_utc, Violation.created_at < end_utc).all()

    date_buckets = {}
    for i in range(days):
        d = (start_ph + timedelta(days=i)).strftime("%Y-%m-%d")
        date_buckets[d] = {"violations": 0, "compliance": 0}

    response_times = []
    for v in rows:
        dt = getattr(v, "created_at", None)
        if dt is None:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        date_key = dt.astimezone(PH_TZ).strftime("%Y-%m-%d")
        if date_key not in date_buckets:
            date_buckets[date_key] = {"violations": 0, "compliance": 0}
        date_buckets[date_key]["violations"] += 1
        if getattr(v, "status", "") and str(getattr(v, "status", "")).lower() == "resolved":
            date_buckets[date_key]["compliance"] += 1
        rt = getattr(v, "response_time", None)
        if isinstance(rt, (int, float)):
            response_times.append(float(rt))
        else:
            resolved_at = getattr(v, "resolved_at", None)
            created_at = getattr(v, "created_at", None)
            try:
                if resolved_at is not None and created_at is not None:
                    if resolved_at.tzinfo is None:
                        resolved_at = resolved_at.replace(tzinfo=timezone.utc)
                    if created_at.tzinfo is None:
                        created_at = created_at.replace(tzinfo=timezone.utc)
                    delta_min = (resolved_at - created_at).total_seconds() / 60.0
                    if delta_min >= 0:
                        response_times.append(delta_min)
            except Exception:
                pass

    performance_over_time = [{"date": k, "violations": v["violations"], "compliance": v["compliance"]} for k, v in sorted(date_buckets.items())]
    average_response_time = round(sum(response_times) / len(response_times), 2) if response_times else 0

    return {"performance_over_time": performance_over_time, "average_response_time": average_response_time}

@router.get("/export")
def export_reports(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
    period: str = Query("today", description="Period filter: today | last_week | last_month")
):
    user_id = current_user.id
    start_ph, start_utc, end_utc, days = _period_bounds(period)

    violations = db.query(
        Violation.id,
        Violation.violation_types,
        Violation.created_at,
        Violation.resolved_at,
        Worker.fullName.label("worker_name"),
        Camera.name.label("camera_location")
    ).outerjoin(Worker, Worker.id == Violation.worker_id)\
     .outerjoin(Camera, Camera.id == Violation.camera_id)\
     .filter(Violation.user_id == user_id, Violation.created_at >= start_utc, Violation.created_at < end_utc)\
     .order_by(Violation.created_at.desc())\
     .all()

    csv_file = StringIO()
    writer = csv.writer(csv_file)
    writer.writerow(["ID", "Violation Type", "Date", "Worker Name", "Camera Location", "Resolved At"])
    for v in violations:
        writer.writerow([v.id, v.violation_types, _to_iso_ph(v.created_at), v.worker_name or "", v.camera_location or "", _to_iso_ph(getattr(v, "resolved_at", None))])

    csv_file.seek(0)
    date_str = start_ph.strftime("%Y%m%d")
    filename = f"report_{period}_{date_str}.csv"

    return StreamingResponse(
        csv_file,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )
