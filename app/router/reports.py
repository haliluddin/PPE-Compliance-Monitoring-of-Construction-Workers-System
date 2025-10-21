from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, case
from datetime import datetime, timedelta
from app.database import get_db
from app.models import Violation, Worker, Camera
from app.router.auth import get_current_user
from datetime import timezone
import csv
from fastapi.responses import StreamingResponse
from io import StringIO
from sqlalchemy import cast, Date

router = APIRouter(prefix="/reports", tags=["Reports"])

@router.get("/")
def get_reports_summary(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
    period: str = Query("today", description="Period filter: today | last_week | last_month")
):
    user_id = current_user.id

    from datetime import timezone
    now = datetime.now(timezone(timedelta(hours=8)))

    # Define date range
    if period == "last_week":
        end_date = datetime(now.year, now.month, now.day)
        start_date = end_date - timedelta(days=7)
    elif period == "last_month":
        end_date = datetime(now.year, now.month, now.day)
        start_date = end_date - timedelta(days=30)
    else:
        start_date = datetime(now.year, now.month, now.day)
        end_date = start_date + timedelta(days=1)

    date_filter = and_(Violation.created_at >= start_date, Violation.created_at < end_date)

    # === Total Incidents ===
    total_incidents = (
        db.query(func.count(Violation.id))
        .filter(Violation.user_id == user_id, date_filter)
        .scalar() or 0
    )

    # === Total Workers Involved ===
    total_workers_involved = (
        db.query(func.count(func.distinct(Violation.worker_id)))
        .filter(Violation.user_id == user_id, date_filter)
        .scalar() or 0
    )

   # === Violation Resolution Rate ===
    resolved_count = (
        db.query(func.count(Violation.id))
        .filter(Violation.user_id == user_id, date_filter, Violation.status == "resolved")
        .scalar() or 0
    )

    total_violations = (
        db.query(func.count(Violation.id))
        .filter(Violation.user_id == user_id, date_filter)
        .scalar() or 0
    )

    violation_resolution_rate = (
        round((resolved_count / total_violations) * 100, 2)
        if total_violations > 0 else 0
    )


    # === High-Risk Locations ===
    high_risk_locations = (
        db.query(Violation.camera_id)
        .filter(Violation.user_id == user_id, date_filter)
        .group_by(Violation.camera_id)
        .having(func.count(Violation.id) > 10)
        .count()
    )

    # === Most Common Violations ===
    most_violations = (
        db.query(Violation.violation_types, func.count(Violation.id).label("count"))
        .filter(Violation.user_id == user_id, date_filter)
        .group_by(Violation.violation_types)
        .order_by(func.count(Violation.id).desc())
        .limit(5)
        .all()
    )

    # === Top Offenders ===
    top_offenders = (
        db.query(Worker.fullName, func.count(Violation.id).label("violations"))
        .join(Violation, Worker.id == Violation.worker_id)
        .filter(Violation.user_id == user_id, date_filter)
        .group_by(Worker.id)
        .order_by(func.count(Violation.id).desc())
        .limit(5)
        .all()
    )

    # === Camera Location Stats ===
    camera_stats = (
        db.query(
            Camera.name.label("location"),
            func.count(Violation.id).label("violations")
        )
        .outerjoin(Violation, and_(
            Camera.id == Violation.camera_id,
            Violation.user_id == user_id,
            Violation.created_at >= start_date,
            Violation.created_at < end_date
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

   # === Worker Violation Resolution Rate ===
    worker_resolution_stats = (
        db.query(
            Worker.id,
            Worker.fullName.label("name"),
            func.count(Violation.id).label("total_violations"),
            func.sum(case((Violation.status == "resolved", 1), else_=0)).label("resolved_violations")
        )
        .outerjoin(Violation, and_(
            Worker.id == Violation.worker_id,
            Violation.user_id == user_id,
            Violation.created_at >= start_date,
            Violation.created_at < end_date
        ))
        .filter(Worker.user_id == user_id)
        .group_by(Worker.id)
        .all()
    )

    worker_data = []
    for rank, w in enumerate(worker_resolution_stats, start=1):
        total = w.total_violations or 0
        resolved = w.resolved_violations or 0
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
        "most_violations": [{"name": v[0], "violators": v[1]} for v in most_violations],
        "top_offenders": [{"name": w[0], "value": w[1]} for w in top_offenders],
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
    now = datetime.now(timezone(timedelta(hours=8)))

    # Define date range
    if period == "last_week":
        end_date = datetime(now.year, now.month, now.day)
        start_date = end_date - timedelta(days=7)
    elif period == "last_month":
        end_date = datetime(now.year, now.month, now.day)
        start_date = end_date - timedelta(days=30)
    else:
        start_date = datetime(now.year, now.month, now.day)
        end_date = start_date + timedelta(days=1)

    date_filter = and_(Violation.created_at >= start_date, Violation.created_at < end_date, Violation.user_id == user_id)

    # --- Performance Over Time (daily counts) ---
  

    daily_stats = (
        db.query(
            cast(Violation.created_at, Date).label("date"),
            func.count(Violation.id).label("violations")
        )
        .filter(date_filter)
        .group_by(cast(Violation.created_at, Date))
        .order_by(cast(Violation.created_at, Date))
        .all()
    )

    performance_data = []
    for day in daily_stats:
        # Approximate compliance % = 100 - violations per day * factor
        compliance = max(0, 100 - day.violations * 5)
        performance_data.append({
            "date": day.date.strftime("%b %d"),  # e.g., "Oct 19"
            "violations": day.violations,
            "compliance": compliance
        })

    # --- Average Response Time (in minutes) ---
    # Response time = difference between frame_ts and created_at (violation detection delay)
    response_times = (
        db.query(func.avg(func.extract('epoch', Violation.created_at - Violation.frame_ts)/60))
        .filter(date_filter, Violation.frame_ts.isnot(None))
        .scalar()
    )
    
    avg_response = round(response_times or 0, 2)  # in minutes

    return {
        "performance_over_time": performance_data,
        "average_response_time": avg_response
    }



@router.get("/export")
def export_reports(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
    period: str = Query("today", description="Period filter: today | last_week | last_month")
):
    user_id = current_user.id
    now = datetime.now(timezone(timedelta(hours=8)))

    if period == "last_week":
        end_date = datetime(now.year, now.month, now.day)
        start_date = end_date - timedelta(days=7)
    elif period == "last_month":
        end_date = datetime(now.year, now.month, now.day)
        start_date = end_date - timedelta(days=30)
    else:
        start_date = datetime(now.year, now.month, now.day)
        end_date = start_date + timedelta(days=1)

    date_filter = and_(Violation.created_at >= start_date, Violation.created_at < end_date, Violation.user_id == user_id)

    violations = db.query(
        Violation.id,
        Violation.violation_types,
        Violation.created_at,
        Worker.fullName.label("worker_name"),
        Camera.name.label("camera_location")
    ).join(Worker, Worker.id == Violation.worker_id)\
     .join(Camera, Camera.id == Violation.camera_id)\
     .filter(date_filter)\
     .order_by(Violation.created_at.desc())\
     .all()

    # Prepare CSV
    csv_file = StringIO()
    writer = csv.writer(csv_file)
    writer.writerow(["ID", "Violation Type", "Date", "Worker Name", "Camera Location"])
    for v in violations:
        writer.writerow([v.id, v.violation_types, v.created_at, v.worker_name, v.camera_location])

    csv_file.seek(0)

    # Generate filename with current date
    date_str = now.strftime("%Y%m%d")  # e.g., 20251019
    filename = f"report_{period}_{date_str}.csv"

    return StreamingResponse(
        csv_file,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )
