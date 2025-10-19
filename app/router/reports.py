from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from datetime import datetime, timedelta
from app.database import get_db
from app.models import Violation, Worker, Camera
from app.router.auth import get_current_user

router = APIRouter(prefix="/reports", tags=["Reports"])

@router.get("/")
def get_reports_summary(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
    period: str = Query("today", description="Period filter: today | last_week | last_month")
):
    user_id = current_user.id

    # Use Philippine time (UTC+8)
    from datetime import timezone
    now = datetime.now(timezone(timedelta(hours=8)))

    # Define start and end date based on selected period
    if period == "last_week":
        end_date = datetime(now.year, now.month, now.day)  # midnight today
        start_date = end_date - timedelta(days=7)          # 7 full days before today
    elif period == "last_month":
        end_date = datetime(now.year, now.month, now.day)
        start_date = end_date - timedelta(days=30)
    else:  # today
        start_date = datetime(now.year, now.month, now.day)
        end_date = start_date + timedelta(days=1)

    # Filter only records in that range
    date_filter = and_(Violation.created_at >= start_date, Violation.created_at < end_date)

    # Total incidents (violations)
    total_incidents = (
        db.query(func.count(Violation.id))
        .filter(Violation.user_id == user_id, date_filter)
        .scalar()
        or 0
    )

    # Total unique workers involved
    total_workers_involved = (
        db.query(func.count(func.distinct(Violation.worker_id)))
        .filter(Violation.user_id == user_id, date_filter)
        .scalar()
        or 0
    )

    # Compliance rate
    total_workers = db.query(func.count(Worker.id)).filter(Worker.user_id == user_id).scalar() or 1
    non_violating_workers = total_workers - total_workers_involved
    compliance_rate = round((non_violating_workers / total_workers) * 100, 2)

    # High-risk locations
    high_risk_locations = (
        db.query(Violation.camera_id)
        .filter(Violation.user_id == user_id, date_filter)
        .group_by(Violation.camera_id)
        .having(func.count(Violation.id) > 10)
        .count()
    )

    # Most common violations
    most_violations = (
        db.query(Violation.violation_types, func.count(Violation.id).label("count"))
        .filter(Violation.user_id == user_id, date_filter)
        .group_by(Violation.violation_types)
        .order_by(func.count(Violation.id).desc())
        .limit(5)
        .all()
    )

    # Top offenders
    top_offenders = (
        db.query(Worker.fullName, func.count(Violation.id).label("violations"))
        .join(Violation, Worker.id == Violation.worker_id)
        .filter(Violation.user_id == user_id, date_filter)
        .group_by(Worker.id)
        .order_by(func.count(Violation.id).desc())
        .limit(5)
        .all()
    )

    return {
        "total_incidents": total_incidents,
        "total_workers_involved": total_workers_involved,
        "compliance_rate": compliance_rate,
        "high_risk_locations": high_risk_locations,
        "most_violations": [{"name": v[0], "violators": v[1]} for v in most_violations],
        "top_offenders": [{"name": w[0], "value": w[1]} for w in top_offenders],
    }
