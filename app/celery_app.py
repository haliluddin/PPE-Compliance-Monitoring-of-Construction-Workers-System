# app/celery_app.py
from celery import Celery
import os

CELERY_BROKER = os.environ.get("CELERY_BROKER", "redis://redis:6379/0")
CELERY_BACKEND = os.environ.get("CELERY_BACKEND", "redis://redis:6379/1")

celery = Celery(
    "ppe_tasks",
    broker=CELERY_BROKER,
    backend=CELERY_BACKEND,
    include=["app.tasks"]
)

celery.conf.update(
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=200
)
