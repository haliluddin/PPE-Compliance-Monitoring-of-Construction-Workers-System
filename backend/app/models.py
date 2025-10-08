from sqlalchemy import Column, Integer, String, Boolean, Date, Text, TIMESTAMP, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

# -----------------------------
# User Model
# -----------------------------
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_supervisor = Column(Boolean, default=False)
    workers = relationship("Worker", back_populates="user")

# -----------------------------
# Camera Model
# -----------------------------
# class Camera(Base):
#     __tablename__ = "cameras"

#     id = Column(Integer, primary_key=True, index=True)
#     name = Column(String(100), nullable=False)
#     ip_address = Column(String(45), nullable=False)
#     rtsp_url = Column(String(200), nullable=False)
#     username = Column(String(50), nullable=True)
#     password = Column(String(50), nullable=True)
#     status = Column(String(20), default="OFFLINE")


# -----------------------------
# Worker Model
# -----------------------------
class Worker(Base):
    __tablename__ = "workers"
    id = Column(Integer, primary_key=True, index=True)
    fullName = Column(String, nullable=False)
    worker_code = Column(String, nullable=False)  
    assignedLocation = Column(String, nullable=False)
    role = Column(String, nullable=False)
    dateAdded = Column(Date, nullable=False)
    status = Column(String, nullable=False)
    registered = Column(Boolean, default=False)  

    user_id = Column(Integer, ForeignKey("users.id"))
    user = relationship("User", back_populates="workers")


