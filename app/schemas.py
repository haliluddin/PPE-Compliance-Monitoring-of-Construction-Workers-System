# app/schemas.py
from pydantic import BaseModel
from datetime import date

class UserCreate(BaseModel):
    username: str
    password: str
    role: str

class UserOut(BaseModel):
    id: int
    username: str
    role: str

    class Config:
        from_attributes = True


class WorkerCreate(BaseModel):
    fullName: str
    worker_code: str
    assignedLocation: str
    role: str
    dateAdded: date
    status: str
    registered: bool = False  

class WorkerResponse(BaseModel):
    id: int
    fullName: str
    worker_code: str
    assignedLocation: str
    role: str
    dateAdded: date
    status: str
    registered: bool  # include in response
    user_id: int

    class Config:
        orm_mode = True
