# app/router/auth.py
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import User
import bcrypt
from jose import JWTError, jwt
from datetime import datetime, timedelta

router = APIRouter()
SECRET_KEY = "supersecretkey"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 2
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

class LoginSchema(BaseModel):
    email: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class RegisterSchema(BaseModel):
    name: str
    email: EmailStr
    password: str

def _normalize_password_bytes(password) -> bytes:
    if password is None:
        password = ""
    if not isinstance(password, (str, bytes)):
        password = str(password)
    if isinstance(password, str):
        pwd_bytes = password.encode("utf-8", errors="ignore")
    else:
        pwd_bytes = password
    return pwd_bytes[:72]

def hash_password(password) -> str:
    pwd = _normalize_password_bytes(password)
    hashed = bcrypt.hashpw(pwd, bcrypt.gensalt())
    return hashed.decode("utf-8", errors="ignore")

def verify_password(plain_password, hashed_password) -> bool:
    try:
        pwd = _normalize_password_bytes(plain_password)
        return bcrypt.checkpw(pwd, hashed_password.encode("utf-8"))
    except Exception:
        return False

def authenticate_user(db: Session, email: str, password: str):
    result = db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user

@router.post("/login")
def login(data: LoginSchema, db: Session = Depends(get_db)):
    user = authenticate_user(db, data.email, data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    expire = datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    token_data = {"sub": str(user.id), "email": user.email, "exp": expire}
    token = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {"id": user.id, "name": user.name, "email": user.email}
    }

@router.post("/register")
def register(data: RegisterSchema, db: Session = Depends(get_db), request: Request = None):
    try:
        result = db.execute(select(User).where(User.email == data.email))
        existing_user = result.scalar_one_or_none()
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")
        hashed = hash_password(data.password)
        new_user = User(name=data.name, email=data.email, hashed_password=hashed, is_supervisor=True)
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        return {"message": "User registered successfully", "user": {"id": new_user.id, "name": new_user.name, "email": new_user.email}}
    except Exception as e:
        import traceback, sys
        traceback.print_exc(file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Register failed: {str(e)}")

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    result = db.execute(select(User).where(User.id == int(user_id)))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

