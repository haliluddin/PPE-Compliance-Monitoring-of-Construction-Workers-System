
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.router.auth import router as auth_router
# from app.router.cameras import router as cameras_router
from app.router.workers import router as workers_router
from app.router.violations import router as violations_router
app = FastAPI()

origins = [
    "http://localhost:5173", 
]

app.add_middleware(
    CORSMiddleware,
   allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
   allow_headers=["*"],
)

app.include_router(auth_router)
# app.include_router(cameras_router)
app.include_router(workers_router)
app.include_router(violations_router)