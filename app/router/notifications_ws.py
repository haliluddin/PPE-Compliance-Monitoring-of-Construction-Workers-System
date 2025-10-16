from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from typing import Dict, List
from app.router.auth import get_current_user
from app.models import User

router = APIRouter(prefix="/ws", tags=["WebSocket Notifications"])

# Store connected clients per user
connected_clients: Dict[int, List[WebSocket]] = {}

@router.websocket("/notifications")
async def websocket_endpoint(websocket: WebSocket, token: str):
    """
    Each supervisor connects using their JWT token:
    ws://localhost:8000/ws/notifications?token=<JWT>
    """
    from jose import jwt
    from app.config import settings  # your secret key config

    try:
        # decode token to get user_id
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id = int(payload.get("user_id"))
    except Exception:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    if user_id not in connected_clients:
        connected_clients[user_id] = []
    connected_clients[user_id].append(websocket)

    try:
        while True:
            await websocket.receive_text()  # keep connection alive
    except WebSocketDisconnect:
        connected_clients[user_id].remove(websocket)
        if not connected_clients[user_id]:
            del connected_clients[user_id]
