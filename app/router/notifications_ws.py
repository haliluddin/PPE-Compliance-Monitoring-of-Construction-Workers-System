#app/router/notifications_ws.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from app.router.auth import get_current_user
from app.router.auth import verify_token
from app.models import User
import asyncio
import json

router = APIRouter()

connected_clients = {}  # user_id -> list of WebSocket connections

@router.websocket("/ws/notifications")
async def websocket_notifications(
    websocket: WebSocket,
    token: str = None
):
    # Accept the connection
    await websocket.accept()

    # ✅ Authenticate the user (simplified)
    from app.router.auth import verify_token
    try:
        user_data = verify_token(token)
        user_id = int(user_data["sub"])
    except Exception:
        await websocket.close()
        return

    if user_id not in connected_clients:
        connected_clients[user_id] = []
    connected_clients[user_id].append(websocket)

    print(f"🔌 User {user_id} connected to notifications WebSocket")

    try:
        while True:
            await websocket.receive_text()  # keep alive, can ignore input
    except WebSocketDisconnect:
        connected_clients[user_id].remove(websocket)
        if not connected_clients[user_id]:
            del connected_clients[user_id]
        print(f"❌ User {user_id} disconnected")

async def broadcast_notification(user_id: int, data: dict):
    """Safely broadcast a notification to all connected WebSocket clients."""
    if user_id not in connected_clients:
        print(f"⚠️ No connected clients for user {user_id}")
        return

    message = json.dumps(data)

    for ws in connected_clients[user_id]:
        try:
            await ws.send_text(message)
        except Exception as e:
            print(f"⚠️ WebSocket send error: {e}")