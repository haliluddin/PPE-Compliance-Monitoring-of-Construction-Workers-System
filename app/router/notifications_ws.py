from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.router.auth import verify_token
import asyncio
import json
from datetime import datetime, timezone, timedelta

router = APIRouter()

connected_clients = {}

@router.websocket("/ws/notifications")
async def websocket_notifications(websocket: WebSocket, token: str = None):
    await websocket.accept()
    try:
        user_data = verify_token(token)
        user_id = int(user_data["sub"])
    except Exception:
        await websocket.close()
        return

    if user_id not in connected_clients:
        connected_clients[user_id] = []
    connected_clients[user_id].append(websocket)

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        try:
            connected_clients[user_id].remove(websocket)
        except Exception:
            pass
        if not connected_clients.get(user_id):
            connected_clients.pop(user_id, None)
    finally:
        try:
            if websocket in connected_clients.get(user_id, []):
                connected_clients[user_id].remove(websocket)
        except Exception:
            pass

async def broadcast_notification(user_id: int, data: dict):
    if user_id not in connected_clients:
        return
    message = json.dumps(data, default=str)
    coros = []
    for ws in list(connected_clients.get(user_id, [])):
        try:
            coros.append(ws.send_text(message))
        except Exception:
            try:
                connected_clients[user_id].remove(ws)
            except Exception:
                pass
    if coros:
        await asyncio.gather(*coros, return_exceptions=True)
