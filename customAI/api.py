# api.py  –  pure FastAPI, no LLM logic
import asyncio, uuid
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from backend import (
    AgentState, store, build_graph, eda_node, execute_node, undo_node, export_node, upload_node
)

app = FastAPI(title="Data-Agent API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions: dict[str, asyncio.Queue[str]] = {}

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    state = AgentState()
    state = upload_node(state, content)  # returns state with raw/work ids
    session_id = str(uuid.uuid4())
    sessions[session_id] = asyncio.Queue()
    return {
        "sessionId": session_id,
        "shape": state.user_message.split("×")[0].split()[-1] + " rows",
        "preview": store.get_df(state.work_id).head(5).to_dict(orient="records"),
        "stats": state.user_message
    }

@app.websocket("/api/ws/{session_id}")
async def ws(ws: WebSocket, session_id: str):
    await ws.accept()
    if session_id not in sessions:
        await ws.send_json({"type": "error", "text": "bad session"})
        await ws.close(); return
    inbox = sessions[session_id]
    graph = build_graph()
    state = AgentState()
    state = upload_node(state, b"")  # dummy – already uploaded
    state = eda_node(state)
    await ws.send_json({"type": "stats", "text": state.user_message})

    try:
        while True:
            data = await ws.receive_json()
            if data.get("type") != "chat": continue
            await inbox.put(data["text"])
            state.next_node = "human_input"
            while state.next_node != "human_input":
                if state.next_node == "execute": state = execute_node(state)
                elif state.next_node == "eda":
                    state = eda_node(state)
                    await ws.send_json({"type": "stats", "text": state.user_message})
                elif state.next_node == "undo":
                    state = undo_node(state)
                    await ws.send_json({"type": "stats", "text": state.user_message})
                elif state.next_node == "export":
                    state = export_node(state)
                    await ws.send_json({"type": "chat", "text": state.user_message})
            await ws.send_json({"type": "chat", "text": state.user_message})
    except WebSocketDisconnect:
        sessions.pop(session_id, None)
