from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse,HTMLResponse
import io
import cv2
import numpy as np
import face_engine as engine
import pandas as pd

app = FastAPI()

# Enable CORS for the HTML frontend
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/",response_class=HTMLResponse)
async def get_ui():
    with open("index.html",encoding=" utf8") as f:
        return f.read()
# ---------------- CORE REAL-TIME ENDPOINTS ---------------- #

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handles real-time stream from index.html"""
    await websocket.accept()
    connection_stability = {} 
    try:
        while True:
            data = await websocket.receive_bytes()
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Processes frame using the face_engine logic
                result = engine.process_one_frame(frame_rgb, connection_stability)
                await websocket.send_json(result)
    except WebSocketDisconnect:
        print("Client disconnected from WebSocket")
        
        
#verification
@app.post("/verify")
async def verify(file: UploadFile = File(...)):
    """Testing endpoint for single image uploads - UPDATED to update DB"""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Create a temporary counter and pre-fill it to (STABILITY_FRAMES - 1)
    # This ensures that process_one_frame triggers the database update on the first try
    temp_stability = {} 
    
    # First, we identify who is in the photo
    # We run it once to get the name
    result = engine.process_one_frame(frame_rgb, {})
    detected_name = result.get("name")

    if detected_name and detected_name != "Unknown":
        # Manually force the counter to the limit for this specific request
        force_counter = {detected_name: engine.STABILITY_FRAMES}
        # Run it again with the forced counter to trigger mark_attendance_db
        result = engine.process_one_frame(frame_rgb, force_counter)
    
    return result

# ---------------- USER MANAGEMENT ENDPOINTS ---------------- #

@app.post("/register")
async def register(name: str = Form(...), file: UploadFile = File(...)):
    """Registers a new face embedding into the database"""
    success = engine.register_new_face(name, await file.read())
    if not success:
        raise HTTPException(status_code=400, detail="No face detected in image")
    return {"message": f"User {name} registered successfully"}

@app.delete("/user/{name}")
def delete_user(name: str):
    """Removes a user from the database"""
    engine.delete_user(name)
    return {"message": f"User {name} deleted"}

# ---------------- REPORTING ENDPOINTS ---------------- #

@app.get("/attendance/yesterday")
def yesterday():
    """Returns attendance logs for the previous day"""
    data = engine.get_attendance_report(1)
    # Formats the raw database tuples into a readable JSON list
    return [{"name": r[0], "start": r[1], "end": r[2], "duration": str(r[3])} for r in data]

@app.get("/attendance/export")
def export_csv():
    """Generates and streams a CSV of the last month's attendance"""
    conn = engine.db_pool.getconn()
    try:
        # Queries attendance logs for the last 30 days
        df = pd.read_sql("SELECT * FROM attendance WHERE timestamp > current_date - interval '1 month'", conn)
        
        stream = io.StringIO()
        df.to_csv(stream, index=False)
        return StreamingResponse(
            iter([stream.getvalue()]), 
            media_type="text/csv", 
            headers={"Content-Disposition": "attachment; filename=attendance_export.csv"}
        )
    finally:
        engine.db_pool.putconn(conn)

@app.get("/health")
def health():
    """System health check including GPU availability"""
    return {"status": "running", "gpu": engine.torch.cuda.is_available()}
