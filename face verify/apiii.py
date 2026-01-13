# apii.py
import uvicorn
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from face_engine import process_one_frame, load_and_sync
from collections import defaultdict

app = FastAPI(title="Face-Attendance API")

# Load face bank ONCE at server startup
KNOWN_FACES, KNOWN_NAMES = load_and_sync()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # --- PER-CONNECTION STATE ---
    # This prevents User A from affecting User B's counter
    stability_counter = defaultdict(int) 
    
    try:
        while True:
            # Receive image bytes from client
            data = await websocket.receive_bytes()
            
            # Decode image
            np_arr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None: continue
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process frame with the local stability_counter
            info = process_one_frame(
                frame_rgb, 
                KNOWN_FACES, 
                KNOWN_NAMES,
                stability_counter
            )

            # Send result back to client
            await websocket.send_json({
                "name": info["name"],
                "status": info["status"],
                "liveness": int(info["liveness"]),
                "progress": info.get("progress", "0/5")
            })

            # If verified, we can close or keep listening
            if info["status"] == "VERIFIED":
                # Optional: break if you want the session to end after one success
                pass 

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")

# Static files for the frontend
app.mount("/", StaticFiles(directory=".", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)