import cv2
import os
import pickle
import face_recognition
import pandas as pd
import numpy as np
from datetime import datetime

# ---------------- CONFIG ---------------- #
KNOWN_FACES_DIR = r"C:\Users\Admin\Documents\Face_reco\known_faces"
# FIXED: Using your exact requested filename
ENCODING_FILE = r"C:\Users\Admin\Documents\Face_reco\known_faces_encodings.pkl"
ATTENDANCE_FILE = "attendance.csv" 

FRAME_RESIZE_FACTOR = 0.25  
PROCESS_EVERY_N_FRAMES = 5  
MATCH_THRESHOLD = 0.5

already_marked_today = set()

# ---------------- FACE DATA (Sync Logic) ---------------- #
def load_and_sync_faces():
    # 1. Load your existing pkl file
    if os.path.exists(ENCODING_FILE):
        with open(ENCODING_FILE, "rb") as f:
            known_faces, known_names = pickle.load(f)
        print(f"\n✅ DATABASE LOADED: Found {len(known_names)} faces in your .pkl file.")
    else:
        known_faces, known_names = [], []
        print(f"\n⚠️ FILE NOT FOUND: Creating {ENCODING_FILE} from scratch.")

    existing_names = {str(n).strip() for n in known_names}
    new_added = False
    
    print("-----------------------------------------------")
    print(f"🔍 SCANNING FOLDER: {KNOWN_FACES_DIR}")
    
    files = [f for f in os.listdir(KNOWN_FACES_DIR) if f.lower().endswith(("jpg", "png", "jpeg"))]
    
    for file in files:
        name_from_file = os.path.splitext(file)[0].strip()
        
        # If name is already in your .pkl file, skip it
        if name_from_file in existing_names:
            print(f"🆗 ALREADY IN PKL: {name_from_file}")
            continue
            
        # If it's a new image, encode it
        print(f"⚙️  ENCODING NEW FACE: {name_from_file}...", end=" ", flush=True)
        path = os.path.join(KNOWN_FACES_DIR, file)
        
        try:
            img = face_recognition.load_image_file(path)
            enc = face_recognition.face_encodings(img)
            
            if enc:
                known_faces.append(enc[0])
                known_names.append(name_from_file)
                existing_names.add(name_from_file)
                new_added = True
                print("✅ SUCCESS")
            else:
                print("❌ FAILED (No face detected in photo)")
        except Exception as e:
            print(f"❌ ERROR (Check if file is corrupted)")

    print("-----------------------------------------------\n")

    # Save only if we actually added something new
    if new_added:
        with open(ENCODING_FILE, "wb") as f:
            pickle.dump((known_faces, known_names), f)
        print(f"💾 SAVED: {ENCODING_FILE} updated with new faces.\n")
    else:
        print("✨ NO CHANGES: Everything in folder matches your .pkl file.\n")

    return known_faces, known_names

# ---------------- ATTENDANCE ---------------- #
def mark_attendance(name):
    global already_marked_today
    if name in already_marked_today:
        return

    now = datetime.now()
    dt_str = now.strftime("%Y-%m-%d")
    tm_str = now.strftime("%H:%M:%S")

    file_exists = os.path.isfile(ATTENDANCE_FILE)
    df = pd.DataFrame([{"Name": name, "Date": dt_str, "Time": tm_str}])
    df.to_csv(ATTENDANCE_FILE, mode='a', index=False, header=not file_exists)
    
    already_marked_today.add(name)
    print(f"📝 ATTENDANCE MARKED: {name}")

# ---------------- LIVE SYSTEM ---------------- #
def run_system(known_faces, known_names):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ CAMERA ERROR: Cannot open webcam.")
        return

    frame_count = 0
    face_locations, face_names = [], []

    while True:
        ret, frame = cap.read()
        if not ret: break

        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            small = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE_FACTOR, fy=FRAME_RESIZE_FACTOR)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            # Using HOG for speed
            face_locations = face_recognition.face_locations(rgb_small, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

            face_names = []
            for enc in face_encodings:
                name = "Unknown"
                if known_faces:
                    dist = face_recognition.face_distance(known_faces, enc)
                    idx = np.argmin(dist)
                    if dist[idx] < MATCH_THRESHOLD:
                        name = known_names[idx]
                        mark_attendance(name)
                face_names.append(name)

        frame_count += 1

        # Draw rectangles
        for (t, r, b, l), name in zip(face_locations, face_names):
            t, r, b, l = [int(v / FRAME_RESIZE_FACTOR) for v in (t, r, b, l)]
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (l, t), (r, b), color, 2)
            cv2.putText(frame, name, (l, t-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Live Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    f, n = load_and_sync_faces()
    run_system(f, n)
