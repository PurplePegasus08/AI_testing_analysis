import cv2
import torch
import numpy as np
import os
import pickle
import pandas as pd
from datetime import datetime
from facenet_pytorch import MTCNN, InceptionResnetV1

# ---------------- CONFIG ---------------- #
KNOWN_FACES_DIR = r"C:\Users\Admin\Documents\Face_reco\known_faces"
ENCODING_FILE = r"C:\Users\Admin\Documents\Face_reco\known_faces_encodings.pkl"
ATTENDANCE_FILE = "attendance.csv"

MATCH_THRESHOLD = 0.6   
CONF_THRESHOLD = 0.90   
# ADJUSTED: Setting this to 50 so low-light live faces pass. 
# Photos/Screens usually stay extremely low (below 30).
LIVENESS_THRESHOLD = 50 
# Required consecutive "real" frames before marking attendance
STABILITY_FRAMES = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

already_marked_today = set()
# Tracking stability for each person
stability_tracker = {}

# ---------------- MODELS ---------------- #
mtcnn = MTCNN(image_size=160, margin=20, device=DEVICE, post_process=True)
model = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)

def check_liveness(face_img):
    if face_img.size == 0: return 0
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def get_embedding(img_rgb, box):
    try:
        x1, y1, x2, y2 = map(int, box)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), max(0, x2), max(0, y2)
        face = img_rgb[y1:y2, x1:x2]
        if face.size == 0: return None
        face_tensor = mtcnn(face)
        if face_tensor is None: return None
        face_tensor = face_tensor.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = model(face_tensor)
        return emb.cpu().numpy()[0]
    except:
        return None

def mark_attendance(name):
    global already_marked_today
    if name == "Unknown" or name in already_marked_today:
        return
    now = datetime.now()
    date_str, time_str = now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")
    file_exists = os.path.isfile(ATTENDANCE_FILE)
    df = pd.DataFrame([{"Name": name, "Date": date_str, "Time": time_str}])
    df.to_csv(ATTENDANCE_FILE, mode='a', index=False, header=not file_exists)
    already_marked_today.add(name)
    print(f"✅ ATTENDANCE SUCCESS: {name}")

def load_and_sync():
    known_faces, known_names = [], []
    if os.path.exists(ENCODING_FILE):
        try:
            with open(ENCODING_FILE, "rb") as f:
                known_faces, known_names = pickle.load(f)
            if len(known_faces) > 0 and len(known_faces[0]) != 512:
                known_faces, known_names = [], []
        except:
            known_faces, known_names = [], []

    existing_names = {str(n).strip() for n in known_names}
    new_added = False
    
    for file in os.listdir(KNOWN_FACES_DIR):
        name = os.path.splitext(file)[0].strip()
        if file.lower().endswith(("jpg", "png", "jpeg")) and name not in existing_names:
            img = cv2.imread(os.path.join(KNOWN_FACES_DIR, file))
            if img is None: continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            boxes, _ = mtcnn.detect(img_rgb)
            if boxes is not None:
                emb = get_embedding(img_rgb, boxes[0])
                if emb is not None:
                    known_faces.append(emb)
                    known_names.append(name)
                    new_added = True

    if new_added:
        with open(ENCODING_FILE, "wb") as f:
            pickle.dump((known_faces, known_names), f)
    return np.array(known_faces), known_names

def run_live():
    faces_array, known_names = load_and_sync()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    print("🚀 System Active. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Checking for 'q' first to ensure it's responsive
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, probs = mtcnn.detect(img_rgb)

        if boxes is not None:
            for box, prob in zip(boxes, probs):
                if prob < CONF_THRESHOLD: continue
                
                x1, y1, x2, y2 = map(int, box)
                face_crop = frame[max(0, y1):y2, max(0, x1):x2]
                liveness_score = check_liveness(face_crop)
                
                # Recognition (Always run this so the name shows)
                emb = get_embedding(img_rgb, box)
                name = "Unknown"
                if emb is not None and len(faces_array) > 0:
                    norm_emb = np.linalg.norm(emb)
                    norm_faces = np.linalg.norm(faces_array, axis=1)
                    dot_prod = np.dot(faces_array, emb)
                    dists = 1 - (dot_prod / (norm_emb * norm_faces))
                    best_idx = np.argmin(dists)
                    if dists[best_idx] < MATCH_THRESHOLD:
                        name = known_names[best_idx]

                # --- Liveness & Attendance Logic ---
                if name != "Unknown":
                    # Only increment stability if score is above the (now lower) threshold
                    if liveness_score > LIVENESS_THRESHOLD:
                        stability_tracker[name] = stability_tracker.get(name, 0) + 1
                    else:
                        stability_tracker[name] = max(0, stability_tracker.get(name, 0) - 1)

                    if stability_tracker[name] >= STABILITY_FRAMES:
                        mark_attendance(name)
                        status = "VERIFIED"
                        color = (0, 255, 0) # Green
                    else:
                        status = f"SCANNING ({stability_tracker[name]}/{STABILITY_FRAMES})"
                        color = (0, 165, 255) # Orange
                else:
                    status = "Unknown"
                    color = (0, 0, 255) # Red

                # Visual Output
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{name} | {status} | Liveness: {int(liveness_score)}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Secure Attendance", frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live()
