# face_engine.py
import cv2
import numpy as np
import pandas as pd
import torch
import os
import pickle
from datetime import datetime
from facenet_pytorch import MTCNN, InceptionResnetV1

# ---------------- CONFIG ---------------- #
KNOWN_FACES_DIR = r"C:\Users\Admin\Documents\Face_reco\known_faces"
ENCODING_FILE   = r"C:\Users\Admin\Documents\Face_reco\encodings_pickle.pkl"
ATTENDANCE_FILE = "attendance.csv"

MATCH_THRESHOLD  = 0.6
CONF_THRESHOLD   = 0.90
LIVENESS_THRESHOLD = 50          
STABILITY_FRAMES = 5             
COOL_DOWN        = 60            
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- MODELS ---------------- #
# Keep these global to load them only once
mtcnn = MTCNN(image_size=160, margin=20, device=DEVICE, post_process=True)
model = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)

# ---------------- CACHE ---------------- #
SEEN_TODAY = set()
LAST_MARK  = {}

def check_liveness(face_img):
    if face_img.size == 0: return 0
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def get_embedding(img_rgb, box):
    try:
        x1, y1, x2, y2 = map(int, box)
        face = img_rgb[max(0, y1):y2, max(0, x1):x2]
        if face.size == 0: return None
        
        face_tensor = mtcnn(face)
        if face_tensor is None: return None
        
        face_tensor = face_tensor.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = model(face_tensor)
        return emb.cpu().numpy()[0]
    except Exception:
        return None

def mark_attendance(name):
    if name == "Unknown": return
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    key = (name, date)
    
    # Check cooldown and daily limit
    if key in SEEN_TODAY: return
    prev = LAST_MARK.get(name)
    if prev and (now - prev).total_seconds() < COOL_DOWN: return

    file_exists = os.path.isfile(ATTENDANCE_FILE)
    df = pd.DataFrame([{"Name": name, "Date": date, "Time": now.strftime("%H:%M:%S")}])
    df.to_csv(ATTENDANCE_FILE, mode='a', index=False, header=not file_exists)
    
    SEEN_TODAY.add(key)
    LAST_MARK[name] = now
    print(f"✅ Attendance marked for: {name}")

def load_and_sync():
    known_faces, known_names = [], []
    if os.path.exists(ENCODING_FILE):
        try:
            with open(ENCODING_FILE, "rb") as f:
                known_faces, known_names = pickle.load(f)
        except:
            pass

    existing_names = {str(n).strip() for n in known_names}
    new_added = False
    
    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)

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
            pickle.dump((np.array(known_faces), known_names), f)
            
    return np.array(known_faces), known_names

def process_one_frame(frame_rgb, known_faces, known_names, stability_counter):
    boxes, probs = mtcnn.detect(frame_rgb)
    if boxes is None:
        return {"name": None, "status": "NO_FACE", "liveness": 0}

    best_i = np.argmax(probs)
    box, prob = boxes[best_i], probs[best_i]
    
    if prob < CONF_THRESHOLD:
        return {"name": None, "status": "LOW_CONFIDENCE", "liveness": 0}

    # Recognition logic
    emb = get_embedding(frame_rgb, box)
    name = "Unknown"
    if emb is not None and len(known_faces) > 0:
        # Vectorized Cosine Similarity
        dists = np.linalg.norm(known_faces - emb, axis=1)
        best_idx = np.argmin(dists)
        if dists[best_idx] < MATCH_THRESHOLD:
            name = known_names[best_idx]

    # Liveness check
    x1, y1, x2, y2 = map(int, box)
    face_crop = frame_rgb[max(0, y1):y2, max(0, x1):x2]
    liveness = check_liveness(face_crop)

    # State update
    if name != "Unknown" and liveness > LIVENESS_THRESHOLD:
        stability_counter[name] += 1
    else:
        # Reset if person leaves or liveness fails
        stability_counter.clear() 

    status = "SCANNING"
    if stability_counter[name] >= STABILITY_FRAMES:
        mark_attendance(name)
        status = "VERIFIED"

    return {
        "name": name,
        "status": status,
        "liveness": liveness,
        "progress": f"{stability_counter[name]}/{STABILITY_FRAMES}"
    }
