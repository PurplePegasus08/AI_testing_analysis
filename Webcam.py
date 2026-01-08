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

MATCH_THRESHOLD = 0.6   # Cosine distance (Lower is stricter)
CONF_THRESHOLD = 0.90   # MTCNN confidence
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

already_marked_today = set()

# ---------------- MODELS ---------------- #
print(f"🚀 Running on: {DEVICE}")
mtcnn = MTCNN(image_size=160, margin=20, device=DEVICE, post_process=True)
model = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)

# ---------------- FUNCTIONS ---------------- #
def get_embedding(img_rgb, box):
    """Extracts embedding from a specific box in the image."""
    try:
        x1, y1, x2, y2 = map(int, box)
        # Ensure coordinates are within image bounds
        x1, y1 = max(0, x1), max(0, y1)
        face = img_rgb[y1:y2, x1:x2]
        
        # MTCNN can resize and normalize the face for the model
        face_tensor = mtcnn(face)
        if face_tensor is None: return None
        
        face_tensor = face_tensor.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = model(face_tensor)
        return emb.cpu().numpy()[0]
    except Exception as e:
        return None

def mark_attendance(name):
    global already_marked_today
    if name == "Unknown" or name in already_marked_today:
        return

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    file_exists = os.path.isfile(ATTENDANCE_FILE)
    df = pd.DataFrame([{"Name": name, "Date": date_str, "Time": time_str}])
    df.to_csv(ATTENDANCE_FILE, mode='a', index=False, header=not file_exists)
    
    already_marked_today.add(name)
    print(f"📝 ATTENDANCE MARKED: {name} at {time_str}")

# ---------------- SYNC DATABASE ---------------- #
def load_and_sync():
    if os.path.exists(ENCODING_FILE):
        with open(ENCODING_FILE, "rb") as f:
            known_faces, known_names = pickle.load(f)
        print(f"✅ Loaded {len(known_names)} faces from {ENCODING_FILE}")
    else:
        known_faces, known_names = [], []

    existing_names = {str(n).strip() for n in known_names}
    new_added = False
    
    print("🔍 Scanning folder for new images...")
    for file in os.listdir(KNOWN_FACES_DIR):
        name = os.path.splitext(file)[0].strip()
        if file.lower().endswith(("jpg", "png", "jpeg")) and name not in existing_names:
            print(f"⚙️  Encoding: {name}...", end=" ", flush=True)
            img = cv2.imread(os.path.join(KNOWN_FACES_DIR, file))
            if img is None: continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            boxes, _ = mtcnn.detect(img_rgb)
            
            if boxes is not None:
                emb = get_embedding(img_rgb, boxes[0])
                if emb is not None:
                    known_faces.append(emb)
                    known_names.append(name)
                    existing_names.add(name)
                    new_added = True
                    print("✅")
                else: print("❌ (Embedding error)")
            else: print("❌ (No face detected)")

    if new_added:
        with open(ENCODING_FILE, "wb") as f:
            pickle.dump((known_faces, known_names), f)
        print("💾 Database Updated.")
    else:
        print("🆗 No new faces to add.")
        
    return known_faces, known_names

# ---------------- MAIN LOOP ---------------- #
def run_live():
    known_faces, known_names = load_and_sync()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    print("🚀 Camera Live. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detection
        boxes, probs = mtcnn.detect(img_rgb)

        if boxes is not None:
            for box, prob in zip(boxes, probs):
                if prob < CONF_THRESHOLD: continue
                
                # Recognition
                emb = get_embedding(img_rgb, box)
                name = "Unknown"
                
                if emb is not None and len(known_faces) > 0:
                    # Cosine distance logic
                    dists = [1 - np.dot(emb, k)/(np.linalg.norm(emb)*np.linalg.norm(k)) for k in known_faces]
                    best_idx = np.argmin(dists)
                    
                    if dists[best_idx] < MATCH_THRESHOLD:
                        name = known_names[best_idx]
                        mark_attendance(name)

                # Draw
                x1, y1, x2, y2 = map(int, box)
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{name} ({prob:.2f})", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("FaceNet Attendance System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live()