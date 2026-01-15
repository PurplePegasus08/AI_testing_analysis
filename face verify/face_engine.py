import cv2
import numpy as np
import pandas as pd
import torch
import os
from datetime import datetime
from facenet_pytorch import MTCNN, InceptionResnetV1
from pgvector.psycopg2 import register_vector
from psycopg2 import pool

# ---------------- CONFIG ---------------- #
KNOWN_FACES_DIR = r"C:\Users\Admin\Documents\Face_reco\known_faces"
ATTENDANCE_FILE = "attendance.csv"

MATCH_THRESHOLD    = 0.4  # Cosine distance (lower is better)
CONF_THRESHOLD     = 0.90
LIVENESS_THRESHOLD = 50          
STABILITY_FRAMES   = 5             
COOL_DOWN          = 60            
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- MODELS ---------------- #
mtcnn = MTCNN(image_size=160, margin=20, device=DEVICE, post_process=True)
model = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)

# ---------------- DATABASE POOL ---------------- #
# Define the pool ONCE at the top level
db_pool = pool.ThreadedConnectionPool(
    1, 20, 
    host="localhost",
    database="for_vector",
    user="postgres",
    password="purple"
)

# ---------------- CACHE ---------------- #
SEEN_TODAY = set()
LAST_MARK  = {}

def check_liveness(face_img):
    if face_img.size == 0: return 0
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def get_embedding(img_rgb, box):
    try:
        x1, y1, x2, y2 = map(int, box) # bounding box
        face = img_rgb[max(0, y1):y2, max(0, x1):x2] #slicing 
        if face.size == 0: return None
        
        face_tensor = mtcnn(face)
        if face_tensor is None: return None
        
        face_tensor = face_tensor.unsqueeze(0).to(DEVICE)
        with torch.no_grad(): #no gradients calculations for saving resources
            emb = model(face_tensor)
        return emb.cpu().numpy()[0]
    except Exception:
        return None

def mark_attendance_db(name):
    if name == "Unknown": return
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    key = (name, date)
    
    if key in SEEN_TODAY: return
    prev = LAST_MARK.get(name)
    if prev and (now - prev).total_seconds() < COOL_DOWN: return

    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            # 1. Get the user_id first
            cur.execute("SELECT id FROM users WHERE name = %s", (name,))
            user_data = cur.fetchone()
            
            if user_data:
                user_id = user_data[0]
                cur.execute(
                    "INSERT INTO attendance (user_id, name) VALUES (%s, %s)",
                    (user_id, name)
                )
                conn.commit()
                
                SEEN_TODAY.add(key)
                LAST_MARK[name] = now
                print(f"SQL: Attendence marked for {name}")
    finally:
        db_pool.putconn(conn)

# ---------------- DATABASE CRUD (Using Pool) ---------------- #

def register_user_in_db(name, embedding):
    conn = db_pool.getconn()
    try:
        register_vector(conn)
        with conn.cursor() as cur:
            # Change your query to this:
           # Change your query to this:
            query = """
            INSERT INTO users (name, embedding) VALUES (%s, %s::vector)
            ON CONFLICT (name) DO UPDATE SET embedding = EXCLUDED.embedding;
            """
            cur.execute(query, (name, embedding.tolist()))
            conn.commit()
            print(f"👤 {name} registered/updated in DB.")
    finally:
        db_pool.putconn(conn)
        


def search_in_db(embedding):
    conn = db_pool.getconn()
    try:
        register_vector(conn)
        with conn.cursor() as cur:
            # Change your query to this:
            query = """
            SELECT name, embedding <=> %s::vector AS distance 
            FROM users 
            ORDER BY distance ASC 
            LIMIT 1;
            """
            cur.execute(query, (embedding.tolist(),))
            result = cur.fetchone()
            
            if result:
                name, distance = result
                if distance < MATCH_THRESHOLD:
                    return name
        return "Unknown"
    finally:
        db_pool.putconn(conn)
        
def migrate_faces_to_db():
    print(" Starting migrating from the folder to DB...")
    if not os.path.exists(KNOWN_FACES_DIR):
        print(f" Folder not found: {KNOWN_FACES_DIR}")
        return
    
    for file in os.listdir(KNOWN_FACES_DIR):
        if file.lower().endswith(('jpg','png','jpeg')):
            name = os.path.splitext(file)[0].strip()
            img_path = os.path.join(KNOWN_FACES_DIR,file)
            
            img = cv2.imread(img_path)
            if img is None:continue
            
            img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            boxes, _ = mtcnn.detect(img_rgb)
            
            if boxes is not None:
                emb = get_embedding(img_rgb, boxes[0])
                if emb is not None:
                    register_user_in_db(name,emb)
                    
                else: print(f"Could not generate embedding for {name}")  
                
            else: print(f"No face detectedin {file}")
    
    print("Migration complete!")        
        
                    
        

def delete_user(name):
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM users WHERE name = %s", (name,))
            conn.commit()
            print(f"🗑️ {name} deleted from DB.")
    finally:
        db_pool.putconn(conn)

# ---------------- FRAME PROCESSING ---------------- #

def process_one_frame(frame_rgb, stability_counter):
    boxes, probs = mtcnn.detect(frame_rgb)
    if boxes is None:
        return {"name": None, "status": "NO_FACE", "liveness": 0}

    best_i = np.argmax(probs)
    box, prob = boxes[best_i], probs[best_i]
    
    if prob < CONF_THRESHOLD:
        return {"name": None, "status": "LOW_CONFIDENCE", "liveness": 0}

    emb = get_embedding(frame_rgb, box)
    name = search_in_db(emb) if emb is not None else "Unknown"
    
    x1, y1, x2, y2 = map(int, box)
    face_crop = frame_rgb[max(0, y1):y2, max(0, x1):x2]
    liveness = check_liveness(face_crop)

    status = "SCANNING"
    if name != "Unknown" and liveness > LIVENESS_THRESHOLD:
        stability_counter[name] = stability_counter.get(name, 0) + 1
        if stability_counter[name] >= STABILITY_FRAMES:
            mark_attendance_db(name)
            status = "VERIFIED"
    else:
        # Reset specific user counter if verification fails
        if name in stability_counter:
            stability_counter[name] = 0

    return {
        "name": name,
        "status": status,
        "liveness": round(liveness, 2),
        "progress": f"{stability_counter.get(name, 0)}/{STABILITY_FRAMES}"
    }
    
if __name__ == "__main__":
    try:
        # Step 1: Migration (Only run once, then comment it out)
        # migrate_faces_to_db()

        # Step 2: Test on a Local Image File
        test_image_path = r"C:\Users\Admin\Documents\Face_reco\input_img\nu1.jpg" # Update this path
        
        if os.path.exists(test_image_path):
            print(f"📸 Testing on image: {test_image_path}")
            frame = cv2.imread(test_image_path)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # We need a dictionary to track stability in this session
            session_stability = {} 
            
            # Process the frame
            # Since we only have one image, we'll simulate "STABILITY_FRAMES" 
            # by calling it in a loop or just checking the result
            result = process_one_frame(frame_rgb, session_stability)
            
            print(f"🔍 Result: {result}")
            
            # If you want to force mark attendance for a single image test:
            if result['name'] and result['name'] != "Unknown":
                mark_attendance_db(result['name'])
        else:
            print("❌ Test image not found. Please provide a valid path.")

    finally:
        if 'db_pool' in globals():
            db_pool.closeall()
            print("🔌 Database pool closed.")
