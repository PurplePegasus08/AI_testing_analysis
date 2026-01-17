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

def live_webcam():
    cap = cv2.VideoCapture(0)
    stability = {} # Corrected spelling
    
    print("Starting webcam... Press 'q' to quit.") # Corrected spelling
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = process_one_frame(rgb_frame, stability)
        
        name = result.get("name") or "Unknown"
        status = result.get("status")
        progress = result.get("progress")
        
        # FIXED: Status check is case-sensitive (VERIFIED)
        color = (0, 255, 0) if status == "VERIFIED" else (0, 165, 255)
        
        # FIXED: Injected the actual 'progress' variable into the string
        display_text = f"{name} - {status} ({progress})"
        cv2.putText(frame, display_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        cv2.imshow("Face Attendance Live", frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()
        
    

def check_liveness(face_img):
    if face_img.size == 0: return 0
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def get_embedding(img_rgb, box):
    try:
        x1, y1, x2, y2 = map(int, box) # bounding box
        face = img_rgb[max(0, y1):y2, max(0, x1):x2] #slicing 
        if face.size == 0: return None
        
        face = cv2.resize(face, (160, 160))
        face = torch.tensor(face).permute(2, 0, 1).float() / 255.0
        face = face.unsqueeze(0).to(DEVICE)
        
        with torch.no_grad(): #no gradients calculations for saving resources
            emb = model(face)
        return emb.cpu().numpy()[0]
    except Exception:
        return None

def mark_attendance_db(name):
    if name == "Unknown": return 
    now = datetime.now()
    
    
    prev = LAST_MARK.get(name)
    if prev and (now-prev).total_seconds() < 600:
        return
    
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM users WHERE name = %s", (name,))
            user_data = cur.fetchone()
            if user_data:
                user_id = user_data[0]
                cur.execute(
                    "INSERT INTO attendance (user_id, name, timestamp) VALUES (%s, %s, %s)",
                    (user_id, name,now)
                )
                conn.commit()
                LAST_MARK[name] = now
    finally:
        db_pool.putconn(conn)
        
def get_attendance_report(day_back=1):
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            query = """
            SELECT name, 
                   MIN(timestamp) as start_time, 
                   MAX(timestamp) as end_time,
                   (MAX(timestamp) - MIN(timestamp)) as duration
            FROM attendance
            WHERE timestamp::date = current_date - %s
            GROUP BY name, timestamp::date;
            """
            # FIXED: Added comma to create a proper tuple
            cur.execute(query, (day_back,)) 
            return cur.fetchall()
    finally:
        # FIXED: Always return connection to pool
        db_pool.putconn(conn)
                

# ---------------- DATABASE CRUD (Using Pool) ---------------- #

def register_new_face(name,file_bytes):
    
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr,cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(img_rgb)
    if boxes is not None:
        emb = get_embedding(img_rgb,boxes[0])
        if emb is not None:
            register_user_in_db(name,emb)
            return True
    return False

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

#For image input checking   
if __name__ == "__main__":
    try:
        # --- OPTION A: Test on a Local Image File (Current) ---
        test_image_path = r"C:\Users\Admin\Documents\Face_reco\input_img\nu1.jpg" 
        
        if os.path.exists(test_image_path):
            print(f"📸 Testing on image: {test_image_path}")
            frame = cv2.imread(test_image_path)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Use a dummy stability dict for the single image
            # We loop 5 times to simulate the 'STABILITY_FRAMES' required for VERIFIED status
            temp_stability = {}
            for i in range(STABILITY_FRAMES):
                result = process_one_frame(frame_rgb, temp_stability)
                print(f"Frame {i+1} result: {result['status']} ({result['progress']})")
            
            print(f"\n🔍 Final Match: {result['name']}")
        else:
            print("❌ Image not found. Check the path.")

        # --- OPTION B: Live Webcam (Commented out until you have one) ---
        # live_webcam() 

    finally:
        if 'db_pool' in globals():
            db_pool.closeall()
            print("🔌 Database pool closed.")
            
            
#For webcam input checking            
            
if __name__ == "__main__":
    try:
        # --- [ACTIVE] OPTION B: Live Webcam ---
        live_webcam() 

    finally:
        if 'db_pool' in globals():
            db_pool.closeall()
            print("🔌 Database pool closed.")
