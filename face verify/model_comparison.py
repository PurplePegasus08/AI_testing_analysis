import cv2
import torch
import numpy as np
import os
from facenet_pytorch import MTCNN
from retinaface import RetinaFace

# Set device to GPU if available, else use CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
mtcnn = MTCNN(keep_all=True, device=DEVICE)  # MTCNN for face detection and landmarks
# RetinaFace is used directly without instantiation as a callable

# Function to process image with MTCNN
def process_with_mtcnn(image):
    # MTCNN returns bounding boxes and landmarks (no probs)
    boxes, landmarks = mtcnn.detect(image)
    return boxes, landmarks

# Function to process image with RetinaFace
def process_with_retinaface(image):
    # RetinaFace's detect_faces() method returns bounding boxes and landmarks
    faces = RetinaFace.detect_faces(image)
    boxes = []
    landmarks = []
    for key in faces:
        identity = faces[key]
        boxes.append(identity['facial_area'])
        landmarks.append(identity['landmarks'])
    return boxes, landmarks

# Function to draw bounding boxes and landmarks on image
def draw_results(image, boxes, landmarks=None, model_name="Model"):
    image_copy = image.copy()
    for box in boxes:
        cv2.rectangle(image_copy, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
    
    if landmarks:
        for landmark in landmarks:
            # For MTCNN, landmarks are a list of tuples (x, y)
            if isinstance(landmark, list):  # Check if landmarks are in a list format
                for point in landmark:
                    cv2.circle(image_copy, tuple(map(int, point)), 2, (0, 0, 255), -1)
            else:
                # Handle the case when landmark is a float or not in list format
                print("Unexpected landmark format:", landmark)
                
    cv2.putText(image_copy, f'{model_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    return image_copy

# Load images from folder
image_folder = r"C:\Users\Admin\Documents\Face_reco\known_faces"
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('jpg', 'jpeg', 'png'))]

# Folder to save the output images
output_folder = r"C:\Users\Admin\Documents\Face_reco\output_comparison"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process and compare each image
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path)

    # Convert image to RGB for the models
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # MTCNN Processing
    mtcnn_boxes, mtcnn_landmarks = process_with_mtcnn(image_rgb)
    mtcnn_result = draw_results(image, mtcnn_boxes, mtcnn_landmarks, "MTCNN")

    # RetinaFace Processing
    retinaface_boxes, retinaface_landmarks = process_with_retinaface(image_rgb)
    retinaface_result = draw_results(image, retinaface_boxes, retinaface_landmarks, "RetinaFace")

    # Create a side-by-side comparison of MTCNN and RetinaFace results
    comparison_image = np.hstack((mtcnn_result, retinaface_result))  # Concatenate images horizontally

    # Save the comparison image
    output_path = os.path.join(output_folder, f"comparison_{image_file}")
    comparison_image_bgr = cv2.cvtColor(comparison_image, cv2.COLOR_RGB2BGR)  # Convert to BGR for saving
    cv2.imwrite(output_path, comparison_image_bgr)  # Save image

    # Print confirmation for each saved image
    print(f"Saved comparison image for {image_file} at {output_path}")
