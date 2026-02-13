import cv2
import mediapipe as mp
import csv
import os
import time

# --- CONFIGURATION ---
VIDEO_FOLDER = 'raw_videos'      # Path to your videos
OUTPUT_FOLDER = 'processed_data' # Where to save results
CHECKPOINT_FILE = 'logs/progress.txt'

# --- INITIALIZE MEDIAPIPE (Optimized for Tremors) ---
mp_hands = mp.solutions.hands
# static_image_mode=False enables the internal Kalman Filter for temporal stabilization
hands = mp_hands.Hands(
    static_image_mode=False,        # CRITICAL: Enables temporal video tracking
    max_num_hands=1, 
    model_complexity=1,             # Standard complexity for best balance
    min_detection_confidence=0.8,   # Higher threshold to avoid false detections
    min_tracking_confidence=0.8     # CHANGE: Forces AI to 'stick' to shaky fingers
)

def extract_landmarks(video_path):
    cap = cv2.VideoCapture(video_path)
    video_name = os.path.basename(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_data = []

    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Convert BGR to RGB (MediaPipe requirement)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Temporal processing happens here automatically because static_image_mode=False
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # --- NOBLE ADDITION: Landmark 0 (Wrist) for Hand Size Normalization ---
                wrist = hand_landmarks.landmark[0]
                # Landmark 4: Thumb Tip | Landmark 8: Index Tip
                thumb = hand_landmarks.landmark[4]
                index = hand_landmarks.landmark[8]
                
                # Store (Timestamp, Wrist, Thumb, Index X, Y, Z)
                frame_data.append([
                    video_name,
                    frame_count / fps, # Timestamp in seconds
                    wrist.x, wrist.y, wrist.z,   # NEW: Added wrist for normalization
                    thumb.x, thumb.y, thumb.z,
                    index.x, index.y, index.z
                ])
        
        frame_count += 1
    
    cap.release()
    return frame_data

# --- BATCH PROCESSING ---
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Load already processed files
processed_files = []
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, 'r') as f:
        processed_files = f.read().splitlines()

video_files = [f for f in os.listdir(VIDEO_FOLDER) if f.endswith(('.mp4', '.avi', '.mov'))]

# Create CSV Header
csv_path = os.path.join(OUTPUT_FOLDER, 'master_coordinates.csv')
file_exists = os.path.isfile(csv_path)

with open(csv_path, mode='a', newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
        # Added Wrist columns to the header
        writer.writerow([
            'video_id', 'timestamp', 
            'wrist_x', 'wrist_y', 'wrist_z', 
            'thumb_x', 'thumb_y', 'thumb_z', 
            'index_x', 'index_y', 'index_z'
        ])

    for video_name in video_files:
        if video_name in processed_files:
            print(f"Skipping {video_name} (Already processed)")
            continue

        print(f"Processing: {video_name}...")
        start_time = time.time()
        
        full_path = os.path.join(VIDEO_FOLDER, video_name)
        data = extract_landmarks(full_path)
        
        # Save results immediately to master CSV
        writer.writerows(data)
        
        # Update checkpoint
        with open(CHECKPOINT_FILE, 'a') as log:
            log.write(f"{video_name}\n")
            
        print(f"Finished {video_name} in {time.time() - start_time:.2f}s")

print("All videos processed. Your master CSV is ready for feature engineering!")