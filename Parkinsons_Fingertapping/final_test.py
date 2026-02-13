import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from scipy.signal import savgol_filter
from tensorflow.keras import layers, Model
import os

# --- 1. MODEL ARCHITECTURE (Required for Lambda loading) ---
def build_model_for_inference(input_shape=(600, 2)):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(filters=64, kernel_size=5, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)
    lstm_out = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    attention = layers.Dense(1, activation='tanh')(lstm_out)
    attention = layers.Flatten()(attention)
    attention = layers.Activation('softmax')(attention)
    attention = layers.RepeatVector(128)(attention) 
    attention = layers.Permute([2, 1])(attention)
    sent_representation = layers.Multiply()([lstm_out, attention])
    sent_representation = layers.Lambda(lambda xin: tf.reduce_sum(xin, axis=1), name='attention_sum')(sent_representation)
    x = layers.Dense(64, activation='relu')(sent_representation)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return Model(inputs=inputs, outputs=outputs)

# --- 2. INITIALIZE AI ---
print("Initializing AI System...")
model = build_model_for_inference()
model.load_weights('parkinsons_final_model.keras')

# --- 3. PREPROCESSING LOGIC ---
def preprocess_sequence(raw_coords):
    coords = np.array(raw_coords)
    # Calculate Euclidean Distance
    dist = np.sqrt(np.sum((coords[:, 0, :] - coords[:, 1, :])**2, axis=1))
    
    # 1. Clean data (Remove NaNs if any)
    dist = dist[~np.isnan(dist)]
    
    # 2. ADAPTIVE AMPLITUDE STRETCHING (The "Severe Case" Fix)
    # This stretches the tiny 0.14 peaks to look like 1.0 peaks
    p_min = np.min(dist)
    p_max = np.max(dist)
    if (p_max - p_min) > 0.01: # Ensure there is actually movement
        normalized = (dist - p_min) / (p_max - p_min)
    else:
        normalized = dist
    
    # 3. Double Smoothing (Cleans the 30fps/60fps jitter)
    smoothed = savgol_filter(normalized, window_length=11, polyorder=3)
    
    # 4. Feature Extraction: Velocity & Acceleration
    # Velocity (slowness) and Acceleration (tremor energy)
    vel = np.diff(smoothed, prepend=smoothed[0])
    acc = np.diff(vel, prepend=vel[0])
    
    # Use Velocity and Acceleration as our 2 channels
    # These are MUCH more sensitive to severe PD than just raw distance
    combined = np.stack([vel, acc], axis=-1)
    
    # 5. Fix length for BiLSTM (600 frames)
    if len(combined) > 600:
        combined = combined[:600]
    else:
        combined = np.pad(combined, ((0, 600 - len(combined)), (0, 0)), 'constant')
        
    return combined.reshape(1, 600, 2)

# --- 4. VIDEO FILE CONFIGURATION ---
# Put your video file name here (ensure it is in the same folder)
video_filename = "tapping_test.mp4" 

if not os.path.exists(video_filename):
    print(f"Error: Could not find {video_filename} in the current folder.")
    exit()

print(f"Processing video: {video_filename}")
cap = cv2.VideoCapture(video_filename)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

recorded_coords = []
frame_count = 0

print("Analyzing hand movement frame by frame...")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break # End of video

    # Convert BGR to RGB for MediaPipe
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            # Extract landmarks for Thumb (4) and Index (8)
            t = hand_lms.landmark[4]
            i = hand_lms.landmark[8]
            recorded_coords.append([[t.x, t.y, t.z], [i.x, i.y, i.z]])
            
            # Optional: Draw on frame to see it processing (slows down execution)
            # mp_draw.draw_landmarks(image, hand_lms, mp_hands.HAND_CONNECTIONS)
    
    frame_count += 1
    if frame_count % 50 == 0:
        print(f"Processed {frame_count} frames...")

cap.release()

# --- 5. FINAL REMARKS ---
if len(recorded_coords) > 50:
    print("\n--- FINAL DIAGNOSIS ---")
    input_data = preprocess_sequence(recorded_coords)
    prob = model.predict(input_data)[0][0]
    
    # Threshold for Diagnostic Accuracy
    threshold = 0.63
    
    if prob > threshold:
        print(f"REMARK: Parkinsonian Rhythm Detected ({prob*100:.1f}%)")
        print("Note: AI detected irregularities in tap speed and distance.")
    else:
        print(f"REMARK: Healthy Movement Pattern ({prob*100:.1f}%)")
        print("Note: Rhythmic pattern is consistent and within normal range.")
else:
    print("Error: AI could not find a hand in the video clearly enough to analyze.")

cv2.destroyAllWindows()