import time
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp

# ==========================================
# HELPER FUNCTIONS & ALGORITHMS
# ==========================================

def calculate_ear(landmarks, eye_indices, width, height):
    """Computes Eye Aspect Ratio (EAR) using Euclidean distances."""
    pts = np.array([[landmarks[i].x * width, landmarks[i].y * height] for i in eye_indices])
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    h = np.linalg.norm(pts[0] - pts[3])
    return (v1 + v2) / (2.0 * h) if h > 0 else 0.0

def calculate_mar(landmarks, mouth_indices, width, height):
    """Computes Mouth Aspect Ratio (MAR) to monitor yawning."""
    pts = np.array([[landmarks[i].x * width, landmarks[i].y * height] for i in mouth_indices])
    vertical = np.linalg.norm(pts[2] - pts[3])
    horizontal = np.linalg.norm(pts[0] - pts[1])
    return vertical / horizontal if horizontal > 0 else 0.0

def estimate_head_pose(landmarks, width, height):
    """Estimates head orientation (Forward, Left, Right)."""
    nose_x = landmarks[1].x * width
    left_jaw_x = landmarks[234].x * width
    right_jaw_x = landmarks[454].x * width
    
    face_width = right_jaw_x - left_jaw_x
    if face_width <= 0:
        return "FORWARD"
        
    relative_pos = (nose_x - left_jaw_x) / face_width
    if relative_pos < 0.35:
        return "TURNED LEFT"
    elif relative_pos > 0.65:
        return "TURNED RIGHT"
    else:
        return "FORWARD"

# ==========================================
# STREAMLIT USER INTERFACE & MAIN APP
# ==========================================

st.set_page_config(page_title="WakeUP - Driver Fatigue Monitor", layout="wide")

st.title("🚗 WakeUP: Real-Time Driver Fatigue & Distraction Monitor")
st.markdown("Monitor drowsiness, yawning, and head posture in real-time using computer vision.")

# Sidebar Settings
st.sidebar.header("⚙️ Settings")
ear_threshold = st.sidebar.slider("EAR Threshold", 0.15, 0.30, 0.21, 0.01)
mar_threshold = st.sidebar.slider("MAR Threshold", 0.40, 0.85, 0.65, 0.05)
closure_limit = st.sidebar.slider("Eye Closure Limit (s)", 1.0, 3.0, 1.5, 0.1)

run_system = st.checkbox("▶️ Start Live Webcam Monitoring", value=False)

col1, col2 = st.columns([2, 1])
with col1:
    video_placeholder = st.empty()
with col2:
    st.subheader("📊 Telemetry Log")
    status_metric = st.empty()
    ear_metric = st.empty()
    mar_metric = st.empty()
    pose_metric = st.empty()

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

eye_closed_start_time = None
avg_ear = 0.0
mar = 0.0
head_status = "FORWARD"

if run_system:
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened() and run_system:
        ret, frame = cap.read()
        if not ret:
            st.error("Webcam access failed.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        status_text = "NORMAL - Alert"
        status_color = (0, 255, 0)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            LEFT_EYE = [33, 160, 158, 133, 153, 144]
            RIGHT_EYE = [362, 385, 387, 263, 373, 380]
            MOUTH = [78, 308, 13, 14, 87, 317]
            
            left_ear = calculate_ear(landmarks, LEFT_EYE, w, h)
            right_ear = calculate_ear(landmarks, RIGHT_EYE, w, h)
            avg_ear = (left_ear + right_ear) / 2.0
            mar = calculate_mar(landmarks, MOUTH, w, h)
            head_status = estimate_head_pose(landmarks, w, h)
            
            # Drowsiness check
            if avg_ear < ear_threshold:
                if eye_closed_start_time is None:
                    eye_closed_start_time = time.time()
                elif time.time() - eye_closed_start_time >= closure_limit:
                    status_text = "🚨 DROWSINESS ALERT!"
                    status_color = (0, 0, 255)
            else:
                eye_closed_start_time = None

            # Yawn check
            if mar > mar_threshold:
                status_text = "⚠️ YAWN DETECTED!"
                status_color = (0, 165, 255)

            # Head pose check
            if head_status != "FORWARD":
                status_text = f"⚠️ DISTRACTION ({head_status})!"
                status_color = (0, 165, 255)

            cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
        status_metric.metric("State", status_text)
        ear_metric.metric("EAR", f"{avg_ear:.3f}")
        mar_metric.metric("MAR", f"{mar:.3f}")
        pose_metric.metric("Head Pose", head_status)
        
    cap.release()
else:
    video_placeholder.info("Tick 'Start Live Webcam Monitoring' to begin.")
