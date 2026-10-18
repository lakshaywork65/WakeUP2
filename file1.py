import time
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
from utils import calculate_ear, calculate_mar, estimate_head_pose

# Page Configuration
st.set_page_config(
    page_title="WakeUP - Driver Fatigue & Distraction Monitor",
    page_icon="🚗",
    layout="wide"
)

# Header Section
st.title("🚗 WakeUP: Real-Time Edge Driver Fatigue System")
st.markdown("""
Welcome to **WakeUP**! This AI system tracks facial indicators in real-time to detect **drowsiness**, 
**yawning**, and **distraction** using computer vision algorithms without sending your video stream to the cloud.
""")

# Sidebar Controls & Thresholds
st.sidebar.header("⚙️ Alert Sensitivity Settings")

ear_threshold = st.sidebar.slider(
    "Eye Aspect Ratio (EAR) Threshold", 
    min_value=0.15, max_value=0.30, value=0.21, step=0.01,
    help="Lower values make drowsiness detection less sensitive."
)

mar_threshold = st.sidebar.slider(
    "Mouth Aspect Ratio (MAR) Threshold", 
    min_value=0.40, max_value=0.85, value=0.65, step=0.05,
    help="Threshold for detecting prolonged yawning."
)

closure_time_limit = st.sidebar.slider(
    "Eye Closure Alert Limit (Seconds)", 
    min_value=1.0, max_value=3.0, value=1.5, step=0.1,
    help="Time eyes must remain closed to trigger a continuous alert."
)

run_system = st.checkbox("▶️ Start Live Webcam Monitoring", value=False)

# Layout Setup
col_video, col_telemetry = st.columns([2, 1])

with col_video:
    video_placeholder = st.empty()

with col_telemetry:
    st.subheader("📊 Live Telemetry & Metrics")
    status_metric = st.empty()
    ear_metric = st.empty()
    mar_metric = st.empty()
    pose_metric = st.empty()

# MediaPipe Face Mesh Setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# State Variables
eye_closed_start_time = None

if run_system:
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened() and run_system:
        ret, frame = cap.read()
        if not ret:
            st.error("Unable to access webcam. Please check browser camera permissions.")
            break

        # Flip horizontally for natural mirror view
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Inference
        results = face_mesh.process(rgb_frame)
        
        status_text = "NORMAL - Driver Alert"
        status_color = (0, 255, 0) # Green
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Landmark Index Maps
            LEFT_EYE = [33, 160, 158, 133, 153, 144]
            RIGHT_EYE = [362, 385, 387, 263, 373, 380]
            MOUTH = [78, 308, 13, 14, 87, 317]
            
            # Calculate Safety Metrics
            left_ear = calculate_ear(landmarks, LEFT_EYE, w, h)
            right_ear = calculate_ear(landmarks, RIGHT_EYE, w, h)
            avg_ear = (left_ear + right_ear) / 2.0
            
            mar = calculate_mar(landmarks, MOUTH, w, h)
            head_status = estimate_head_pose(landmarks, w, h)
            
            # Drowsiness Logic (Temporal Window Check)
            if avg_ear < ear_threshold:
                if eye_closed_start_time is None:
                    eye_closed_start_time = time.time()
                else:
                    duration = time.time() - eye_closed_start_time
                    if duration >= closure_time_limit:
                        status_text = "🚨 DROWSINESS ALERT: EYES CLOSED!"
                        status_color = (255, 0, 0) # Red
            else:
                eye_closed_start_time = None

            # Yawn Check
            if mar > mar_threshold:
                status_text = "⚠️ WARNING: Excessive Yawning!"
                status_color = (255, 165, 0) # Orange
                
            # Head Distraction Check
            if head_status != "FORWARD":
                status_text = f"⚠️ DISTRACTION ALERT: Looking {head_status}!"
                status_color = (255, 165, 0)

            # Draw HUD overlays on frame
            cv2.putText(frame, f"WakeUP System Active", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, status_text, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2)

            # Update Telemetry Display
            status_metric.metric("Driver State", status_text)
            ear_metric.metric("Eye Aspect Ratio (EAR)", f"{avg_ear:.3f}")
            mar_metric.metric("Mouth Aspect Ratio (MAR)", f"{mar:.3f}")
            pose_metric.metric("Head Pose", head_status)

        # Render video stream
        frame_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_display, channels="RGB", use_container_width=True)
        
    cap.release()
else:
    video_placeholder.info("👈 Tick 'Start Live Webcam Monitoring' above to begin.")
