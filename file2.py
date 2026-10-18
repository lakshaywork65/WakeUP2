import numpy as np

def calculate_ear(landmarks, eye_indices, width, height):
    """
    Computes Eye Aspect Ratio (EAR) using Euclidean distances.
    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    """
    pts = np.array([[landmarks[i].x * width, landmarks[i].y * height] for i in eye_indices])
    
    # Vertical distances
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    
    # Horizontal distance
    h = np.linalg.norm(pts[0] - pts[3])
    
    if h == 0:
        return 0.0
        
    ear = (v1 + v2) / (2.0 * h)
    return ear

def calculate_mar(landmarks, mouth_indices, width, height):
    """
    Computes Mouth Aspect Ratio (MAR) to monitor vertical open mouth state (yawning).
    """
    pts = np.array([[landmarks[i].x * width, landmarks[i].y * height] for i in mouth_indices])
    
    vertical = np.linalg.norm(pts[2] - pts[3])
    horizontal = np.linalg.norm(pts[0] - pts[1])
    
    if horizontal == 0:
        return 0.0
        
    mar = vertical / horizontal
    return mar

def estimate_head_pose(landmarks, width, height):
    """
    Estimates simplified head orientation (Forward, Turned Left, Turned Right)
    using nose tip position relative to outer jaw contours.
    """
    nose_x = landmarks[1].x * width
    left_jaw_x = landmarks[234].x * width
    right_jaw_x = landmarks[454].x * width
    
    face_width = right_jaw_x - left_jaw_x
    if face_width <= 0:
        return "FORWARD"
        
    relative_nose_pos = (nose_x - left_jaw_x) / face_width
    
    if relative_nose_pos < 0.35:
        return "TURNED LEFT"
    elif relative_nose_pos > 0.65:
        return "TURNED RIGHT"
    else:
        return "FORWARD"
