import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

# ตั้งค่า Mediapipe สำหรับการตรวจจับท่าทาง
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
pose = mp_pose.Pose()

# ตั้งค่ากล้อง
cap = cv2.VideoCapture(0)

# ตัวแปรสำหรับนับคะแนน
score = 0
sets = 0
pose_detected = False  # สถานะการตรวจจับท่าถูกต้อง
max_sets = 3  # จำนวนเซ็ตที่ต้องการให้ทำ
exercise_complete = False  # สถานะเมื่อทำครบทุกท่าแล้ว

def calculate_distance(a, b):
    """คำนวณระยะห่างระหว่างจุดสองจุด (a, b)"""
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)

while cap.isOpened() and not exercise_complete:
    ret, frame = cap.read()
    if not ret:
        break
    
    # แปลงภาพเป็น RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # ใช้ Mediapipe ตรวจจับท่าทาง
    result_pose = pose.process(frame_rgb)
    
    # ตรวจจับท่าทางจาก Mediapipe Pose
    if result_pose.pose_landmarks:
        landmarks = result_pose.pose_landmarks.landmark
        
        # วาดเส้นโครงร่างทั้งร่างกาย
        mp_drawing.draw_landmarks(
            frame, 
            result_pose.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
        
        # ดึงตำแหน่งของข้อต่อที่เกี่ยวข้อง
        shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
        elbow_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
        wrist_right = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]
        
        # คำนวณระยะห่างจากไหล่ถึงข้อมือ (เมื่อแขนอยู่ด้านหน้า)
        wrist_to_shoulder_distance = calculate_distance(wrist_right, shoulder_right)
        elbow_to_shoulder_distance = calculate_distance(elbow_right, shoulder_right)
        
        # ตรวจจับการขยับแขนจากข้างลำตัวมาด้านหน้า (วัดการเคลื่อนไหวของข้อมือเทียบกับไหล่)
        if wrist_right[0] < shoulder_right[0] and wrist_to_shoulder_distance < elbow_to_shoulder_distance:
            cv2.putText(frame, "Arm moved to the front", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if not pose_detected:
                score += 1  # เพิ่มคะแนนเมื่อท่าถูกต้อง
                pose_detected = True  # ตั้งสถานะเป็นตรวจจับท่าถูกต้องแล้ว
        else:
            pose_detected = False  # รีเซ็ตสถานะเมื่อท่าไม่ถูกต้อง
        
        # แสดงจำนวนเซ็ตที่ทำได้
        if score >= 12:
            sets += 1
            score = 0  # รีเซ็ตคะแนนสำหรับเซ็ตถัดไป
            cv2.putText(frame, f"Set Completed! Sets: {sets}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # แสดงจำนวนเซ็ตที่ทำได้ตลอดเวลา
        cv2.putText(frame, f"Sets: {sets}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # จบการออกกำลังกายเมื่อครบเซ็ตที่ต้องการ
        if sets >= max_sets:
            exercise_complete = True

    # แสดงภาพที่ได้พร้อมเส้นโครงร่างทั้งร่างกาย
    cv2.imshow("Image", frame)

    # กด 'q' เพื่อออกจากโปรแกรม
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปล่อยกล้องและปิดหน้าต่างเมื่อจบโปรแกรม
cap.release()
cv2.destroyAllWindows()

