from flask import Flask, Response, render_template_string
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)
CORS(app)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# ตัวแปรสำหรับการนับเซ็ตและแต้ม
count = 0
mark = 0 
score = 0

sets = 0
kit = 0
series = 0

max_set1= 3
max_set2= 3
max_set3 = 3

pose_detected = False
exercies_complete = False
hand_open = False

pose_complete = False
mark_complete = False
series_complete = False

cap = cv2.VideoCapture(1)

def calculate_distance(a, b):
    a, b = np.array(a), np.array(b)
    return np.linalg.norm(a - b)

def check_finger_position(finger, elbow):
    return finger[1] > elbow[1]

def generate_frames():
    global count, mark, score, sets, kit, series
    global pose_detected, exercies_complete, hand_open
    global pose_complete, mark_complete, series_complete

    while cap.isOpened():
        if exercies_complete:
            # ส่งข้อความ "completed" เมื่อ exercise_complete เป็น True
            yield (b'--frame\r\n'
                   b'Content-Type: text/plain\r\n\r\n' + b'completed' + b'\r\n')
            break

        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with mp_pose.Pose() as pose, mp_hands.Hands() as hands:
            result_pose = pose.process(frame_rgb)
            results = hands.process(frame_rgb)

            if result_pose.pose_landmarks:
                landmarks = result_pose.pose_landmarks.landmark
                mp_drawing.draw_landmarks(
                    frame,
                    result_pose.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )

                # ท่าที่ 1: Arm moved to the front
                if not pose_complete:
                    shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
                    elbow_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
                    wrist_right = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]

                    wrist_to_shoulder_distance = calculate_distance(wrist_right, shoulder_right)
                    elbow_to_shoulder_distance = calculate_distance(elbow_right, shoulder_right)

                    if wrist_right[0] < shoulder_right[0] and wrist_to_shoulder_distance < elbow_to_shoulder_distance:
                        cv2.putText(frame, f"Arm moved to the front", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        if not pose_detected:
                            score += 1
                            pose_detected = True
                    else:
                        pose_detected = False

                    if score >= 6:
                        sets += 1
                        score  = 0
                        cv2.putText(frame, f"Set frist Completed!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                    cv2.putText(frame, f"Set frist exerciesr : {sets}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                    if sets >= max_set1:
                        pose_complete = True

                # ท่าที่ 2: Finger bent down
                if pose_complete and not mark_complete:
                    elbow_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
                    finger_right = [landmarks[mp_pose.PoseLandmark.RIGHT_INDEX].x, landmarks[mp_pose.PoseLandmark.RIGHT_INDEX].y]

                    if check_finger_position(finger_right, elbow_right):
                        cv2.putText(frame, f"Finger bent down", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (300, 0, 0), 3, cv2.LINE_AA)
                        if not pose_detected:
                            count += 1
                            pose_detected = True
                    else:
                        pose_detected = False

                    if count >= 6:
                        mark += 1
                        count = 0
                        cv2.putText(frame, f"Set  second Completed! ", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3, cv2.LINE_AA)

                    cv2.putText(frame, f"Set second exerciesr: {mark}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    if mark >= max_set2:
                        mark_complete = True

                # ท่าที่ 3: มือกางออก
                if mark_complete and not series_complete and results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                        thumb_tip_y = hand_landmarks.landmark[4].y
                        index_tip_y = hand_landmarks.landmark[8].y
                        middle_tip_y = hand_landmarks.landmark[12].y
                        ring_tip_y = hand_landmarks.landmark[16].y
                        pinky_tip_y = hand_landmarks.landmark[20].y

                        if (thumb_tip_y < hand_landmarks.landmark[3].y and
                                index_tip_y < hand_landmarks.landmark[6].y and
                                middle_tip_y < hand_landmarks.landmark[10].y and
                                ring_tip_y < hand_landmarks.landmark[14].y and
                                pinky_tip_y < hand_landmarks.landmark[18].y):

                            if not hand_open:
                                kit += 1
                                cv2.putText(frame, f"Hand up", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (300, 0, 0), 3, cv2.LINE_AA)
                                hand_open = True

                                if kit >= 6:
                                    cv2.putText(frame, f"Set third Completed! ", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3, cv2.LINE_AA)
                                    series += 1
                                    kit = 0
                                    if series >= max_set3:
                                        print("Complete 3 series!")
                                        series_complete = True
                                        exercies_complete = True
                                        break
                        else:
                            hand_open = False
                    cv2.putText(frame, f"Set thrid exerciesr: {series}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/styled_video_feed')
def styled_video_feed():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Styled Video Feed</title>
        <style>
            body, html {
                height: 100%;
                margin: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                background-color: #000;
            }
            #videoFeed {
                width: 100vw;
                height: 100vh;
                object-fit: cover;
            }
            #backButton {
                position: absolute;
                top: 20px;
                left: 20px;
                padding: 10px 20px;
                font-size: 18px;
                background-color: #fff;
                border: none;
                cursor: pointer;
                border-radius: 5px;
            }
            #cornerVideo {
                position: absolute;
                bottom: 10px;
                right: 10px; 
                width: 600px; 
                height: auto;
                border: 2px solid #fff;
                border-radius: 5px;
            }
        </style>
    </head>
    <body>
        <button id="backButton" onclick="goBack()">กลับไปหน้า Home</button>
        <img src="/video_feed" id="videoFeed" alt="Video Feed">
        
        <!-- เพิ่มแท็กวิดีโอ -->
        <video id="cornerVideo" autoplay loop muted>
            <source src="/static/Exercise.mp4" type="video/mp4">
            Your browser does not support the video tag.
        </video>
    
        <script>
            function goBack() {
                localStorage.setItem('showHomePage', 'true');
                window.location.href = 'http://localhost:3000/';
            }

            const videoFeed = document.getElementById('videoFeed');
            videoFeed.onload = function() {
                // เช็คถ้าได้รับข้อความ "completed"
                fetch('/video_feed').then(response => {
                    if (response.ok) {
                        response.text().then(text => {
                            if (text.includes('completed')) {
                                goBack(); // ย้อนกลับไปหน้า Home
                            }
                        });
                    }
                });
            };
        </script>
    </body>
    </html>
    """
    return render_template_string(html_content)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
