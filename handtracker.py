import cv2
import mediapipe as mp
import csv
import time

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera not opened")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

csv_file = open("tracked_data.csv", "w", newline="")
csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "time",
    "hand_id",
    "middle_x",
    "middle_y",
    "middle_z",
    "chin_x",
    "chin_y",
    "chin_z"
])

start_time = time.time()
last_save = 0.0

# Store last valid mouth
last_chin_x = last_chin_y = last_chin_z = ""

with mp_hands.Hands(
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands, mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False
) as face_mesh:

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hand_results = hands.process(rgb)
        face_results = face_mesh.process(rgb)

        now = time.time() - start_time

        # ---------------- Mouth (stored as chin) ----------------
        if face_results.multi_face_landmarks:

            face = face_results.multi_face_landmarks[0].landmark

            lm13 = face[13]
            lm14 = face[14]

            last_chin_x = (lm13.x + lm14.x) / 2
            last_chin_y = (lm13.y + lm14.y) / 2
            last_chin_z = (lm13.z + lm14.z) / 2

            cx, cy = int(last_chin_x * w), int(last_chin_y * h)
            cv2.circle(frame, (cx, cy), 6, (0,255,255), -1)

        # ---------------- Hands ----------------
        if hand_results.multi_hand_landmarks:

            for hid, hand in enumerate(hand_results.multi_hand_landmarks):

                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                lm11 = hand.landmark[11]
                lm12 = hand.landmark[12]

                mx = (lm11.x + lm12.x) / 2
                my = (lm11.y + lm12.y) / 2
                mz = (lm11.z + lm12.z) / 2

                px, py = int(mx * w), int(my * h)
                cv2.circle(frame, (px, py), 8, (255,0,255), -1)

                if now - last_save >= 0.5:
                    csv_writer.writerow([
                        round(now,3),
                        hid,
                        mx,my,mz,
                        last_chin_x,
                        last_chin_y,
                        last_chin_z
                    ])

            if now - last_save >= 0.5:
                last_save = now

        cv2.imshow("Hand + Mouth Tracker", frame)

        if cv2.waitKey(1) == 27:
            break

csv_file.close()
cap.release()
cv2.destroyAllWindows()
