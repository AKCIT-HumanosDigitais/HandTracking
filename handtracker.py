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

# Chin landmark IDs
CHIN_IDS = [152, 148, 377]

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

        chin_points = []

        # ---------------- Chin (3 point average) ----------------
        if face_results.multi_face_landmarks:

            face_lms = face_results.multi_face_landmarks[0].landmark

            for cid in CHIN_IDS:
                lm = face_lms[cid]
                chin_points.append((lm.x, lm.y, lm.z))

                # draw each raw chin point
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (0,255,255), -1)

        # Compute average chin
        if chin_points:
            chin_x = sum(p[0] for p in chin_points) / len(chin_points)
            chin_y = sum(p[1] for p in chin_points) / len(chin_points)
            chin_z = sum(p[2] for p in chin_points) / len(chin_points)

            fx, fy = int(chin_x * w), int(chin_y * h)
            cv2.circle(frame, (fx, fy), 10, (0,0,255), -1)

        else:
            chin_x = chin_y = chin_z = ""

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
                        chin_x,chin_y,chin_z
                    ])

        if now - last_save >= 0.5:
            last_save = now

        cv2.imshow("Hand + Chin Tracker", frame)

        if cv2.waitKey(1) == 27:
            break

csv_file.close()
cap.release()
cv2.destroyAllWindows()
