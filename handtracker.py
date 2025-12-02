import cv2
import mediapipe as mp
import csv
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# --- Init webcam ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# --- CSV File ---
csv_file = open("middle_finger_center.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["time", "hand", "center_x", "center_y", "center_z"])

# --- Timer ---
start_time = time.time()
last_save_time = 0.0   # seconds

# Middle Finger ID mapping
MCP = 9   # Middle finger MCP joint
TIP = 12  # Middle finger tip

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape

        # BGR → RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        # True elapsed time
        current_time = time.time() - start_time

        # ---- SAVE DATA EVERY 0.5 SECONDS ----
        if current_time - last_save_time >= 0.5:
            if results.multi_hand_landmarks:
                for hand_id, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    lm = hand_landmarks.landmark

                    # Compute center of middle finger (MCP + TIP) / 2
                    cx = (lm[MCP].x + lm[TIP].x) / 2
                    cy = (lm[MCP].y + lm[TIP].y) / 2
                    cz = (lm[MCP].z + lm[TIP].z) / 2

                    # Save in CSV (normalized coordinates)
                    csv_writer.writerow([
                        round(current_time, 2),
                        hand_id,
                        cx, cy, cz
                    ])

            last_save_time = current_time

        # ---- DRAW TRACKER ON SCREEN ----
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                lm = hand_landmarks.landmark

                # Calculate pixel position for center
                cx = int(((lm[MCP].x + lm[TIP].x) / 2) * w)
                cy = int(((lm[MCP].y + lm[TIP].y) / 2) * h)

                # Draw circle
                cv2.circle(frame, (cx, cy), 12, (0, 255, 0), -1)

                # Draw hand skeleton
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

        # Show output
        cv2.imshow("Middle Finger Center Tracker", frame)

        # Exit with ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

csv_file.close()
cap.release()
cv2.destroyAllWindows()
