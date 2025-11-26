import cv2
import mediapipe as mp
import csv

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

# Create CSV file
csv_file = open("hand_wrist_only.csv", "w", newline="")
csv_writer = csv.writer(csv_file)

# Header: frame, hand_id, wrist_x, wrist_y, wrist_z
csv_writer.writerow(["frame", "hand", "wrist_x", "wrist_y", "wrist_z"])

frame_i = 0

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # BGR -> RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe
        results = hands.process(img)

        # RGB -> BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # --- SAVE ONLY EVERY 5 FRAMES ---
        if frame_i % 5 == 0:
            if results.multi_hand_landmarks:
                for hand_id, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    
                    wrist = hand_landmarks.landmark[0]   # wrist landmark (ID = 0)

                    csv_writer.writerow([
                        frame_i,
                        hand_id,
                        wrist.x,
                        wrist.y,
                        wrist.z
                    ])

        # Draw skeleton on screen
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

        cv2.imshow("Hand Tracker", img)
        frame_i += 1

        # ESC to exit
        if cv2.waitKey(1) & 0xFF == 27:
            break

csv_file.close()
cap.release()
cv2.destroyAllWindows()
