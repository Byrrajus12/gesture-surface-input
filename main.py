import cv2
from utils.hand_tracker import HandTracker
import numpy as np

cap = cv2.VideoCapture(0)
tracker = HandTracker()
keyboard_corners = []
homography_ready = False

print("🟢 Tap the 4 keyboard corners (TL → TR → BR → BL)")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = tracker.find_hands(frame)
    landmarks = tracker.get_landmark_positions(frame)

    # Show fingertip
    if landmarks and 8 in landmarks:
        x, y = landmarks[8]
        cv2.circle(frame, (x, y), 10, (0, 255, 0), cv2.FILLED)

    # Tap detection: manually press 't' to save current point
    key = cv2.waitKey(1)
    if not homography_ready and key == ord('t') and landmarks and 8 in landmarks:
        keyboard_corners.append(landmarks[8])
        print(f"📍 Corner {len(keyboard_corners)} set at {landmarks[8]}")
        if len(keyboard_corners) == 4:
            homography_ready = True
            print("✅ Calibration complete!")

    # Visualize saved corners
    for pt in keyboard_corners:
        cv2.circle(frame, pt, 8, (255, 0, 0), cv2.FILLED)

    cv2.imshow("Calibration", frame)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
