import mediapipe as mp
import cv2

class HandTracker:
    def __init__(self, max_hands=1):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None

    def find_hands(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def get_landmark_positions(self, img):
        h, w, _ = img.shape
        landmarks = {}
        if self.results and self.results.multi_hand_landmarks:
            handLms = self.results.multi_hand_landmarks[0]
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks[id] = (cx, cy)
        return landmarks
