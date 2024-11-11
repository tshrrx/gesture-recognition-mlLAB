import cv2
import numpy as np
import mediapipe as mp
import time
from gesture_detection import is_thumb_up, is_fist, is_peace_sign, is_open_hand, is_waving, is_call_me

kernel = np.ones((5, 5), np.uint8)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

gesture_label = ""
gesture_time = 0

cap = cv2.VideoCapture(0)
ret = True
while ret:
    
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(framergb)

    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * frame.shape[1])
                lmy = int(lm.y * frame.shape[0])
                landmarks.append([lmx, lmy])

            
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

        
        if len(landmarks) > 8:
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            thumb_index_dist = np.linalg.norm(np.array(thumb_tip) - np.array(index_tip))

            current_time = time.time()

            if thumb_index_dist < 30:
                gesture_label = "OK"
                gesture_time = current_time
            elif is_thumb_up(landmarks):
                gesture_label = "Thumbs Up"
                gesture_time = current_time
            elif is_fist(landmarks):
                gesture_label = "Fist"
                gesture_time = current_time
            elif is_peace_sign(landmarks):
                gesture_label = "Peace Sign"
                gesture_time = current_time
            elif is_open_hand(landmarks):
                gesture_label = "Open Hand"
                gesture_time = current_time
            elif is_waving(landmarks):
                gesture_label = "Waving"
                gesture_time = current_time
            elif is_call_me(landmarks):
                gesture_label = "Call Me"
                gesture_time = current_time
            else:
                gesture_label = ""

           
            if current_time - gesture_time > 5:
                gesture_label = ""

           
            cv2.putText(frame, f"Gesture: {gesture_label}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow("Output", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
