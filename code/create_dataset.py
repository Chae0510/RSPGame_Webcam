import cv2
import mediapipe as mp
import numpy as np
import time, os
import pandas as pd

# actions는 딱히 쓸 필요 없음
actions = ['scissors']
seq_length = 30
secs_for_action = 30

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

created_time = int(time.time())
# os.makedirs('dataset', exist_ok=True)

while cap.isOpened():
    for idx, action in enumerate(actions):
        data = []

        ret, img = cap.read()
        img = cv2.flip(img, 1)
        cv2.putText(img, f'Waiting for collecting {action.upper()} action...', org=(10, 30), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        cv2.imshow('img', img)
        cv2.waitKey(3000)

        start_time = time.time()

        while time.time() - start_time < secs_for_action:
            ret, img = cap.read()
            img = cv2.flip(img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if result.multi_hand_landmarks:
                for res in result.multi_hand_landmarks:
                    joint = np.zeros((21, 3))
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.z]

                    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :]  # Parent joint
                    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :]  # Child joint
                    v = v2 - v1  # [20, 3]
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]  # Normalize v
                    angle = np.arccos(np.clip(np.einsum('nt,nt->n', v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:]), -1.0, 1.0))  # [15,]
                    angle = np.degrees(angle)  # Convert radian to degree

                    angle_label = np.append(angle, 9)  # 여기서 원하는 index를 지정해서 저장
                    data.append(angle_label)

                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            cv2.imshow('img', img)
            if cv2.waitKey(1) == ord('q'):
                break

        data = np.array(data)
        print(action, data.shape)
        df = pd.DataFrame(data)
        df.to_csv(os.path.join('/path/to/dataset', f'raw_{action}_{created_time}.csv'), index=False, header=False)

    break
