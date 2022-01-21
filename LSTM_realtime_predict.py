import cv2
import mediapipe as mp
import preprocessing as pre
from tensorflow import keras
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

#감사 데이터 부족, 위: 손가락 인식 모누 안됌 존경: 받침 손가락 제대로 인식불가
label = ['THANKYOU', 'POLICE', 'HEAD', 'HELLO', 'DOWN', 'UP', 'HOME', 'RESPECT', 'FRIEND', 'DAD', 'STATIC']
# Actions that we try to detect
actions = np.array(label)

twoD_list = []
face = [10, 234, 152, 454]
body = range(11, 23)
sequence = []
sentence = []
new_model = keras.models.load_model('sign_language_action.h5')

colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245), (245, 117, 16), (117, 245, 16), (16, 117, 245),
          (245, 117, 16), (117, 245, 16), (16, 117, 245), (245, 117, 16), (117, 245, 16)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
    return output_frame
flag = 0
# For webcam input:
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        success, image = cap.read()
        image = cv2.flip(image, 1)
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            break

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        total_list = pre.extract_landmarks(results)
        sequence.append(total_list)
        sequence = sequence[-50:] #10에서 하나씩 업데이트가 되며, 10개가 쌓인 이후에는 현재 인덱스포함 이전 10개의 데이터를 가져옴
        #print('sequence #: ', len(sequence))

        if len(sequence) == 50:
            twoD_array = pre.set_relative_axial(sequence)
            twoD_array_interpolated = pre.preprocessing_inter(twoD_array, 0)
            twoD_array_discrete = pre.preprocessing_inter(twoD_array, 1)

            X_data1 = pre.argumentation(twoD_array, sampling_frame=10) # (20, 10, 174)? oo
            X_data2 = pre.argumentation(twoD_array_interpolated, sampling_frame=10)
            X_data3 = pre.argumentation(twoD_array_discrete, sampling_frame=10)

            res1 = new_model.predict(X_data1)
            res2 = new_model.predict(X_data2)
            res3 = new_model.predict(X_data3)                           # (20, 10) = 20개의 video, 각 video에서의 10개의 action확률
            #print(X_data.shape, res.shape)                             # 그리고 20개의 video는 같은 행동임
            #print("predict is ", actions[np.argmax(res[4])])
            resul = [np.argmax(res1[i]) for i in range(20)]+[np.argmax(res2[i]) for i in range(20)]+[np.argmax(res3[i]) for i in range(20)]
            print(np.bincount(resul))
            res = np.bincount(resul)/60

            #res = new_model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)], ' ', res[np.argmax(res)]*100, " %")



            #---초기화---
            sequence = []
            white1 = np.full((image.shape[0], image.shape[1], 3), 255, dtype=np.uint8)
            alpha = 0.7
            image = cv2.addWeighted(image, alpha, white1, (1 - alpha), 0)
            # print to imshow landmark
            flag = 1

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Draw landmark annotation on the image. holistic
        image = pre.print_landmarksNindex(image, results.face_landmarks, face)
        image = pre.print_landmarksNindex(image, results.pose_landmarks, body)

        # Draw the hand annotations on the image.
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Flip the image horizontally for a selfie-view display.
        if flag == 1:
            image = prob_viz(res, actions, image, colors)
        image = cv2.resize(image, (1500, 1000))
        cv2.imshow('MediaPipe Hands', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
