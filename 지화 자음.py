import cv2
import os
import math
import mediapipe as mp
from math import atan2, degrees
import matplotlib.pyplot as plt
import numpy
import glob
import os.path

list_x = []#튜플의 리스트로..!!? 있을까?
list_y = []
list_z = []
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def only_angle_between(p1, p2, p3):
  x1, y1 = p1
  x2, y2 = p2
  x3, y3 = p3
  deg1 = (360 + degrees(atan2(x1 - x2, y1 - y2))) % 360
  deg2 = (360 + degrees(atan2(x3 - x2, y3 - y2))) % 360
  return abs(deg2 - deg1) if abs(deg2 - deg1) < 180 else 360 - abs(deg2 - deg1)
def angle_index(a, b, c):
  p1 = (list_x[a], list_y[a])
  p2 = (list_x[b], list_y[b])
  p3 = (list_x[c], list_y[c])
  #print(only_angle_between(p1, p2, p3))
  return only_angle_between(p1, p2, p3)

def point_in_ROI(v_index, range_x, range_y): #v_index is vector 5->8 <=> (5, 8)
  V1 = (list_x[v_index[1]] - list_x[v_index[0]], list_y[v_index[1]] - list_y[v_index[0]])
  factor = math.sqrt(V1[0] ** 2 + V1[1] ** 2)
  V1 = tuple(elem / factor for elem in V1)

  #print(V1[0], V1[1])
  if range_x[0]<=V1[0]<=range_x[1] and range_y[0]<=V1[1]<=range_y[1]:
    return True
  else:
    return False


exe_instruction = 2
exeute_lable = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ', 'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ', 'ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# For local folder input:


image_index = 0 #입력된 이미지 수
count_nothing = 0 #아무것도 인식되지 않는 수
count_ac = 0 # 특정 지화가 인식되는 수
i = 0

# For webcam input:
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            break

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
          for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            image_height, image_width, _ = image.shape

            list_x = []  # 튜플의 리스트로..!!? 있을까?
            list_y = []
            list_z = []
            for landmark_point in range(len(hand_landmarks.landmark)):
              list_x.append(hand_landmarks.landmark[landmark_point].x * image_width)
              list_y.append(hand_landmarks.landmark[landmark_point].y * image_height)
              list_z.append(hand_landmarks.landmark[landmark_point].z * image_height)

            # ㄱ
            if 70 <= angle_index(4, 5, 8) <= 130 and point_in_ROI((5, 8), (-0.5, 0.37), (0.87, 1)) \
                    and (point_in_ROI((5, 4), (-1, -0.75), (-0.75, 0.41)) or point_in_ROI((5, 4), (0.75, 0.99), (-0.75, 0.41))):
              print(image_index, "ㄱ")

            # ㄴ
            elif 80 <= angle_index(4, 5, 8) <= 135 and (point_in_ROI((5, 8), (-1, -0.92), (-0.30, 0.45)) or point_in_ROI((5, 8), (0.92, 1), (-0.36, 0.35))) and \
                    (point_in_ROI((5, 4), (-0.58, 0.58), (-1, -0.83)) or point_in_ROI((5, 4), (-0.62, 0.12), (-1, -0.80)))and \
                 list_z[12] > list_z[4] and list_z[20] > list_z[8]:
              print(image_index, "ㄴ")

            # ㄹ
            elif 5 <= angle_index(8, 5, 12) <= 32 and 10 <= angle_index(12, 9, 16) <= 35 and \
                 (point_in_ROI((5, 8), (-1, -0.93), (-0.4, 0.07)) or point_in_ROI((5, 8), (0.90, 1), (-0.50, 0.08))) and \
                 (point_in_ROI((9, 12), (-1, -0.97), (-0.25, 0.19)) or point_in_ROI((9, 12), (0.96, 1), (-0.27, 0.15))) and \
                 (point_in_ROI((13, 16), (-1, -0.933), (-0.14, 0.43)) or point_in_ROI((13, 16), (0.93, 1), (-0.31, 0.38))):
              print(image_index, "ㄹ")

            # ㄷ
            elif 0 <= angle_index(8, 5, 12) <= 50 and (point_in_ROI((5, 8), (-1, -0.90), (-0.5, 0.1)) or point_in_ROI((5, 8), (0.83, 1), (-0.58, 0.08))) and \
                 (point_in_ROI((9, 12), (-1, -0.87), (-0.17, 0.58)) or point_in_ROI((9, 12), (0.83, 1), (-0.17, 0.5))):
              print(image_index, "ㄷ")

            # ㅁ
            elif (point_in_ROI((2, 4), (-1, -0.5), (-1, -0.2)) or point_in_ROI((2, 4), (0.4, 1), (-1, -0.1))) and (
                    point_in_ROI((5, 8), (-1, 1), (-1, -0.7)) or point_in_ROI((5, 8), (-1, 1), (-1, -0.8))) \
                    and (point_in_ROI((9, 12), (-0.5, 1), (-1, 1)) or point_in_ROI((9, 12), (-1, 0.5), (-1, -0.8))):
              print(image_index , " ㅁ")

            # ㅂ
            elif 5 <= angle_index(8, 5, 12) <= 20 and 5 <= angle_index(12, 9, 16) <= 20 and 165 <= angle_index(5, 6, 7) <= 180 \
                    and 7 <= angle_index(16, 13, 20) <= 25 and (point_in_ROI((2, 4), (-0.95, -0.2), (-1, -0.3)) or point_in_ROI((2, 4), (0.2, 1), (-1, -0.1))) \
                    and (point_in_ROI((5, 8), (-0.3, 0.5), (-1, -0.8)) or point_in_ROI((5, 8), (-0.2, 0.3), (-1, -0.8))) \
                    and (point_in_ROI((14, 16), (-0.5, 0.5), (-1, -0.8)) or point_in_ROI((14, 16), (-0.2, 0.3), (-1, -0.8))) and 1 <= angle_index(12, 6, 8) <= 30:
              print(image_index , " ㅂ")
              i += 1

            # ㅅ
            elif 1 <= angle_index(16, 0, 4) <= 35 and 34 <= angle_index(12, 6, 8) <= 90 and 145 <= angle_index(5, 6, 7) <= 180 \
              and 150 <= angle_index(9, 10, 11) <= 180 and (point_in_ROI((2, 4), (-0.8, 0.5), (0.6, 1)) or point_in_ROI((2, 4), (-0.4, 0.7), (0.7, 1))) \
              and (point_in_ROI((6, 8), (-0.1, 1), (0.8, 1)) or point_in_ROI((6, 8), (-0.5, 0), (0.8, 1))) \
              and (point_in_ROI((10, 12), (-0.7, 0.6), (0.7, 1)) or point_in_ROI((10, 12), (-0.5, 0.5),(0.8, 1))) and 1 <= angle_index(8, 5, 12) <= 70:
              print(image_index , " ㅅ")
              i += 1

            # ㅇ
            elif 160 <= angle_index(9, 10, 11) <= 180 and 160 <= angle_index(13, 14, 15) <= 180 and 145 <= angle_index(17, 18, 19) <= 180 \
                    and 5 <= angle_index(12, 9, 16) <= 45 and 1 <= angle_index(16, 13, 20) <= 60 \
                    and (point_in_ROI((9, 12), (-0.6, 0.6), (-1, -0.8)) or point_in_ROI((9, 12), (-0.6, 0.3), (-1, -0.8))):
              print(image_index , " o")
              i += 1

            # ㅈ
            elif 170 <= angle_index(5, 6, 7) <= 180 and 170 <= angle_index(9, 10, 11) <= 180 and 10 <= angle_index(8, 5, 12) <= 50 \
                    and (point_in_ROI((2, 4), (0.3, 1), (0.2, 1)) or point_in_ROI((2, 4), (-1, -0.4), (0.2, 1))) \
                    and (point_in_ROI((10, 12), (-0.6, 0.5), (0.8, 1)) or point_in_ROI((10, 12), (-0.4, 0.6), (0.8, 1))) \
                    and (point_in_ROI((6, 8), (-0.6, 0.6), (0.8, 1)) or point_in_ROI((6, 8), (-0.6, 0.4),(0.8, 1))) and 35 <= angle_index(16, 0, 4) <= 80:
              print(image_index , " ㅈ")
              i += 1

            # ㅊ (flip까지 완료)
            elif 25 <= angle_index(8, 2, 3) <= 80 and 170 <= angle_index(5, 6, 8) <= 180 and 170 <= angle_index(9, 10, 12) <= 180 and 170 <= angle_index(13, 14, 16) <= 180 and \
              (point_in_ROI((2, 4), (0.73, 1), (-0.3, 0.7)) or point_in_ROI((2, 4), (-1, 0), (0, 1))) and \
              (point_in_ROI((8, 5), (-0.5, 0.2), (-1, -0.8)) or point_in_ROI((8, 5), (-0.2, 0.3), (-1, -0.8))) and \
              (point_in_ROI((12, 9), (-0.5, 0.5), (-1, -0.8)) or point_in_ROI((12, 9), (-0.4, 0.4), (-1, -0.8))) and point_in_ROI((16, 13), (-0.5, 0.2), (-1, -0.8)):
              print(image_index , " ㅊ")
              i += 1

            # ㅋ (flip까지 완료)
            elif 45 <= angle_index(12, 9, 4) <= 140 and (point_in_ROI((5, 4), (0.7, 1), (-0.6, 0.5)) or point_in_ROI((5, 4), (-1, -0.6), (-0.6, 0.7))) and \
              165 <= angle_index(9, 10, 12) <= 180 and point_in_ROI((12, 9), (-0.5, 0.4), (-1, -0.8)) and \
                    list_z[8] > list_z[6] and list_z[16] > list_z[14] and list_z[20] >list_z[18]:
              print(image_index , " ㅋ")
              i += 1

            # ㅌ (flip까지 완료)
            elif 165 <= angle_index(8, 7, 1) <= 180 and angle_index(10, 9, 6) > angle_index(14, 13, 10) and 165 <= angle_index(5, 6, 8) <= 180 and \
                    165 <= angle_index(9, 10, 12) <= 180 and 165 <= angle_index(13, 14, 16) <= 180 \
                    and (point_in_ROI((5, 8), (0.7, 1), (-0.7, 0.2)) or point_in_ROI((5, 8), (-1, -0.7), (-0.7, 0.2))) \
                    and (point_in_ROI((9, 12), (0.7, 1), (-0.4, 0.5)) or point_in_ROI((9, 12), (-1, -0.7), (-0.4, 0.5))) \
                    and (point_in_ROI((13, 16), (0.7, 1), (-0.4, 0.5)) or point_in_ROI((13, 16), (-1, -0.7), (-0.4, 0.5))) \
                    and (point_in_ROI((10, 12), (0.7, 1), (-0.4, 0.5)) or point_in_ROI((10, 12), (-1, -0.7), (-0.4, 0.5))) \
                    and (point_in_ROI((14, 16), (0.7, 1), (-0.4, 0.5)) or point_in_ROI((14, 16), (-1, -0.7), (-0.4, 0.5))) \
                    and (point_in_ROI((11, 12), (0.7, 1), (-0.4, 0.5)) or point_in_ROI((11, 12), (-1, -0.7), (-0.4, 0.5))) \
                    and (point_in_ROI((15, 16), (0.7, 1), (-0.4, 0.5)) or point_in_ROI((15, 16), (-1, -0.7), (-0.4, 0.5))) \
                    and list_z[4] > list_z[5] and list_z[20] > list_z[19] and list_z[19] > list_z[18]:
              print(image_index , " ㅌ")
              i += 1

            # ㅍ (flip까지 완료)
            elif 85 <= angle_index(4, 3, 2) <= 130 and list_z[5] > list_z[8] and list_z[9] > list_z[12] and \
              list_z[13] >list_z[16] and list_z[17] > list_z[20]:
              print(image_index , " ㅍ")
              i += 1

            # ㅎ (flip까지 완료)
            elif point_in_ROI((4, 2), (-0.65, 0.6), (0.75, 1)) and (point_in_ROI((5, 6), (0.8, 1), (-0.3, 0.6)) or point_in_ROI((5, 6), (-1, -0.8), (-0.3, 0.6))) \
                    and (point_in_ROI((9, 10), (0.8, 1), (-0.3, 0.6)) or point_in_ROI((9, 10), (-1, -0.8), (-0.3, 0.6))) \
                    and (point_in_ROI((13, 14), (0.8, 1), (-0.3, 0.6)) or point_in_ROI((13, 14), (-1, -0.8), (-0.3, 0.6))) \
                    and (point_in_ROI((17, 18), (0.8, 1), (-0.3, 0.6)) or point_in_ROI((17, 18), (-1, -0.8), (-0.3, 0.6))) \
                    and list_z[5] < list_z[8] and list_z[9] < list_z[12] and list_z[13] < list_z[16] and list_z[17] <list_z[20]:
              print(image_index , " ㅎ")
              i += 1
            else:
              print(image_index, " nothing")
              count_nothing += 1

        image_index += 1

        cv2.imshow('MediaPipe Hands', image)

        if cv2.waitKey(33) & 0xFF == 27:
          break

