import cv2
import mediapipe as mp
import sys
import numpy as np
import pandas as pd
import time
import csv
import os.path
import numpy


def print_landmarksNindex(img, landmarks, index_list):
    # print("landmarks: ", landmarks) #인식 불가시 None
    # print("landmarks.landmark: ", landmarks.landmark)
    if landmarks is None:
        # print("this landmarks are not access!!!!!!!!!!!11")
        return img
    shape = img.shape

    for i in index_list:
        landmark_x = int(landmarks.landmark[i].x * shape[1])
        landmark_y = int(landmarks.landmark[i].y * shape[0])
        cv2.circle(img, (landmark_x, landmark_y), 2, (0, 0, 225), -1)
        cv2.putText(img, str(i), (landmark_x, landmark_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 225, 0), 0)

    return img


# video capture
source_path = 'C:\\Users\\PKNU\\Downloads\\hand_sign_THANKYOU2.mp4'
file_name = 'C:\\Users\\PKNU\\Downloads\\respect02.csv'


csv_index = ['FRAME_NUM',
             'face_X[10]', 'face_X[234]', 'face_X[152]', 'face_X[454]',
             'face_Y[10]', 'face_Y[234]', 'face_Y[152]', 'face_Y[454]',
             'face_Z[10]', 'face_Z[234]', 'face_Z[152]', 'face_Z[454]',

             'body_X[11]', 'body_X[12]', 'body_X[13]', 'body_X[14]', 'body_X[15]', 'body_X[16]', 'body_X[17]',
             'body_X[18]', 'body_X[19]', 'body_X[20]', 'body_X[21]', 'body_X[22]',
             'body_Y[11]', 'body_Y[12]', 'body_Y[13]', 'body_Y[14]', 'body_Y[15]', 'body_Y[16]', 'body_Y[17]',
             'body_Y[18]', 'body_Y[19]', 'body_Y[20]', 'body_Y[21]', 'body_Y[22]',
             'body_Z[11]', 'body_Z[12]', 'body_Z[13]', 'body_Z[14]', 'body_Z[15]', 'body_Z[16]', 'body_Z[17]',
             'body_Z[18]', 'body_Z[19]', 'body_Z[20]', 'body_Z[21]', 'body_Z[22]',

             'R_WRIST_X', 'R_THUMB_CMC_X', 'R_THUMB_MCP_X', 'R_THUMB_IP_X', 'R_THUMB_TIP_X', 'R_INDEX_FINGER_MCP_X',
             'R_INDEX_FINGER_PIP_X', 'R_INDEX_FINGER_DIP_X', 'R_INDEX_FINGER_TIP_X', 'R_MIDDLE_FINGER_MCP_X',
             'R_MIDDLE_FINGER_PIP_X', 'R_MIDDLE_FINGER_DIP_X', 'R_MIDDLE_FINGER_TIP_X', 'R_RING_FINGER_MCP_X',
             'R_RING_FINGER_PIP_X', 'R_RING_FINGER_DIP_X', 'R_RING_FINGER_TIP_X', 'R_PINKY_MCP_X', 'R_PINKY_PIP_X',
             'R_PINKY_DIP_X', 'R_PINKY_TIP_X',
             'R_WRIST_Y', 'R_THUMB_CMC_Y', 'R_THUMB_MCP_Y', 'R_THUMB_IP_Y', 'R_THUMB_TIP_Y', 'R_INDEX_FINGER_MCP_Y',
             'R_INDEX_FINGER_PIP_Y', 'R_INDEX_FINGER_DIP', 'R_INDEX_FINGER_TIP', 'R_MIDDLE_FINGER_MCP',
             'R_MIDDLE_FINGER_PIP_Y', 'R_MIDDLE_FINGER_DIP_Y', 'R_MIDDLE_FINGER_TIP_Y', 'R_RING_FINGER_MCP_Y',
             'R_RING_FINGER_PIP_Y', 'R_RING_FINGER_DIP_Y', 'R_RING_FINGER_TIP_Y', 'R_PINKY_MCP_Y', 'R_PINKY_PIP_Y',
             'R_PINKY_DIP_Y', 'R_PINKY_TIP_Y',
             'R_WRIST_Z', 'R_THUMB_CMC_Z', 'R_THUMB_MCP_Z', 'R_THUMB_IP_Z', 'R_THUMB_TIP_Z', 'R_INDEX_FINGER_MCP_Z',
             'R_INDEX_FINGER_PIP_Z', 'R_INDEX_FINGER_DIP_Z', 'R_INDEX_FINGER_TIP_Z', 'R_MIDDLE_FINGER_MCP_Z',
             'R_MIDDLE_FINGER_PIP_Z', 'R_MIDDLE_FINGER_DIP_Z', 'R_MIDDLE_FINGER_TIP_Z', 'R_RING_FINGER_MCP_Z',
             'R_RING_FINGER_PIP_Z', 'R_RING_FINGER_DIP_Z', 'R_RING_FINGER_TIP_Z', 'R_PINKY_MCP_Z', 'R_PINKY_PIP_Z',
             'R_PINKY_DIP_Z', 'R_PINKY_TIP_Z',

             'L_WRIST_X', 'L_THUMB_CMC_X', 'L_THUMB_MCP_X', 'L_THUMB_IP_X', 'L_THUMB_TIP_X', 'L_INDEX_FINGER_MCP_X',
             'L_INDEX_FINGER_PIP_X', 'L_INDEX_FINGER_DIP_X', 'L_INDEX_FINGER_TIP_X', 'L_MIDDLE_FINGER_MCP_X',
             'L_MIDDLE_FINGER_PIP_X', 'L_MIDDLE_FINGER_DIP_X', 'L_MIDDLE_FINGER_TIP_X', 'L_RING_FINGER_MCP_X',
             'L_RING_FINGER_PIP_X', 'L_RING_FINGER_DIP_X', 'L_RING_FINGER_TIP_X', 'L_PINKY_MCP_X',
             'L_PINKY_PIP_X', 'L_PINKY_DIP_X', 'L_PINKY_TIP_X',
             'L_WRIST_Y', 'L_THUMB_CMC_Y', 'L_THUMB_MCP_Y', 'L_THUMB_IP_Y', 'L_THUMB_TIP_Y', 'L_INDEX_FINGER_MCP_Y',
             'L_INDEX_FINGER_PIP_Y', 'L_INDEX_FINGER_DIP', 'L_INDEX_FINGER_TIP', 'L_MIDDLE_FINGER_MCP',
             'L_MIDDLE_FINGER_PIP_Y', 'L_MIDDLE_FINGER_DIP_Y', 'L_MIDDLE_FINGER_TIP_Y', 'L_RING_FINGER_MCP_Y',
             'L_RING_FINGER_PIP_Y', 'L_RING_FINGER_DIP_Y', 'L_RING_FINGER_TIP_Y', 'L_PINKY_MCP_Y',
             'L_PINKY_PIP_Y', 'L_PINKY_DIP_Y', 'L_PINKY_TIP_Y',
             'L_WRIST_Z', 'L_THUMB_CMC_Z', 'L_THUMB_MCP_Z', 'L_THUMB_IP_Z', 'L_THUMB_TIP_Z', 'L_INDEX_FINGER_MCP_Z',
             'L_INDEX_FINGER_PIP_Z', 'L_INDEX_FINGER_DIP_Z', 'L_INDEX_FINGER_TIP_Z', 'L_MIDDLE_FINGER_MCP_Z',
             'L_MIDDLE_FINGER_PIP_Z', 'L_MIDDLE_FINGER_DIP_Z', 'L_MIDDLE_FINGER_TIP_Z', 'L_RING_FINGER_MCP_Z',
             'L_RING_FINGER_PIP_Z', 'L_RING_FINGER_DIP_Z', 'L_RING_FINGER_TIP_Z', 'L_PINKY_MCP_Z',
             'L_PINKY_PIP_Z', 'L_PINKY_DIP_Z', 'L_PINKY_TIP_Z']

def extract_landmarks(results):
    face = [10, 234, 152, 454]
    body = range(11, 23)

    # csv data list
    total_list = []

    face_list_x = []
    face_list_y = []
    face_list_z = []
    pose_list_x = []
    pose_list_y = []
    pose_list_z = []
    right_finger_list_x = []
    right_finger_list_y = []
    right_finger_list_z = []
    left_finger_list_x = []
    left_finger_list_y = []
    left_finger_list_z = []

    if results.face_landmarks is None:
        face_list_x = [0 for i in range(4)]
        face_list_y = [0 for i in range(4)]
        face_list_z = [0 for i in range(4)]
    else:
        for face_index in face:
            face_list_x.append(results.face_landmarks.landmark[face_index].x)
            face_list_y.append(results.face_landmarks.landmark[face_index].y)
            face_list_z.append(results.face_landmarks.landmark[face_index].z)

    if results.pose_landmarks is None:
        pose_list_x = [0 for i in range(12)]
        pose_list_y = [0 for i in range(12)]
        pose_list_z = [0 for i in range(12)]
    else:
        for pose_index in body:
            pose_list_x.append(results.pose_landmarks.landmark[pose_index].x)
            pose_list_y.append(results.pose_landmarks.landmark[pose_index].y)
            pose_list_z.append(results.pose_landmarks.landmark[pose_index].z)

    if results.left_hand_landmarks is None:
        left_finger_list_x = [0 for i in range(21)]
        left_finger_list_y = [0 for i in range(21)]
        left_finger_list_z = [0 for i in range(21)]
    else:
        for finger_index_left in range(21):
            left_finger_list_x.append(results.left_hand_landmarks.landmark[finger_index_left].x)
            left_finger_list_y.append(results.left_hand_landmarks.landmark[finger_index_left].y)
            left_finger_list_z.append(results.left_hand_landmarks.landmark[finger_index_left].z)

    if results.right_hand_landmarks is None:
        right_finger_list_x = [0 for i in range(21)]
        right_finger_list_y = [0 for i in range(21)]
        right_finger_list_z = [0 for i in range(21)]
    else:
        for finger_index_right in range(21):
            right_finger_list_x.append(results.right_hand_landmarks.landmark[finger_index_right].x)
            right_finger_list_y.append(results.right_hand_landmarks.landmark[finger_index_right].y)
            right_finger_list_z.append(results.right_hand_landmarks.landmark[finger_index_right].z)

    # x-> y-> z or all_x-> all_y-> all_z/////////////////////////////////////////////////////////////////////////////////////
    total_list = face_list_x + face_list_y + face_list_z + \
                 pose_list_x + pose_list_y + pose_list_z + \
                 right_finger_list_x + right_finger_list_y + right_finger_list_z + \
                 left_finger_list_x + left_finger_list_y + left_finger_list_z
    return total_list
def set_relative_axial(twoD_list):
    # 각 좌표들의 정규화 필요
    # 1. 모든 이미지 크기 동일화 = 원본이미지의 너비와 높이로 나눠주면 문제 없음.
    # 하나의 죄표를 원점으로 하는 상대좌표 화가 필요 -> 절대로 인식되는 좌표를 원점으로!
    # 전제: 얼굴이 포함되는 영상을 찍음을 전제함.-> 턱 좌표를 원점으로 :: 턱 x = 2, y = 6, z = 10
    # face 4개 x= 0 to 3, body 12개 x= 12 to 23, 한 손 21개 right_x = 48 to 68, left_x = 111 to 131
    # face y =                     y =                    right_y =           left_y =
    # face z =                     z =                    right_z =           left_z =
    twoD_array = np.array(twoD_list)
    # x 상대 좌표 처리
    # print(list(range(1, 5))+list(range(13, 25))+list(range(49, 70))+list(range(112, 133)))
    for i in list(range(0, 4)) + list(range(12, 24)) + list(range(48, 69)) + list(range(111, 132)):
        twoD_array[:, i] -= twoD_array[:, 2]
    # y 상대 좌표
    for i in list(range(4, 8)) + list(range(24, 36)) + list(range(69, 90)) + list(range(132, 153)):
        twoD_array[:, i] -= twoD_array[:, 6]
    # z 상대 좌표 처리
    for i in list(range(8, 12)) + list(range(36, 48)) + list(range(90, 111)) + list(range(153, 174)):
        twoD_array[:, i] -= twoD_array[:, 10]

    return twoD_array

def save_3csv_landmarks(source_path, file_name, csv_index):
    mp_drawing = mp.solutions.drawing_utils
    #mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic

    face = [10, 234, 152, 454]
    body = range(11, 23)

    cap = cv2.VideoCapture(source_path)
    # cap = cv2.VideoCapture(0)                          #.////////////////////////////////////////////////////////////////
    length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("# of frames", cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("fps: ", cap.get(cv2.CAP_PROP_FPS))

    # save to csv
    '''
    orig_stdout = sys.stdout
    sys.stdout = open(file_name, 'w', newline='')
    wr = csv.writer(sys.stdout)
    wr.writerow(csv_index)
    sys.stdout.close()
    '''
    count = 0
    twoD_list = []

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            success, image = cap.read()
            image = cv2.flip(image, 1)
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break
            count += 1

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)

            total_list = extract_landmarks(results)
            #sys.stdout = open(file_name, 'a', newline='')
            #wr = csv.writer(sys.stdout)
            #wr.writerow([count] + [i for i in total_list])
            #sys.stdout.close()
            #sys.stdout = orig_stdout
            twoD_list.append(total_list)

            # print to imshow landmark
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # Draw landmark annotation on the image. holistic
            image = print_landmarksNindex(image, results.face_landmarks, face)
            image = print_landmarksNindex(image, results.pose_landmarks, body)

            # Draw the hand annotations on the image.
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # Flip the image horizontally for a selfie-view display.
            image = cv2.resize(image, (960, 540))
            cv2.imshow('MediaPipe Hands', image)
            # cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))

            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()

    twoD_array = set_relative_axial(twoD_list)
    df = pd.DataFrame(twoD_array)
    df.columns = csv_index[1:]
    # print(len(csv_index))
    df.to_csv(file_name.replace(".csv", "_relative coordinates.csv"))


# ----------------------------------------------------------------------finish to make csv/array
# start to interpolate
# 1. bilinear interpolation
# 2. near이웃 보간
# 모든 영상에 대해서 인식이 안됀 인덱스는 0으로 그대로 유지시킨다.
#또한 영상의 전 후로 인식이 될 부분이 영상이 드러나지 않은경우는 0으로 처리하는 것으로 본다.

    twoD_array_interpolated = preprocessing_inter(twoD_array, 0)
    df = pd.DataFrame(twoD_array_interpolated)
    df.columns = csv_index[1:]
    # print(len(csv_index))
    df.to_csv(file_name.replace(".csv", "_relative coordinates_interpolation_linear.csv"))

    twoD_array_discrete = preprocessing_inter(twoD_array, 1)
    df = pd.DataFrame(twoD_array_discrete)
    df.columns = csv_index[1:]
    # print(len(csv_index))
    df.to_csv(file_name.replace(".csv", "_relative coordinates_interpolation_discret.csv"))

def preprocessing_inter(img_array, flag=0):
    interpolated_array = img_array.copy()
    # 0인 값 찾기
    for col in range(img_array.shape[1]):
        for row in range(img_array.shape[0]):
            if row == 0:
                continue
            if img_array[row, col] == 0:  # ------------is empty! Have to interpolate

                a = {'x': 0, 'y': 0}  # 이전 값 추적
                b = {'x': 0, 'y': 0}  # 이후 값 추적

                for i in range(row):
                    a['y'] = img_array[row - (i + 1), col]
                    a['x'] = row - (i + 1)
                    if a['y'] != 0:
                        break
                for i in range(img_array.shape[0] - row - 1):
                    b['y'] = img_array[row + (i + 1), col]
                    b['x'] = row + (i + 1)
                    if b['y'] != 0:
                        break

                if b['y'] == 0:
                    # this is 제일 마지막에 아예 사라지는 값 처리
                    break
                elif a['y'] == 0:
                    continue
                else:  # 앞뒤로 값을 가지고 있는 경우!
                    if flag == 0:  # linear-interpolation
                        #print("this is linear")
                        interpolated_array[row, col] = (b['y'] - a['y']) / (b['x'] - a['x']) * (row - a['x']) + a['y']
                    else:  # nn-interpolation
                        interpolated_array[row, col] = a['y']

    return interpolated_array


# data 양 늘리기 -> 마지막 혹은 첫 몇가지의 프레임들을 빼는 것으로 데이터 증량
#data argumentation
#좌우 대칭으로 만들기 각 픽셀에-1 곱하면 될듯
#10프레임을 사용하는것으로 가정한다면
#모든 데이터들을 100개 프레임으로 RESCALE하고 10Hz으로 SAMPLING하기
#이후에 10N+I의 9개 추가 데이터들 확보 가능

#몇프레임으로 학습시킬지에 대한 것은 여러개를 해봐야 알듯!
def argumentation_save(video_keypoints, DATA_PATH, action, name, sampling_frame=10):
    video_keypoints = cv2.resize(video_keypoints, dsize=(video_keypoints.shape[1], sampling_frame*10), interpolation=cv2.INTER_AREA)

    #10배 데이터 증강
    for i in range(10):
        video_sampling = video_keypoints[i::10, :]
        npy_path = os.path.join(DATA_PATH, action, name + f'-({i})')
        np.save(npy_path, video_sampling)

    # 좌우반전
    for i in range(10):
        video_sampling = video_keypoints[i::10, :]*(-1)
        npy_path = os.path.join(DATA_PATH, action, name + f'-({10+i})')
        np.save(npy_path, video_sampling)


def argumentation(video_keypoints, sampling_frame=10):
    video_keypoints = cv2.resize(video_keypoints, dsize=(video_keypoints.shape[1], sampling_frame * 10), interpolation=cv2.INTER_AREA)
    sequence = []

    # 10배 데이터 증강
    for i in range(10):
        video_sampling = video_keypoints[i::10, :]
        sequence.append(video_sampling)

    # 좌우반전
    for i in range(10):
        video_sampling = video_keypoints[i::10, :] * (-1)
        sequence.append(video_sampling)

    return np.array(sequence)

