import dlib
import cv2 as cv
import numpy as np
import csv

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor('C:/Users/harby/Desktop/shape_predictor_68_face_landmarks.dat')

cap = cv.VideoCapture('C:/Users/harby/Downloads/hand_sign_DAD.mp4')

ALL = list(range(0, 68))
RIGHT_EYEBROW = list(range(17, 22))
LEFT_EYEBROW = list(range(22, 27))
RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
NOSE = list(range(27, 36))
MOUTH_OUTLINE = list(range(48, 61))
MOUTH_INNER = list(range(61, 68))
JAWLINE = list(range(0, 17))

index = ALL

count = 0
sign_type = '안녕'
IF_FIRST_TIME_TO_DO = 1
file_name = 'C:/Users/harby/Downloads/face_landmark2.csv'


f = open(file_name, 'w', newline='')
writer = csv.writer(f)


def initializer():
    list_init = ['1_x', '1_y', '2_x', '2_y', '3_x', '3_y', '4_x', '4_y', '5_x', '5_y', '6_x', '6_y', '7_x', '7_y', '8_x', '8_y',
         '9_x', '9_y', '10_x', '10_y', '11_x', '11_y', '12_x', '12_y', '13_x', '13_y', '14_x', '14_y', '15_x', '15_y',
         '16_x', '16_y', '17_x', '17_y', '18_x', '18_y', '19_x', '19_y', '20_x', '20_y', '21_x', '21_y', '22_x', '22_y',
         '23_x', '23_y', '24_x', '24_y', '25_x', '25_y', '26_x', '26_y', '27_x', '27_y', '28_x', '28_y', '29_x', '29_y',
         '30_x', '30_y', '31_x', '31_y', '32_x', '32_y', '33_x', '33_y', '34_x', '34_y', '35_x', '35_y', '36_x', '36_y',
         '37_x', '37_y', '38_x', '38_y', '39_x', '39_y', '40_x', '40_y', '41_x', '41_y', '42_x', '42_y', '43_x', '43_y',
         '44_x', '44_y', '45_x', '45_y', '46_x', '46_y', '47_x', '47_y', '48_x', '48_y', '49_x', '49_y', '50_x', '50_y',
         '51_x', '51_y', '52_x', '52_y', '53_x', '53_y', '54_x', '54_y', '55_x', '55_y', '56_x', '56_y', '57_x', '57_y',
         '58_x', '58_y', '59_x', '59_y', '60_x', '60_y', '61_x', '61_y', '62_x', '62_y', '63_x', '63_y', '64_x', '64_y',
         '65_x', '65_y', '66_x', '66_y', '67_x', '67_y', '68_x', '68_y']
    writer.writerow(list_init)


if IF_FIRST_TIME_TO_DO == 1:
    initializer()

while True:

    ret, img_frame = cap.read()

    img_gray = cv.cvtColor(img_frame, cv.COLOR_BGR2GRAY)

    dets = detector(img_gray, 1)

    for face in dets:
        shape = predictor(img_frame, face)  # 얼굴에서 68개 점 찾기

        image_width = img_frame.shape[0]
        image_height = img_frame.shape[1]

        list_points_np = []
        list_points_csv = []
        for p in shape.parts():
            list_points_np.append([p.x, p.y])
            list_points_csv = list_points_csv + [p.x/image_width] + [p.y/image_height]

        writer.writerow(list_points_csv)
        print(list_points_csv, type(list_points_csv))
        list_points_np = np.array(list_points_np)

        for i, pt in enumerate(list_points_np[index]):
            pt_pos = (pt[0], pt[1])
            cv.circle(img_frame, pt_pos, 2, (255, 255, 255), -1)  # 좌표 점 찍기

        cv.rectangle(img_frame, (face.left(), face.top()), (face.right(), face.bottom()),  # 얼굴 사각형 그리기
                     (0, 0, 255), 3)

    cv.imshow('result', img_frame)
    key = cv.waitKey(1)

    if key == 27:
        break

cap.release()

f.close()