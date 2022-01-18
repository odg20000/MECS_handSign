import cv2
import numpy as np
import csv

def print_landmarksNindex3(img, landmark):
    shape = img.shape

    landmark_x = []
    landmark_y = []

    landmark_x.append(landmark[0:4])
    landmark_x.append(landmark[12:24])
    landmark_x.append(landmark[48:69])
    landmark_x.append(landmark[111:132])
    landmark_y.append(landmark[4:8])
    landmark_y.append(landmark[24:36])
    landmark_y.append(landmark[69:90])
    landmark_y.append(landmark[132:153])

    landmark_x_con = np.concatenate(landmark_x).tolist()
    landmark_y_con = np.concatenate(landmark_y).tolist()

    for i in range(58):
        landmark_x_draw = int(landmark_x_con[i] * shape[1])
        landmark_y_draw = int(landmark_y_con[i] * shape[0])
        cv2.circle(image, (landmark_x_draw, landmark_y_draw), 2, (0, 0, 225), -1)

    return img

# video capture
source_video_path = 'C:/gyuri/MECS/video/hand_sign_DAD.mp4'

#read csv ---------------------------------------------------------------------------------
data = []
file = open('test01_nninterpolation.csv','r')
reader = csv.reader(file)
for line in reader:
    data.append(line[1:]) #frame num빼고 나머지만 append
file.close()

twoD_array = np.array(data[1:]) #첫줄 index 제외하고 나머지만 저장
twoD_array=twoD_array.astype(float)

# face 4개 x = 0 to 3 / body 12개 x = 12 to 23 / 한 손 21개 right_x = 48 to 68 / left_x = 111 to 131
# face    y = 4 to 7            y = 24 to 35              right_y = 69 to 89   left_y = 132 to 152
# face    z = 8 to 11           z = 36 to 47              right_z = 90 to 110  left_z = 153 to 173

cap = cv2.VideoCapture(source_video_path)
length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS)
print("# of frames", cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("fps: ", cap.get(cv2.CAP_PROP_FPS))

count = 0
frame = 0
out = cv2.VideoWriter('output.mp4', -1, 20.0, (960, 540))
while cap.isOpened():
    success, image = cap.read()
    image = cv2.flip(image, 1)
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        break

    landmark = twoD_array[count, :]
    count += 1
    image = print_landmarksNindex3(image, landmark)

    # Flip the image horizontally for a selfie-view display.
    image = cv2.resize(image, (960, 540))
    cv2.imshow('MediaPipe Hands', image)
    # cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    out.write(image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
