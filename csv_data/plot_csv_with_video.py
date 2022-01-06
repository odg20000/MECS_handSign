import cv2
import mediapipe as mp
import sys
import numpy as np
import pandas as pd
import time
import csv
import os.path
import numpy


def print_landmarksNindex(img, landmarks):
    shape = img.shape

    for i in landmarks:#----------------------------------------------------------------
        landmark_x = int(landmarks.landmark[i].x * shape[1])
        landmark_y = int(landmarks.landmark[i].y * shape[0])
        cv2.circle(image, (landmark_x, landmark_y), 2, (0, 0, 225), -1)
        #cv2.putText(image, str(i), (landmark_x, landmark_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 225, 0), 0)

    return img


# video capture
source_video_path = 'C:\\Users\\PKNU\\Downloads\\hand_sign_THANKYOU2.mp4'
source_csv_path = 'C:\\Users\\PKNU\\Downloads\\respect02.csv'

#read csv ---------------------------------------------------------------------------------

cap = cv2.VideoCapture(source_video_path)
length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS)
print("# of frames", cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("fps: ", cap.get(cv2.CAP_PROP_FPS))

count = 0
while cap.isOpened():
    success, image = cap.read()
    image = cv2.flip(image, 1)
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        break
    count += 1

    # Draw landmark annotation on the image. holistic
    # seperate csv ---------------------------------------------------------------------------------
    landmark = ....
    image = print_landmarksNindex(image, landmark)

    # Flip the image horizontally for a selfie-view display.
    image = cv2.resize(image, (960, 540))
    cv2.imshow('MediaPipe Hands', image)
    # cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
