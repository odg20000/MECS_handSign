import cv2
import mediapipe as mp

def print_landmarksNindex(img, landmarks, index_list):
    #print("landmarks: ", landmarks) #인식 불가시 None
    #print("landmarks.landmark: ", landmarks.landmark)
    if landmarks is None:
        #print("this landmarks are not access!!!!!!!!!!!11")
        return img
    shape = img.shape

    for i in index_list:
        landmark_x = int(landmarks.landmark[i].x * shape[1])
        landmark_y = int(landmarks.landmark[i].y * shape[0])
        cv2.circle(image, (landmark_x, landmark_y), 2, (0, 0, 225), -1)
        cv2.putText(image, str(i), (landmark_x, landmark_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 225, 0), 0)

    return img


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

face = [10, 234, 152, 454]

# For webcam input:
#cap = cv2.VideoCapture('C:\\Users\\PKNU\\Downloads\\hand_sign_THANKYOU2.mp4')
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

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Draw landmark annotation on the image. holistic
    image = print_landmarksNindex(image, results.face_landmarks, face)
    image = print_landmarksNindex(image, results.pose_landmarks, range(11, 23))

    # Draw the hand annotations on the image.
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    # Flip the image horizontally for a selfie-view display.
    image = cv2.resize(image, (960, 540))
    cv2.imshow('MediaPipe Hands', image)
    #cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))

    if cv2.waitKey(100) & 0xFF == 27:
      break

cap.release()
