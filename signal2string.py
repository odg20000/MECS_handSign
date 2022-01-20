import joblib
import cv2
import mediapipe as mp
import hangul_utils

# alldata -> all data
# v2 -> remove num
# v3 -> remove num, yoo
# v4 -> remove num, pa
# v5 -> remove num, yoo, pa
# v6 -> remove num, ma
# v7 -> remove num, ma, pa
# v8 -> remove num, yoo, ma, pa
# v9 -> remove num, yoo, ma, pa, ye, e
# v10 -> remove num, yoo, ma, pa, ye, e, na
# v11 = v10의 한글버전

# v10에서 럭스, 부경대 두가지 인식 성공 [eo]잘 안됨


alph_dic = {'ga': 'ㄱ', 'ba': 'ㅂ', 'woo': 'ㅜ', 'yeo': 'ㅕ', 'aa': 'ㅇ', 'da': 'ㄷ', 'ae': 'ㅐ', 'ra': 'ㄹ', 'eo': 'ㅓ',
            'sa': 'ㅅ', 'ui': 'ㅢ', 'yae': 'ㅒ', 'ya': 'ㅑ', 'ja': 'ㅈ', 'cha': 'ㅊ', 'ka': 'ㅋ', 'ta': 'ㅌ', 'ha': 'ㅎ',
            'a': 'ㅏ', 'oh': 'ㅗ', 'oe': 'ㅚ', 'yo': 'ㅛ', 'wi': 'ㅟ', 'eu': 'ㅡ', 'lee': 'ㅣ'}

sc = joblib.load('sc_hinge_v10.pkl')


hangul_result = []


def signal_detection(data):
    detected = 0
    present = signal_detection.signal_present = data
    previous = signal_detection.signal_previous
    signal_detection.signal_times

    if present == previous:
        signal_detection.signal_times += 1
        if signal_detection.signal_times >= signal_detection.signal_threshold:
            print('Signal Detected : ', present)
            detected = 1
            signal_detection.signal_times = 0
    else:
        signal_detection.signal_times = 0
    # print('DEBUG::: STATIC VAR CHECK ::: ', signal_detection.signal_times)
    signal_detection.signal_previous = present
    if detected == 1:
        return present
    else:
        return 0


signal_detection.signal_times = 0
signal_detection.signal_previous = ['*']
signal_detection.signal_present = ['*']
signal_detection.signal_threshold = 60


def axis_move(data):
    x_zero = data[0]
    y_zero = data[21]
    z_zero = data[42]
    for i in range(0, 21):
        data[i] = data[i] - x_zero
    for i in range(21, 42):
        data[i] = data[i] - y_zero
    for i in range(42, 63):
        data[i] = data[i] - z_zero
    return data


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:
    while True:
        success, img = cap.read()

        # Convert the BGR image to RGB before processing.
        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        list_of_hand = []
        if not results.multi_hand_landmarks:
            cv2.imshow("HandTracking", img)
            cv2.waitKey(1)
            continue
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                list_of_hand = [
                    hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x,
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x,
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x,
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x,
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x,
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x,
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x,
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x,
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x,
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x,
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x,
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x,
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x,
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x,
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x,
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x,
                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x,
                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x,
                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x,
                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x,
                    hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y,
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y,
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y,
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y,
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y,
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y,
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y,
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y,
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y,
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y,
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y,
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y,
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y,
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y,
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y,
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y,
                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y,
                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y,
                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y,
                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y,
                    hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].z,
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].z,
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].z,
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].z,
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].z,
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].z,
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].z,
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].z,
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z,
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].z,
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].z,
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].z,
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].z,
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].z,
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].z,
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].z,
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].z,
                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].z,
                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].z,
                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].z,
                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].z
                ]

                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imshow("HandTracking", img)
        cv2.waitKey(1)

        list_of_hand = axis_move(list_of_hand)
        temp = [list_of_hand]
        # print(temp)

        # print(*sc.predict(temp))
        result_signal = signal_detection(sc.predict(temp))
        sending = ''.join(sc.predict(temp))
        if signal_detection.signal_times % 15 == 0:
            print(alph_dic[sending])

        if result_signal != 0:
            hangul_result.append(alph_dic[sending])
            print(hangul_result)
            print('[', hangul_utils.join_jamos(hangul_result), ']')
