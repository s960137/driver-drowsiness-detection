#   python eyes_mouths.py --shape-predictor shape_predictor_68_face_landmarks.dat

import cv2
import dlib
import time
import pygame
from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np

# 初始化pygame用于聲音播放
pygame.mixer.init()
alert_sound = pygame.mixer.Sound("alert.wav")

# 初始化dlib的face檢測器和shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # 換為shape predictor路徑

# 计算眼睛纵横比
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# 计算嘴巴纵横比
def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[13], mouth[19])
    B = dist.euclidean(mouth[14], mouth[18])
    C = dist.euclidean(mouth[15], mouth[17])
    D = dist.euclidean(mouth[12], mouth[16])
    mar = (A + B + C) / (2.0 * D)
    return mar

EYE_AR_THRESH = 0.23
EYE_AR_CONSEC_FRAMES = 3
FATIGUE_EAR_THRESH = 1.0  # 眨眼時間超過1秒為疲勞
MOUTH_AR_THRESH = 0.75

COUNTER = 0
TOTAL = 0
CONSEC_BLINKS = 0
CONSEC_YAWNS = 0

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

cap = cv2.VideoCapture(0)
start_time = time.time()
blink_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    fps = cap.get(cv2.CAP_PROP_FPS)  # 得到偵率

    for rect in faces:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        mar = mouth_aspect_ratio(mouth)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)

        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            COUNTER += 1
            CONSEC_BLINKS += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
                COUNTER = 0
                blink_count += 1
        else:
            if CONSEC_BLINKS >= EYE_AR_CONSEC_FRAMES:
                blink_duration = CONSEC_BLINKS / fps
                if blink_duration >= FATIGUE_EAR_THRESH:
                    cv2.putText(frame, "FATIGUE DETECTED", (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    pygame.mixer.Sound.play(alert_sound)  # 播放警告聲
            CONSEC_BLINKS = 0

        if mar > MOUTH_AR_THRESH:
            CONSEC_YAWNS += 1
            if CONSEC_YAWNS >= 1:  # 一次打哈欠就觸發警告
                cv2.putText(frame, "YAWN DETECTED", (10, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                pygame.mixer.Sound.play(alert_sound)  # 播放警告聲
                CONSEC_YAWNS = 0
        else:
            CONSEC_YAWNS = 0

        cv2.putText(frame, "EAR: {:.2f}".format(ear), (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "MAR: {:.2f}".format(mar), (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    elapsed_time = time.time() - start_time

    if elapsed_time > 60:
        bpm = blink_count / (elapsed_time / 60.0)
        print("Last minute blinks:", blink_count)
        print("Blinks per minute (BPM):", bpm)

        if bpm > 30:
            cv2.putText(frame, "FATIGUE DETECTED", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            pygame.mixer.Sound.play(alert_sound)  # 播放警告聲

        blink_count = 0
        start_time = time.time()

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
