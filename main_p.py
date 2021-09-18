import cv2
from mediapipe_HandDetectionModule import HandDetection
from mediapipe_fullbodyPosModule import PosDetection
from realsense_depth import *
import math
import time

hand = HandDetection(max_num_hands=1)
full_body = PosDetection(min_detection_confidence=0.9)
dc = DepthCamera()

# Initialization
frame_size = (1280, 720)
lower_bond = (520, 140)
upper_bond = (1260, 140)
operation_point = (40, 420)
rect_bond = (1260, 700)
ptime = 0
fps = 0
frac = 0
distance = math.dist(lower_bond, upper_bond)
frame_delay = 0

# result = cv2.VideoWriter('final.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, frame_size)

while True:
    ret, depth_frame, color_frame = dc.get_frame()
    color_frame = cv2.flip(color_frame, 1)
    cv2.rectangle(color_frame, lower_bond, rect_bond, (0, 0, 255), 3)
    cv2.circle(color_frame, lower_bond, 10, (0, 255, 0), -1)
    cv2.putText(color_frame, 'Start', (lower_bond[0] - 50, lower_bond[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0),
                2)
    cv2.circle(color_frame, upper_bond, 10, (0, 255, 0), -1)

    ROI_frame = color_frame[140:700, 520:1260]

    lms = full_body.findLandmarkloc(ROI_frame, False)
    # hand_lms = hand.handLandmark(color_frame)
    # if len(hand_lms) != 0:
    if len(lms) != 0:
        cv2.rectangle(color_frame, (25, 660), (380, 700), (255, 0, 255), -1)
        cv2.putText(color_frame, 'Right Hand Detected', (25, 690), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)

        right_shoulder_x, right_shoulder_y = lms[11][1], lms[11][2]
        right_elbow_x, right_elbow_y = lms[13][1], lms[13][2]
        right_wrist_x, right_wrist_y = lms[15][1], lms[15][2]

        cv2.circle(ROI_frame, (right_shoulder_x, right_shoulder_y), 5, (0, 0, 255), -1)
        cv2.circle(ROI_frame, (right_elbow_x, right_elbow_y), 5, (0, 0, 255), -1)
        cv2.circle(ROI_frame, (right_wrist_x, right_wrist_y), 5, (0, 0, 255), -1)

        # if right_wrist_x > upper_bond[0] or right_wrist_x < lower_bond[0] or right_wrist_y < lower_bond[1]:
        #     cv2.putText(color_frame, f'No operation', operation_point, cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)
        # else:
        if right_wrist_y + 10 > right_elbow_y:
            cv2.putText(color_frame, f'No operation', operation_point, cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)
        else:
            if right_elbow_y < right_shoulder_y:
                if frame_delay % 5 == 0:
                    x, y = right_wrist_x, lower_bond[1]
                    dist2 = math.dist((0, 0), (x, y))
                    frac = round((dist2 / distance), 2)
                    frame_delay = 0
                cv2.putText(color_frame, f'FULL,fraction:{frac}', operation_point, cv2.FONT_HERSHEY_PLAIN, 3,
                            (0, 255, 0), 2)

            else:
                if frame_delay % 5 == 0:
                    x, y = right_wrist_x, lower_bond[1]
                    dist2 = math.dist((0, 0), (x, y))
                    frac = round((dist2 / distance), 2)
                    frame_delay = 0
                cv2.putText(color_frame, f'HALF,fraction:{frac}', operation_point, cv2.FONT_HERSHEY_PLAIN, 3,
                            (0, 255, 0), 2)

        cv2.putText(color_frame, f'Total Frame ROI Dist:{int(distance)}', (40, 110), cv2.FONT_HERSHEY_PLAIN, 2,
                    (255, 0, 0), 2)
        cv2.putText(color_frame, f'Dist form start:{round(int(distance) * frac, 2)}', (40, 165), cv2.FONT_HERSHEY_PLAIN,
                    2, (0, 0, 255), 2)
        cv2.putText(color_frame, f'(in +x direction)', (60, 195), cv2.FONT_HERSHEY_PLAIN,
                    2, (255, 0, 255), 1)

    else:
        cv2.rectangle(color_frame, (25, 660), (350, 700), (0, 255, 0), -1)
        cv2.putText(color_frame, 'Hand Not Detected', (25, 690), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

    ctime = time.time()
    if ctime - ptime != 0:
        fps = 1 / (ctime - ptime)
    ptime = ctime
    cv2.putText(color_frame, f'FPS:{int(fps)}', (40, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    frame_delay += 1
    color_frame[lower_bond[1]:lower_bond[1] + ROI_frame.shape[0],
    lower_bond[0]:lower_bond[0] + ROI_frame.shape[1]] = ROI_frame
    # cv2.imshow('Final', ROI_frame)
    cv2.imshow('FUll', color_frame)
    # result.write(color_frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break







