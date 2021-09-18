import math
import cv2
from mediapipe_fullbodyPosModule import PosDetection
from mediapipe_HandDetectionModule import HandDetection
import time
from All_Function import *
from realsense_depth import DepthCamera

# cap = cv2.VideoCapture(0)
hand = HandDetection(min_detection_confidence=0.5)
full_body = PosDetection()
dc = DepthCamera()

ptime = 0
fps = 0
x, y = 0, 0
cor_x, cor_y = 0, 0

frame_delay = 0
hand_d = 'Hand Not Detected'
full_or_not = 'No operation'
finger_on_or_not = 'Finger Not Detected'
up_condition = None
calibration = False
max_x, max_y = 0, 0
itr = 0
frac_x, frac_y = 0, 0

result = cv2.VideoWriter('final4.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, win_size)
distt_wrist = 0
distt_shoulder = 0

while True:
    # ret, frame = cap.read()
    # print(frame.shape)
    ret, depth, frame = dc.get_frame()
    frame = cv2.resize(frame, win_size)
    frame = cv2.flip(frame, 1)  # flip frame
    cv2.rectangle(frame, lower_bound, rect_bottom_right, (0, 0, 255), 2)

    roi_frame = frame[lower_bound[1]:rect_bottom_right[1], lower_bound[0]:upper_bound[0]]
    roi_depth = depth[lower_bound[1]:rect_bottom_right[1], lower_bound[0]:upper_bound[0]]
    # print(roi_frame.shape)
    # print(roi_depth.shape)

    # print(roi_frame.shape)

    lms = full_body.findLandmarkloc(roi_frame, False)  # detect full body landmark
    hand_lms = hand.handLandmark(roi_frame, False)  # detect finger landmark

    if len(lms) != 0:  # if fulll body detected

        shoulder_x, shoulder_y = lms[11][1], lms[11][2]
        elbow_x, elbow_y = lms[13][1], lms[13][2]
        wrist_x, wrist_y = lms[15][1], lms[15][2]
        cor_x, cor_y = lms[11][1], lms[11][2]
        hand_d = 'Hand Detected'

        is_finger_detect = check_finger_open(frame, lms, hand_lms)  # check which hand finger detect
        # if len(hand_lms) != 0:
        full_or_not = is_hand_up(lms)
        if full_or_not == "FULL" or full_or_not == "HALF":
            x, y = frac_x_and_y(lms)  # find x, y coordinate of right hand wrist
            if wrist_x < roi_depth.shape[0] and wrist_y < roi_depth.shape[1]:
                distt_wrist = roi_depth[wrist_x, wrist_y]
            # distance_shoulder = roi_depth[shoulder_x, shoulder_y]

        if is_finger_detect:
            finger_on_or_not = 'Finger Detected'
            up_condition = check_thumpsUp_or_Down(hand_lms)
            if up_condition == 'DOWN' and full_or_not != 'No operation':
                calibration = False
                max_x, max_y = 0, 0

            elif up_condition == 'UP' and full_or_not != 'No operation':
                calibration = True


        else:
            finger_on_or_not = 'Finger Not Detected'

        cv2.rectangle(frame, (25, win_size[1] - 60), (350, win_size[1] - 20), (0, 255, 0), -1)
        cv2.putText(frame, hand_d, (25, win_size[1] - 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

        draw_wrist_elbow_shoulder(roi_frame, lms)  # draw circle at right hand wrist position



    else:
        hand_d = 'Hand Not Detected'

    if not calibration:
        cv2.putText(roi_frame, 'Please calibrate x and y position....', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1,
                    (0, 255, 0), 1)

        if max_x < x:
            max_x = x

        if max_y < y:
            max_y = y

        # print(calibration)

    # Draw all parameter
    if hand_d == 'Hand Detected':
        wrist_x, wrist_y = lms[15][1], lms[15][2]
        cv2.putText(frame, finger_on_or_not, (25, win_size[1] - 90), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
        draw_coordinate(roi_frame, cor_x, cor_y)
        cv2.putText(frame, up_condition, (25, win_size[1] - 130), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
        cv2.circle(roi_frame, (cor_x + max_x, cor_y), 15, (255, 0, 255), -1)
        cv2.circle(roi_frame, (cor_x, cor_y - max_y), 15, (255, 0, 255), -1)
        cv2.putText(roi_frame, f'wrist_depth:{int((abs(distt_wrist)) / 10)}', (wrist_x, wrist_y),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (51, 255, 255), 1)
    if calibration:
        if max_x < x:
            max_x = x

        if max_y < y:
            max_y = y

        if max_x != 0 and max_y != 0:
            frac_x = x / max_x
            frac_y = y / max_y

        cv2.putText(frame, f"x: {x},y: {y}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, f"frac_x:{round(frac_x, 2)},frac_y:{round(frac_y, 2)}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 0), 2)

        cv2.putText(frame, full_or_not, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.rectangle(frame, (25, win_size[1] - 60), (350, win_size[1] - 20), (0, 255, 0), -1)
    cv2.putText(frame, hand_d, (25, win_size[1] - 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
    cv2.putText(frame, f"max_x:{max_x},max_y:{max_y}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (51, 255, 255), 2)

    # Add roi to full frame
    frame[lower_bound[1]:rect_bottom_right[1], lower_bound[0]:upper_bound[0]] = roi_frame

    # Calculate FPS
    cTime = time.time()
    if (cTime - ptime) != 0:
        fps = 1 / (cTime - ptime)
    ptime = cTime
    cv2.putText(frame, f"FPS:{int(fps)}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Final', frame)
    result.write(frame)
    # frame_delay += 1

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
