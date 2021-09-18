import cv2
import math
import numpy as np

# win_size = (1080, 720)
# win_size = (640, 480)
win_size = (1280, 720)
lower_bound_y = 60
rect_bottom_x = 30
lower_bound = (int(win_size[0] / 2) - 280, lower_bound_y)
upper_bound = (win_size[0] - 40, lower_bound_y)
rect_bottom_right = (win_size[0] - rect_bottom_x, win_size[1] - rect_bottom_x)
origin = [0, 0]


def draw_wrist_elbow_shoulder(frame, lms):
    shoulder_x, shoulder_y = lms[11][1], lms[11][2]
    elbow_x, elbow_y = lms[13][1], lms[13][2]
    wrist_x, wrist_y = lms[15][1], lms[15][2]
    # cv2.circle(frame, (shoulder_x, shoulder_y), 5, (0, 0, 255), -1)
    # cv2.circle(frame, (elbow_x, elbow_y), 5, (0, 0, 255), -1)
    cv2.circle(frame, (wrist_x, wrist_y), 5, (0, 0, 255), -1)


def draw_coordinate(frame, cor_x, cor_y):
    shoulder_x, shoulder_y = cor_x, cor_y
    cv2.line(frame, (shoulder_x, shoulder_y), (upper_bound[0] - lower_bound[0], shoulder_y), (0, 220, 0), 2)
    cv2.line(frame, (shoulder_x, shoulder_y), (shoulder_x, 0), (220, 220, 0), 2)


def is_hand_up(lms):
    shoulder_x, shoulder_y = lms[11][1], lms[11][2]
    elbow_x, elbow_y = lms[13][1], lms[13][2]
    wrist_x, wrist_y = lms[15][1], lms[15][2]
    if wrist_y < elbow_y:
        if elbow_y + 30 < shoulder_y:
            check = 'FULL'
        else:
            check = 'HALF'
    else:

        check = 'No operation'
    return check


def frac_x_and_y(lms):
    shoulder_x, shoulder_y = lms[11][1], lms[11][2]
    elbow_x, elbow_y = lms[13][1], lms[13][2]
    wrist_x, wrist_y = lms[15][1], lms[15][2]
    x = wrist_x - shoulder_x
    y = shoulder_y - wrist_y
    x_total_dist = math.dist((shoulder_x, shoulder_y), (win_size[0], shoulder_y))
    y_total_dist = math.dist((shoulder_x, shoulder_y), (shoulder_x, 0))
    return x, y


def check_finger_open(frame, lms, hand_lms):
    wrist_x, wrist_y = lms[15][1], lms[15][2]
    check = False
    if len(hand_lms) != 0:
        if len(hand_lms) == 21:
            l_hand_wrist_x, l_hand_wrist_y = hand_lms[0][1], hand_lms[0][2]
            thrs = math.dist((wrist_x, wrist_y), (l_hand_wrist_x, l_hand_wrist_y))
            if thrs < 150:
                check = True
            else:
                check = False

        elif len(hand_lms) == 42:
            l_hand_wrist_x, l_hand_wrist_y = hand_lms[0][1], hand_lms[0][2]
            r_hand_wrist_x, r_hand_wrist_y = hand_lms[21][1], hand_lms[21][2]
            thrs1 = math.dist((wrist_x, wrist_y), (l_hand_wrist_x, l_hand_wrist_y))
            thrs2 = math.dist((wrist_x, wrist_y), (r_hand_wrist_x, r_hand_wrist_y))
            if thrs1 < thrs2:
                if thrs1 < 150:
                    check = True
                else:
                    check = False
            else:
                if thrs2 < 150:
                    check = True
                else:
                    check = False
        else:
            check = False
    else:
        check = False

    return check


def wrist_depth(frame, depth, lms):
    # Depth calculation for wrist
    wrist_x, wrist_y = lms[15][1], lms[15][2]
    roi_depth = depth[lower_bound[1]:rect_bottom_right[1], lower_bound[0]:upper_bound[0]]
    cv2.imshow('Depth', roi_depth)
    # print(roi_depth.shape)
    distt = roi_depth[wrist_x, wrist_y]
    # print(dist)
    cv2.putText(frame, f"{distt / 10}", (wrist_x, wrist_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    return ang + 360 if ang < 0 else ang


def check_thumpsUp_or_Down(hand_lms):
    x1, y1 = hand_lms[0][1], hand_lms[0][2]
    x2, y2 = hand_lms[5][1], hand_lms[5][2]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    index_finger_dist = math.dist((cx, cy), (hand_lms[8][1], hand_lms[8][2])) - math.dist((cx, cy), (
    hand_lms[7][1], hand_lms[7][2]))
    middle_finger_dist = math.dist((cx, cy), (hand_lms[12][1], hand_lms[12][2])) - math.dist((cx, cy), (
    hand_lms[11][1], hand_lms[11][2]))
    ring_finger_dist = math.dist((cx, cy), (hand_lms[16][1], hand_lms[16][2])) - math.dist((cx, cy), (
    hand_lms[15][1], hand_lms[15][2]))
    pinky_finger_dist = math.dist((cx, cy), (hand_lms[20][1], hand_lms[20][2])) - math.dist((cx, cy), (
    hand_lms[19][1], hand_lms[19][2]))

    # index_finger_points = (hand_lms[5][1], hand_lms[5][2]), (hand_lms[6][1], hand_lms[6][2]), (hand_lms[6][1], hand_lms[6][2])
    index_finger_angle = getAngle((hand_lms[5][1], hand_lms[5][2]), (hand_lms[6][1], hand_lms[6][2]),
                                  (hand_lms[8][1], hand_lms[8][2]))
    middle_finger_angle = getAngle((hand_lms[9][1], hand_lms[9][2]), (hand_lms[10][1], hand_lms[10][2]),
                                   (hand_lms[12][1], hand_lms[12][2]))
    ring_finger_angle = getAngle((hand_lms[13][1], hand_lms[13][2]), (hand_lms[14][1], hand_lms[14][2]),
                                 (hand_lms[16][1], hand_lms[16][2]))
    pinky_finger_angle = getAngle((hand_lms[17][1], hand_lms[17][2]), (hand_lms[18][1], hand_lms[18][2]),
                                  (hand_lms[20][1], hand_lms[20][2]))

    angles = (index_finger_angle, middle_finger_angle, ring_finger_angle, pinky_finger_angle)
    countt = [i for i in angles if i < 90]

    if index_finger_dist < 0 and middle_finger_dist < 0 and ring_finger_dist < 0 and pinky_finger_dist < 0:
        if hand_lms[4][2] == np.min(
                [hand_lms[4][2], hand_lms[8][2], hand_lms[12][2], hand_lms[16][2], hand_lms[20][2]]):
            if (len(countt) > 1 and hand_lms[4][2] < hand_lms[3][2]):
                return "UP"
        elif hand_lms[4][2] == np.max(
                [hand_lms[4][2], hand_lms[8][2], hand_lms[12][2], hand_lms[16][2], hand_lms[20][2]]):
            if (hand_lms[4][2] > hand_lms[3][2]) or (len(countt) > 1 and hand_lms[4][2] >= hand_lms[2][2]):
                return "DOWN"
    else:
        return None
