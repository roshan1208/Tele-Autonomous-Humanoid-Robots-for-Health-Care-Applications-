import cv2
import mediapipe as mp
import time


class HandDetection:
    def __init__(self, static_image_mode=False,
                 max_num_hands=2,
                 min_detection_confidence=0.7,
                 min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils

    def detectHand(self, frame,new_img, draw=True):
        '''

        :param frame: image frame
        :param draw: bool value true will draw landmark
        :return: return lanmark detected frame
        '''
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(frameRGB)
        if self.results.multi_hand_landmarks:
            for handLMs in self.results.multi_hand_landmarks:
                if draw:
                    # self.mpDraw.draw_landmarks(frame, handLMs, self.mpHands.HAND_CONNECTIONS)
                    self.mpDraw.draw_landmarks(new_img, handLMs, self.mpHands.HAND_CONNECTIONS)
        return new_img, frame

    def handLandmark(self, frame, draw=True):
        '''

        :param frame: image frame
        :param draw: true will draw cirle of the position of all landmark
        :return: retrun list of id and position of each landmark
        '''
        lms = []
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(frameRGB)
        if self.results.multi_hand_landmarks:
            for handLMs in self.results.multi_hand_landmarks:
                for id, lm in enumerate(handLMs.landmark):
                    h, w, c = frame.shape
                    cX, cY = int(lm.x * w), int(lm.y * h)
                    lms.append([id, cX, cY])
                    if draw:
                        cv2.circle(frame, (cX, cY), 5, (255, 0, 255), -1)
        return lms


def main():
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetection()
    while True:
        ret, frame = cap.read()
        # xh, xw, xc = frame.shape
        # frame = cv2.resize(frame, (int(xw / 5), int(xh / 5)))
        frame = detector.detectHand(frame, False)
        lms = detector.handLandmark(frame)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(frame, f"FPS:{int(fps)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)
        cv2.imshow('Image', frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
