import cv2
import mediapipe as mp
import time
import numpy as np

class PosDetection:
    def __init__(self, static_image_mode=False,
                 model_complexity=1,
                 smooth_landmarks=True,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.myPose = mp.solutions.pose
        self.pose = self.myPose.Pose()
        self.myDraw = mp.solutions.drawing_utils

    def drawPosition(self, frame,new_imgRGB, draw=True):
        # new_img = np.ones(shape=frame.shape, dtype=np.int32)
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # new_imgRGB = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(frameRGB)
        if self.results.pose_landmarks:
            if draw:
                # self.myDraw.draw_landmarks(frame, self.results.pose_landmarks, self.myPose.POSE_CONNECTIONS)
                self.myDraw.draw_landmarks(new_imgRGB, self.results.pose_landmarks, self.myPose.POSE_CONNECTIONS)

        return new_imgRGB,frame

    def findLandmarkloc(self, frame, draw=True):
        lms = []
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(frameRGB)
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = frame.shape
                cX, cY = int(lm.x * w), int(lm.y * h)
                lms.append([id, cX, cY])
                if draw:
                    cv2.circle(frame, (cX, cY), 3, (0, 0, 255), -1)
                    self.myDraw.draw_landmarks(frame, self.results.pose_landmarks, self.myPose.POSE_CONNECTIONS)
        return lms


def main():
    cap = cv2.VideoCapture('video/4.mp4')
    pTime = 0
    detector = PosDetection()
    while True:
        ret, frame = cap.read()
        xh, xw, xc = frame.shape
        frame = cv2.resize(frame, (int(xw / 5), int(xh / 5)))
        frame = detector.drawPosition(frame)
        lis = detector.findLandmarkloc(frame, False)
        print(lis)
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
