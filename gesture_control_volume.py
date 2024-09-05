import cv2
import mediapipe as mp
import math
from ctypes import cast, POINTER # use to interfear with systems audio API through pycaw
from comtypes import CLSCTX_ALL # control audio endpoint devices 
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume # pycaw (Python core audio windows) control windows audio devices using python 

class HandTracker:
    def __init__(self):
        # initialize hand tracking module 
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils

        # custom colors for the drawing lines 
        self.hand_landmarks_style = self.mpDraw.DrawingSpec(color =( 0, 0, 0)) 
        self.hand_connection_style = self.mpDraw.DrawingSpec(color =( 0, 255, 255), thickness = 2) 

        # start the webcam
        self.cap = cv2.VideoCapture(0)

        # initialize volume control
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = cast(interface, POINTER(IAudioEndpointVolume))
        self.minVol = self.volume.GetVolumeRange()[0]
        self.maxVol = self.volume.GetVolumeRange()[1]

    def process_frame(self, img):
        # convert BRG into RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)

        # Drawing hand landmarks
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, 
                                           handLms, 
                                           self.mpHands.HAND_CONNECTIONS,
                                           self.hand_landmarks_style,
                                           self.hand_connection_style
                                        )
                lmList =[]
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape 
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append((id, cx, cy))
                    cv2.circle(img, (cx, cy), 10, (0, 255 , 255))

                if lmList:
                    # coordinates for the thumb tip and the index finger tip
                    thumb_tip = lmList[4][1:3]
                    index_tip = lmList[8][1:3]

                    # Drawing the line between the two finger tips 
                    cv2.line(img, thumb_tip, index_tip, (0, 255, 255), 2)

                    # calculate the distance between the index and the thumb
                    length = math.hypot(index_tip[0] - thumb_tip[0], index_tip[1] - thumb_tip[1])

                    # convert the distance into volume level
                    vol = self.minVol + (self.maxVol - self.minVol) * (length / 200)  # Assuming max distance as 200 pixels
                    vol = min(max(vol, self.minVol), self.maxVol)
                    self.volume.SetMasterVolumeLevel(vol, None)

                    # circle at the midpoint 
                    midX, midY = (thumb_tip[0] + index_tip[0]) // 2, (thumb_tip[1] + index_tip[1]) // 2
                    cv2.circle(img, (midX, midY), 15, (255, 0, 210), cv2.FILLED)

        return img
    
    def run(self):
        while True:
            # capture frames from webcam 
            success, img = self.cap.read()
            if not success:
                break

            # process the frames
            img = self.process_frame(img)

            # display
            cv2.imshow("Image", img)

            # if q is pressed exit 
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # release the webcam and close the window 
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    hand_tracker = HandTracker()
    hand_tracker.run()

