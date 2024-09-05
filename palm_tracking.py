import cv2
import mediapipe as mp

# capture video input 
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error:Couldn't open the webcam")
    exit()

# start the hand tracking module from mediapipe 
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands = 1, min_detection_confidence = 0.7)
mp_draw = mp.solutions.drawing_utils 


while True:
    ret, frame = cap.read()
    if not ret:
        print("Error:Fail to capture image")
        break
    
    # convert the frames to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # process the frame to detect hands 
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # drawing landmarks in the frame 
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)


    # display the frame 
    cv2.imshow('Webcam Feed', frame)

    # press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()