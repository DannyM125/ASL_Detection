import cv2
import mediapipe as mp
import time
import os

cap = cv2.VideoCapture(0) # capture video

mp_hands = mp.solutions.hands
hands = mp_hands.Hands() # default params
mp_drawing = mp.solutions.drawing_utils

# Custom drawing specifications
hand_landmarks_style = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=5) # BGR
hand_connections_style = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2) # BGR

""" # fps stuff:
current_time = 0
previous_time = 0 """

directory = "imageData/C"
while True:
    success, img = cap.read() # read each frame as an image
    img = img[50:650, 200:700] # crop img
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert BGR to RGB because mediapipe wants RGB
    results = hands.process(imgRGB) # img with mediapipe overlay
    
    if results.multi_hand_landmarks: # if there is a hand on screen
        for hand_landmarks in results.multi_hand_landmarks: # for each hand landmark on screen
            mp_drawing.draw_landmarks(
                img, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS, 
                hand_landmarks_style,
                hand_connections_style
            ) # draw that landmark on the img, then connect the landmarks
            
            for id, landmark in enumerate(hand_landmarks.landmark):
                height, width, channels = img.shape  # get features of the img
                # x and y are normalized to [0.0, 1.0] by the image width and height respectively
                center_x = int(landmark.x * width)
                center_y = int(landmark.y * height)
                if id % 4 == 0 and id != 0: # which landmarks to print based on ID
                    cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), cv2.FILLED)
                    
    """ # fps stuff:
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (20, 20, 255), 3) # overlay fps onto img"""
    cv2.imshow("Image", img)

    key = cv2.waitKey(1)
    if key == ord('s'): # if the s key is pressed
        img_name = f"{directory}/img_{int(time.time())}.jpg" # generate unique image name
        cv2.imwrite(img_name, img) # save the image
        print(f"Image saved as {img_name}")

    elif key == ord('q'): # if the q key is pressed
        cap.release() # release the video capture object
        cv2.destroyAllWindows() # close all OpenCV windows
        break # exit the loop
