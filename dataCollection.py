import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
imgSize = 300

offset = 20

folder = "imageData/C"
count = 0
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        # 3 - full color, np.uint8 - 8bit value (0-255), *255 to change the values from 1 to 255

        imgCrop = img[y-offset : y+h+offset, x-offset : x+w+offset]
        # Offset insures that the tips of the fingers aren't cut off

        aspectRatio = h/w

        # resizing img if the height is greater than width
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w) # calculated width
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2) # creating a gap to center img
            imgWhite[:, wGap:wCal+wGap] = imgResize
            # Put imgCrop inside imgWhite to overlay img onto a set size background
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)  # calculated width
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)  # creating a gap to center img
            imgWhite[hGap:hCal + hGap,:] = imgResize
            # Put imgCrop inside imgWhite to overlay img onto a set size background


        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("image" , img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        count += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.png', imgWhite)
        print(count)

