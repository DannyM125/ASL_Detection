import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt") #Using model and class labels
labels = ["A", "B", "C"]
imgSize = 300
offset = 20

folder = "imageData/C"
count = 0
while True:
    success, img = cap.read()
    imgOutput = img.copy()
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
            prediction, index = classifier.getPrediction(imgWhite)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)  # calculated width
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)  # creating a gap to center img
            imgWhite[hGap:hCal + hGap,:] = imgResize
            # Put imgCrop inside imgWhite to overlay img onto a set size background
            prediction, index = classifier.getPrediction(imgWhite)

        cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                      (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 4)


        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("image" , imgOutput)
    key = cv2.waitKey(1)
