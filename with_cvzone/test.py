import cv2
import mediapipe as mp
import time
import numpy as np
import tensorflow as tf

cap = cv2.VideoCapture(0)  # capture video

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()  # default params
mp_drawing = mp.solutions.drawing_utils

# Load your Keras model
model = tf.keras.models.load_model("Model/keras_model.h5")

# Define your class labels
class_labels = ["A", "B", "C"]  # Add more as needed

# Custom drawing specifications
hand_landmarks_style = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=5)  # BGR
hand_connections_style = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)  # BGR

while True:
    success, img = cap.read()  # read each frame as an image
    img = img[50:650, 200:700]  # crop img
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert BGR to RGB because mediapipe wants RGB
    results = hands.process(imgRGB)  # img with mediapipe overlay

    if results.multi_hand_landmarks:  # if there is a hand on screen
        for hand_landmarks in results.multi_hand_landmarks:  # for each hand landmark on screen
            mp_drawing.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                hand_landmarks_style,
                hand_connections_style
            )  # draw that landmark on the img, then connect the landmarks

            # Preprocess image for the Keras model
            resized_img = cv2.resize(img, (224, 224))
            preprocessed_img = np.expand_dims(resized_img, axis=0) / 255.0

            # Predict using the Keras model
            predictions = model.predict(preprocessed_img)
            predicted_class_index = np.argmax(predictions)
            predicted_class = class_labels[predicted_class_index]

            cv2.putText(img, predicted_class, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Image", img)

    key = cv2.waitKey(1)
    if key == ord('q'):  # if the q key is pressed
        cap.release()  # release the video capture object
        cv2.destroyAllWindows()
        break