# ASL Detection
The objective of this project is to detect and recognize American Sign Language (ASL) signs in real time using a webcam. This is accomplished using computer vision and machine learning techniques, with the help of libraries such as Numpy, TensorFlow, MediaPipe, and OpenCV.

During runtime, the webcam is used to capture live video data, which is processed by the OpenCV library. OpenCV provides the necessary tools to preprocess and analyze video data, enabling the detection and recognition of ASL signs in real-time. Once a sign is detected, the corresponding letter is output to the screen.

The object training process is carried out using the Teachable Machine image model maker, which is a web-based tool for creating custom machine learning models. My model was trained to identify ASL signs by using a dataset of images representing the signs A, B, and C.

## NOTES:

I trained my OWN model! I put a TensorFlow Keras model and its labels in a folder named "Model" in my actual project. Also, keep in mind the algorithm for saving files to certain folders with the corresponding letter in the data collection stage was made for Windows, therefore needs some tweaking to work on other OS.

Train your own model to improve its accuracy and versatility.



https://github.com/DannyM125/ASL_Detection/assets/123900134/b0962ced-2176-4059-abe8-30ee0e5f2e90


# Using CVZONE
Using the CVZONE library allows us to create bounding boxes around the detected areas. This library also includes many other conveniences that would allow for more efficient and reliable data collection.

![Untitled](https://github.com/DannyM125/ASL_Detection/assets/123900134/1df6edd1-524f-4df9-b8f7-5488bf18f846)
![Untitled2](https://github.com/DannyM125/ASL_Detection/assets/123900134/d0d73dd8-7d53-45e7-9386-6f8fba8479be)
![Untitled3](https://github.com/DannyM125/ASL_Detection/assets/123900134/1578e31a-5ad2-4987-acad-06a67908e480)


![image4](https://github.com/DannyM125/ASL_Detection/assets/123900134/d0caeca4-79f9-4c51-a195-2aeeae33fd8b)
![image5](https://github.com/DannyM125/ASL_Detection/assets/123900134/aa3951a2-fb2a-4d93-83b5-ac426230fc10)
![image6](https://github.com/DannyM125/ASL_Detection/assets/123900134/80557100-5ee5-4081-9269-a8599a3b80ca)
