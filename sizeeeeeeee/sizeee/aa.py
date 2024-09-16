import cv2
import time
import numpy as np  # Correct import for numpy

from sizeee.app import model


# Mock read_channel for testing purposes (instead of SPI)
def read_channel(channel):
    # Return a random value for testing purposes (mocked sensor value)
    return 512  # Mocked value for testing

mq135_channel = 0  # AO pin connected to CH0
mq2_channel = 1    # AO pin connected to CH1

# Gender detection function using OpenCV and a pre-trained model
def detect_gender(frame, model):
    face = cv2.resize(frame, (64, 64))  # Resize the frame
    face = np.array(face, dtype="float") / 255.0  # Normalize the image data
    face = np.expand_dims(face, axis=0)  # Expand dimensions for model input

    pred = model.predict(face)[0]  # Predict using the model
    label = "Male" if pred[0] > pred[1] else "Female"  # Determine the label

    return label

# Open the video capture (camera)
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()  # Read frame from the camera
    if not ret:
        break

    # Assume that you have a pre-trained gender detection model
    # You'll need to replace 'model' with the actual loaded model
    label = detect_gender(frame, model)  # Detect gender
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Display label on frame
    cv2.imshow("Gender Detection", frame)  # Show the frame

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()

