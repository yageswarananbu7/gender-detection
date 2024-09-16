import cv2
import numpy as np
import torch
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import pandas as pd  # Importing pandas to fix the error

# Load YOLOv5 model
model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Pre-trained YOLOv5 model

# Load the gender detection model
model_gender = load_model('gender_detection_model.h5')

def preprocess_image(image, target_size=(64, 64)):
    image = cv2.resize(image, target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

def predict_gender(image):
    processed_image = preprocess_image(image)
    prediction = model_gender.predict(processed_image)
    gender = np.argmax(prediction, axis=1)[0]
    return 'woman' if gender == 1 else 'man'

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam or provide a video file path

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection with YOLOv5
    results = model_yolo(frame)

    # Extract results (Pandas DataFrame format)
    detections = results.pandas().xyxy[0]

    # Filter detections for persons
    person_detections = detections[detections['name'] == 'person']

    for index, row in person_detections.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        person_image = frame[y1:y2, x1:x2]

        if person_image.size != 0:
            # Predict gender
            gender = predict_gender(person_image)

            # Draw bounding box and label
            label = f'Person: {gender}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame with bounding boxes and labels
    cv2.imshow('Real-Time Gender Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
