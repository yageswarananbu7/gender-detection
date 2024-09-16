import cv2
import torch

# Load YOLOv5 model (you can use a custom-trained model or a pre-trained one)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # 'yolov5s' is the small model, you can choose others

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam, or provide a video file path

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    results = model(frame)

    # Extract results
    detections = results.pandas().xyxy[0]  # Pandas DataFrame format

    # Filter detections for persons
    person_detections = detections[detections['name'] == 'person']

    # Draw bounding boxes and count people
    for index, row in person_detections.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the frame with bounding boxes
    cv2.imshow('Person Detection', frame)

    # Display count
    person_count = len(person_detections)
    print(f'Number of people detected: {person_count}')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
