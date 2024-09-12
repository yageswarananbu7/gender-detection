import cv2
import numpy as np

from app import model


def detect_gender(frame, model):
    face = cv2.resize(frame, (64, 64))
    face = np.array(face, dtype="float") / 255.0
    face = np.expand_dims(face, axis=0)

    pred = model.predict(face)[0]
    label = "Male" if pred[0] > pred[1] else "Female"

    return label


cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    label = detect_gender(frame, model)
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Gender Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2


def detect_gender(frame, model):
    face = cv2.resize(frame, (64, 64))
    face = np.array(face, dtype="float") / 255.0
    face = np.expand_dims(face, axis=0)

    pred = model.predict(face)[0]
    label = "man" if pred[0] > pred[1] else "woman"

    return label


cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    label = detect_gender(frame, model)
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Gender Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
