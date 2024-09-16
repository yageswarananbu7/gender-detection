import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
def load_data(data_dir, img_size=(64, 64)):
    data, labels = [], []
    for label, gender in enumerate(['man', 'woman']):
        path = os.path.join(data_dir, gender)
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            image = cv2.imread(img_path)
            image = cv2.resize(image, img_size)
            image = img_to_array(image)
            data.append(image)
            labels.append(label)
    data = np.array(data, dtype="float") / 255.0
    labels = to_categorical(labels, num_classes=2)
    return data, labels

train_dir = "C:\\Users\\YAGESWARAN\\Downloads\\train\\train"
test_dir = "C:\\Users\\YAGESWARAN\\Downloads\\test\\test"
X_train, y_train = load_data(train_dir)
X_test, y_test = load_data(test_dir)




def build_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(2, activation='softmax'))  # 2 classes: Male, Female

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


model = build_model()
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# Save the trained model
model.save('gender_detection_model.h5')

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc * 100:.2f}%")