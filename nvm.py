import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cv2
import matplotlib.pyplot as plt

DIRECTORY = 'D:/OPEN CV/smart_bin/garbage_classification/train'

CATEGORIES = ['battery', 'biological', 'cardboard', 'clothes', 'glass', 'medicinal', 'metal', 'paper', 'plastic']

data = []


for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        label = CATEGORIES.index(category)
        arr = cv2.imread(img_path)
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        new_arr = cv2.resize(arr, (60, 60))
        data.append([new_arr, label])

# print(data)

import random

random.shuffle(data)
x = []
y = []
for features, label in data:
    x.append(features)
    y.append(label)



x = np.array(x)
y = np.array(y)

x = x/255

x = x.reshape(-1, 60, 60, 3)
print(type(x))
print(type(y))
print(len(y))
print(len(x))
print(x.shape)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

model = Sequential()

model.add(Conv2D(32, (3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(128, (3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(128, (3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(128, input_shape = x.shape[1:], activation = 'relu'))

model.add(Dense(10, activation = 'softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x, y, epochs=25, validation_split=0.1)

model.save('trash2.h5')
