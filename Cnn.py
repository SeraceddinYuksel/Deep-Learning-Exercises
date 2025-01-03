# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 15:19:28 2025

@author: SERACEDDIN
"""

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D # feature extraction
from tensorflow.keras.layers import Flatten, Dense, Dropout  # Classification
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator #data augmentation

from sklearn.metrics import classification_report

import warnings 
warnings.filterwarnings("ignore")

#♣load dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# Normalization
x_train = x_train.astype("float32")/255
x_test = x_test.astype("float32")/255

# one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test,10)

# %%  Data Augmentation

datagen = ImageDataGenerator(
    rotation_range = 20,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip =True,
    fill_mode ="nearest"
    )
datagen.fit(x_train)

# %% Creat,compile and train model
model = Sequential()

model.add(Conv2D(32, (3,3), padding = "same", activation = "relu", input_shape = x_train.shape[1:]))
model.add(Conv2D(32, (3,3), activation = "relu"))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3,3), padding = "same", activation = "relu"))
model.add(Conv2D(64, (3,3), activation = "relu"))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(512, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

model.summary()

model.compile(optimizer = RMSprop(learning_rate = 0.0001,decay= 1e-6),
              loss = "categorical_crossentropy",
              metrics = ["accuracy"])

history= model.fit(datagen.flow(x_train, y_train , batch_size = 32),
          epochs=100,
          validation_data = (x_test, y_test))
# %%  Test model and evaluate performance

y_pred = model.predict(x_test)
y_pred_class = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis =1)

class_labels = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]
report = classification_report(y_true, y_pred_class, target_names = class_labels)
print(report)

plt.figure()
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label = "Train Accuracy")
plt.plot(history.history["val_accuracy"], label = "Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label = "Train loss")
plt.plot(history.history["val_loss"], label = "Validation loss")
plt.xlabel("Epochs")
plt.ylabel("loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid()
plt.show()



