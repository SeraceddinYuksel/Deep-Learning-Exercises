#load dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM,Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import warnings 
warnings.filterwarnings("ignore")


newsgroup = fetch_20newsgroups(subset = "all")
X = newsgroup.data
y = newsgroup.target

#tokenizer
tokenizer = Tokenizer(num_words = 10000)
tokenizer.fit_on_texts(X)
X_sequences = tokenizer.texts_to_sequences(X)
X_padded = pad_sequences(X_sequences,maxlen = 100)

#label encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train Test split
X_train, X_test, y_train, y_test = train_test_split(X_padded, y_encoded, test_size = 0.2, random_state = 42)
# %% create model
def lstm_model():
    model = Sequential()
    model.add(Embedding(input_dim = 10000, output_dim = 64, input_length = 100))
    #lstm layer
    model.add(LSTM(units = 64, return_sequences = False))
    #dropout
    model.add(Dropout(0.5))
    #Dense
    model.add(Dense(20, activation = "softmax"))
    #model compile
    model.compile(optimizer = "adam",
                  loss = "sparse_categorical_crossentropy",
                  metrics = ["accuracy"])
    return model
model = lstm_model()
model.build(input_shape=(None, 100))
model.summary()

# %% train model
early_stopping = EarlyStopping(monitor = "val_accuracy", patience = 5, restore_best_weights = True)

history = model.fit(X_train, y_train,
                            epochs = 5,
                            batch_size = 32,
                            validation_split = 0.1,
                            callbacks = [early_stopping])

# %% model evaluation
#evaluate with test set

loss,accuracy = model.evaluate (X_test, y_test)
print(f"Test loss: {loss: .4f}, Test Accuracy: {accuracy: .4f}")
#visualization
plt.figure()

plt.subplot(1,2,1)
plt.plot(history.history["loss"], label = "Training Loss")
plt.plot(history.history["val_loss"], label = "Validation Loss")
plt.title("Training and validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid("True")


plt.subplot(1,2,2)
plt.plot(history.history["accuracy"], label = "Training Accuracy")
plt.plot(history.history["val_accuracy"], label = "Validation accuracy")
plt.title("Training and validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid("True")




