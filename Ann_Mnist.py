
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense

from keras.models import load_model


import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

(x_train, y_train), (x_test, y_test) = mnist.load_data()


plt.figure(figsize=(10,5))
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(x_train[i], cmap = "gray")
    plt.title(f"Index:{i}, Label: {y_train[i]}")
    plt.axis("off") 
plt.show()

x_train = x_train.reshape((x_train.shape[0],x_train.shape[1] * x_train.shape[2])).astype("float32")/255
x_test = x_test.reshape((x_test.shape[0],x_test.shape[1] * x_test.shape[2])).astype("float32")/255

y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)

model = Sequential()
model.add(Dense(512, activation= "relu", input_shape = (28*28,)))

model.add(Dense(256, activation= "tanh"))

model.add( Dense(10, activation = "softmax"))
model.summary()


model.compile(optimizer="adam",
              loss = "categorical_crossentropy",
              metrics = ["accuracy"])

early_stopping = EarlyStopping(monitor = "val_loss", patience =5, restore_best_weights=True)

checkpoint = ModelCheckpoint("ann_best_model.keras", monitor = "val_loss", save_best_only= True)

history = model.fit (x_train, y_train,
           epochs =10,
           batch_size =60,
           validation_split=0.2,
           callbacks=[early_stopping, checkpoint])

test_loss, test_acc = model.evaluate(x_test,y_test)
print (f"Test acc: {test_acc}, test_loss:{test_loss}")

plt.figure()
plt.plot(history.history["accuracy"], marker= "o", label = "Training Acuuracy")
plt.plot(history.history["val_accuracy"], marker= "o", label = "Validation Acuuracy")
plt.title ("ANN Acurarcy on MNIST Data Set")
plt.xlabel("Epochs")
plt.ylabel("Acc")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(history.history["loss"], marker= "o", label = "Training Loss")
plt.plot(history.history["val_loss"], marker= "o", label = "Validation Loss")
plt.title ("ANN loss on MNIST Data Set")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

model.save("final_mnist_ann_model.keras")

loaded_model = load_model("final_mnist_ann_model.keras")

test_loss, test_acc = loaded_model.evaluate(x_test,y_test)
print(f"Loaded Model result -> Test acc:{test_acc},test loss: {test_loss}")
