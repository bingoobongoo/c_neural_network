import time
import tensorflow as tf
import keras
from keras import Sequential
from keras.api.layers import Dense, Flatten, MaxPooling2D, Conv2D, Input
from keras.api.datasets import fashion_mnist
from keras.api.utils import to_categorical

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

x_train = x_train.reshape(-1, 784)
x_test  = x_test.reshape(-1, 784)
# x_train = x_train.reshape(-1, 28, 28, 1)
# x_test  = x_test.reshape(-1, 28, 28, 1)

y_train = to_categorical(y_train, 10)
y_test  = to_categorical(y_test, 10) 

model = Sequential([
    Dense(600, activation="relu"),
    Dense(200, activation="relu"),
    Dense(10, activation="softmax")
])

opt = keras.optimizers.SGD(learning_rate=0.01)

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

class TimerCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.start = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self.start
        print(f"Epoch {epoch+1} took {elapsed:.3f} seconds")

model.fit(x_train, y_train, epochs=5, batch_size=32, callbacks=[TimerCallback()])
print(model.count_params())