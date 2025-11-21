import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import keras
from keras import layers, models

num_classes = 10

# -----------------------------
# Load CIFAR-10
# -----------------------------
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test  = keras.utils.to_categorical(y_test,  num_classes)

# -----------------------------
# Model: mirrors your C layout
# -----------------------------
inputs = layers.Input(shape=(32, 32, 3))

# Conv2D input layer is implicit in Keras via Input

# ---- Block 1 ----
x = layers.Conv2D(
    filters=32,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding="valid",          # most likely what you want in C
    kernel_initializer="he_normal",
)(inputs)
# x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)

# ---- Block 2 ----
x = layers.Conv2D(
    filters=64,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding="valid",
    kernel_initializer="he_normal",
)(x)
# x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)

# ---- Flatten ----
x = layers.Flatten()(x)

# ---- Dense + BN ----
x = layers.Dense(
    128,
    kernel_initializer="he_normal",
)(x)
# x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)

# ---- Output ----
outputs = layers.Dense(
    num_classes,
    activation="softmax",
    kernel_initializer=keras.initializers.GlorotNormal(),
    bias_initializer="zeros",
)(x)

model = models.Model(inputs, outputs)

# -----------------------------
# Compile & train
# -----------------------------
model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.0),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

history = model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=10,
    validation_split=0.1,
    verbose=1,
)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")