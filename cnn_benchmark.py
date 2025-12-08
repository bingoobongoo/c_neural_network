import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import keras
from keras import layers, models
tf_backend = "keras"

# Ensure channels_last globally (optional; default is channels_last)
keras.backend.set_image_data_format("channels_last")

# ---- Data: Fashion-MNIST, channels-last (N, 28, 28, 1) ----
num_classes = 10
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0
x_train = x_train[..., None]  # (N, 28, 28, 1)
x_test  = x_test[..., None]

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test  = keras.utils.to_categorical(y_test,  num_classes)

# ---- Model (NHWC), matches your spatial sizes/params ----
# Conv1: (28,28,1) --k=8--> (21,21,16)
# MaxPool2D k=2 s=2: (10,10,16)
# Conv2: k=4 s=1 valid: (7,7,4)
# Flatten: 7*7*4 = 196
# Dense(10)

inputs = layers.Input(shape=(28, 28, 1))
#--- Block 1 ---
# add_conv_layer(16, 3, 1, net);
x = layers.Conv2D(
    filters=16, 
    kernel_size=(3, 3), 
    strides=(1, 1), 
    padding="valid",
    kernel_initializer="he_normal",
    use_bias=True 
)(inputs)

# add_batch_norm_conv2D_layer(0.1, net);
# C uses alpha=0.1, so Keras momentum is usually 1 - 0.1 = 0.9
x = layers.BatchNormalization(momentum=0.9)(x)

# Activation (Implicitly RELU from global config)
x = layers.Activation("relu")(x)

# add_max_pool_layer(2, 2, net);
x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)


# --- Block 2 ---
# add_conv_layer(32, 3, 1, net);
x = layers.Conv2D(
    filters=32, 
    kernel_size=(3, 3), 
    strides=(1, 1), 
    padding="valid",
    kernel_initializer="he_normal",
    use_bias=True
)(x)

# add_batch_norm_conv2D_layer(0.1, net);
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.Activation("relu")(x)

# add_max_pool_layer(2, 2, net);
x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)


# --- Flatten ---
# add_flatten_layer(net);
x = layers.Flatten()(x)


# --- Dense Layer ---
# add_dense_layer(128, net);
x = layers.Dense(
    128, 
    kernel_initializer="he_normal", 
    use_bias=True
)(x)

# add_batch_norm_dense_layer(0.1, net);
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.Activation("relu")(x)


# --- Output Layer ---
# add_output_layer(10, net);
outputs = layers.Dense(
    10, 
    activation="softmax",
    kernel_initializer="glorot_uniform", 
    bias_initializer="zeros"
)(x)

model = models.Model(inputs, outputs)
model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.0),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ---- Train & evaluate ----
history = model.fit(
    x_train, y_train,
    batch_size=32, epochs=10, validation_split=0.1, verbose=1
)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test acc: {test_acc:.4f}")
