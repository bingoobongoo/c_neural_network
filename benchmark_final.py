import time
import tensorflow as tf
import numpy as np
import keras
from keras import Sequential
from keras.layers import Dense, InputLayer
from keras.utils import to_categorical
from keras.api.datasets import fashion_mnist, cifar10
import csv
import os

NUM_THREADS = 4

os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(NUM_THREADS)
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

tf.config.threading.set_intra_op_parallelism_threads(NUM_THREADS)
tf.config.threading.set_inter_op_parallelism_threads(1)

# (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# --- 1. Data Loading & Preprocessing ---
# Using CIFAR-10 (matches 3072 input size)
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Normalize
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

# Flatten inputs (Matches C: add_input_layer(x_train->n_cols...))
# 32*32*3 = 3072
x_train = x_train.reshape(-1, 3072)
x_test  = x_test.reshape(-1, 3072)
# x_train = x_train.reshape(-1, 784)
# x_test  = x_test.reshape(-1, 784)

# One-hot encode labels (Matches C: add_output_layer(10...))
y_train = to_categorical(y_train, 10)
y_test  = to_categorical(y_test, 10)

indices = np.arange(x_train.shape[0])
np.random.shuffle(indices)
x_train = x_train[indices]
y_train = y_train[indices]

# --- 2. Model Architecture ---
# Equivalent to:
# add_dense_layer(1000, net) -> He Normal
# ...
# add_output_layer(10, net)  -> Glorot Normal (Softmax)

model = Sequential([
    InputLayer(input_shape=(3072,)),
    
    Dense(1000, activation="relu", 
          kernel_initializer="he_normal", bias_initializer="zeros"),
    Dense(1000, activation="relu", 
          kernel_initializer="he_normal", bias_initializer="zeros"),
    Dense(1000, activation="relu", 
          kernel_initializer="he_normal", bias_initializer="zeros"),
    Dense(1000, activation="relu", 
          kernel_initializer="he_normal", bias_initializer="zeros"),
    Dense(500, activation="relu", 
          kernel_initializer="he_normal", bias_initializer="zeros"),
    Dense(500, activation="relu", 
          kernel_initializer="he_normal", bias_initializer="zeros"),
    Dense(300, activation="relu", 
          kernel_initializer="he_normal", bias_initializer="zeros"),
    
    Dense(100, activation="relu", 
          kernel_initializer="he_normal", bias_initializer="zeros"),
    
    Dense(10, activation="softmax", 
          kernel_initializer="glorot_normal", bias_initializer="zeros")
])

# --- 3. Optimization ---
# optimizer_sgd_new(0.001)
opt = keras.optimizers.Adam(0.0001, 0.9, 0.999)

model.compile(optimizer=opt, 
              loss='categorical_crossentropy', 
              metrics=['accuracy'],
              jit_compile=True)

model.summary()

# --- 4. Custom CSV Logger Callback ---
class CSVAndTimerCallback(tf.keras.callbacks.Callback):
    def __init__(self, filename="results_python.csv"):
        super().__init__()
        self.filename = filename
        self.epoch_start_time = 0
        
        # Create/Overwrite file and write header
        with open(self.filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            # Exact header format matching your C code output
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'time', 'samples/s'])

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.perf_counter() # High precision timer

    def on_epoch_end(self, epoch, logs=None):
        elapsed_time = time.perf_counter() - self.epoch_start_time
        
        # Calculate samples per second
        # Assuming validation_split=0.1, actual training samples is 90% of total x_train
        num_train_samples = x_train.shape[0] * 0.9 
        samples_per_sec = num_train_samples / elapsed_time if elapsed_time > 0 else 0

        # Get metrics
        t_loss = logs.get('loss')
        v_loss = logs.get('val_loss')
        t_acc  = logs.get('accuracy')
        v_acc  = logs.get('val_accuracy')

        # Print to console (optional, for visibility)
        print(f" - time: {elapsed_time:.4f}s - samples/s: {samples_per_sec:.2f}")

        # Write to CSV
        with open(self.filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1, 
                f"{t_loss:.6f}", 
                f"{v_loss:.6f}", 
                f"{t_acc:.6f}", 
                f"{v_acc:.6f}", 
                f"{elapsed_time:.6f}", 
                f"{samples_per_sec:.6f}"
            ])

# --- 5. Training ---
# 20 epochs, batch size 32, 10% validation split
print("Starting training...")
model.fit(
    x_train, y_train, 
    epochs=10, 
    batch_size=32, 
    validation_split=0.1,  # 10% of training data used for validation
    callbacks=[CSVAndTimerCallback("results_python.csv")],
    verbose=1 
)

print(f"Total parameters: {model.count_params()}")

# --- 6. Final Evaluation on Test Set ---
print("\n" + "="*40)
print("FINAL TEST SET EVALUATION")
print("="*40)

# Run evaluation
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)

print(f"\nFinal Test Loss:     {test_loss:.6f}")
print(f"Final Test Accuracy: {test_acc:.6f}")