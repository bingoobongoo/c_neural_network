import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms
import numpy as np
import csv
import os

# --- Configuration ---
NUM_THREADS = 4
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
CSV_FILENAME = "results_pytorch.csv"

# --- 1. System & Threading Setup ---
os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
os.environ["MKL_NUM_THREADS"] = str(NUM_THREADS)

# Limit PyTorch internal threads to match the benchmark constraint
torch.set_num_threads(NUM_THREADS)
torch.set_num_interop_threads(1)

device = torch.device("cpu") # Force CPU to match C implementation benchmark
print(f"Using device: {device} with {torch.get_num_threads()} threads")

# --- 2. Data Loading & Preprocessing ---
# Define transform: Convert to Tensor and Normalize (0-1)
# PyTorch ToTensor() moves [0, 255] -> [0.0, 1.0] automatically.
# We also flatten 28x28 -> 784 here to match the Keras 'InputLayer(input_shape=(784,))'
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.flatten(x)) 
])

# Load Fashion MNIST
full_train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Split Training into Train (90%) and Validation (10%)
# Matches Keras: validation_split=0.1 + Shuffle
train_size = int(0.9 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], 
                                          generator=torch.Generator().manual_seed(42))

# Create DataLoaders
# shuffle=True matches the manual shuffling done in the Keras script
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 3. Model Architecture ---
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        
        # Layers (Input 784 -> 1000 -> 500 -> 300 -> 100 -> 10)
        self.fc1 = nn.Linear(784, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 300)
        self.fc4 = nn.Linear(300, 100)
        self.fc5 = nn.Linear(100, 10) # Output (Logits)
        
        self.relu = nn.ReLU()
        
        # --- Weight Initialization ---
        # Matches Keras: 'he_normal' for ReLUs, 'glorot_normal' for Output
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.fc2.bias)
        
        nn.init.kaiming_normal_(self.fc3.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.fc3.bias)
        
        nn.init.kaiming_normal_(self.fc4.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.fc4.bias)
        
        nn.init.xavier_normal_(self.fc5.weight) # Glorot Normal
        nn.init.zeros_(self.fc5.bias)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x) # Return logits (CrossEntropyLoss in PyTorch handles Softmax internally)
        return x

model = NeuralNet().to(device)

# Count Parameters (should match Keras model.summary())
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")

# --- 4. Optimization ---
# PyTorch CrossEntropyLoss includes Softmax + NLLLoss. 
# It expects raw logits, not probabilities.
# It also expects Class Indices (0-9), so we don't need one-hot encoding for y.
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

# --- 5. Training Loop with CSV Logging ---

# Initialize CSV file
with open(CSV_FILENAME, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'time', 'samples/s'])

print("Starting training...")

for epoch in range(EPOCHS):
    epoch_start_time = time.perf_counter()
    
    # --- Training Phase ---
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()       # Zero gradients
        outputs = model(inputs)     # Forward pass
        loss = criterion(outputs, labels) # Calculate loss
        loss.backward()             # Backward pass
        optimizer.step()            # Update weights
        
        # Metrics Calculation
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    epoch_duration = time.perf_counter() - epoch_start_time
    
    # Calculate Average Train Metrics
    epoch_train_loss = running_loss / total_train
    epoch_train_acc = correct_train / total_train
    samples_per_sec = total_train / epoch_duration

    # --- Validation Phase ---
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad(): # Disable gradient calculation
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
            
    epoch_val_loss = val_loss / total_val
    epoch_val_acc = correct_val / total_val

    # --- Logging ---
    print(f"Epoch {epoch+1}/{EPOCHS} "
          f"- loss: {epoch_train_loss:.4f} - acc: {epoch_train_acc:.4f} "
          f"- val_loss: {epoch_val_loss:.4f} - val_acc: {epoch_val_acc:.4f} "
          f"- time: {epoch_duration:.4f}s - samples/s: {samples_per_sec:.2f}")

    with open(CSV_FILENAME, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch + 1,
            f"{epoch_train_loss:.6f}",
            f"{epoch_val_loss:.6f}",
            f"{epoch_train_acc:.6f}",
            f"{epoch_val_acc:.6f}",
            f"{epoch_duration:.6f}",
            f"{samples_per_sec:.6f}"
        ])

# --- 6. Final Evaluation on Test Set ---
print("\n" + "="*40)
print("FINAL TEST SET EVALUATION")
print("="*40)

model.eval()
test_loss = 0.0
correct_test = 0
total_test = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        test_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()

final_test_loss = test_loss / total_test
final_test_acc = correct_test / total_test

print(f"\nFinal Test Loss:     {final_test_loss:.6f}")
print(f"Final Test Accuracy: {final_test_acc:.6f}")