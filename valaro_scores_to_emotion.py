import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, utils
from torch.utils.data import Dataset
from sklearn.metrics import confusion_matrix
import time

start_time = time.time()

class Linear(nn.Module):

    def __init__(self, n_input, n_output, n_hidden):
        super(Linear, self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.drop = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = x.to(device)
        hidden = self.fc1(x)
        hidden = self.drop(hidden)
        out = self.fc2(hidden)
        out = F.softmax(out, dim=0)
        return out

class EmotionDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        self.width = len(data.columns)
        self.data = np.asarray(data)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        # 1 is subtracted to bring labels into range 0-7 for PyTorch compatability
        # labels are stored as 1 to 8 in CSV file (in column index 0)
        label = int(self.data[index, 3]-1)
        # column 0 is labels, column 1 onwards is features
        features = self.data[index, 1:3]
        return np.asarray(features,dtype=np.float32), label


# Hyperparametres
device = 'cpu'
n_inputs = 2  # equal to the outputs from val_model i.e. valence and arousal
n_outputs = 8  # equal to the number of emotions to categorise
n_hidden = 32
LEARNING_RATE = 0.00005
BATCH_SIZE = 10
N_EPOCHS = 17
model = Linear(n_inputs, n_outputs, n_hidden)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Load datasets
train_dataset = EmotionDataset(r"C:\\Users\\darre\\PycharmProjects\\Emotion_Classifier\\bimodal_data\\train_val_labels.csv")
test_dataset = EmotionDataset(r"C:\\Users\\darre\\PycharmProjects\\Emotion_Classifier\\bimodal_data\\test_val_labels.csv")


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           shuffle=True,
                                           batch_size=BATCH_SIZE)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          shuffle=True,
                                          batch_size=BATCH_SIZE)

# Initiate lists to hold training accuracies and losses through each epoch
train_losses = []
train_acc = []

# Train the model
def train(epoch):
    model.train()
    n_correct = 0
    n_samples = 0
    running_loss = 0

    for data, labels in train_loader:
        data = data.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    train_loss = running_loss / len(train_loader)
    train_acc.append(acc)
    train_losses.append(train_loss)
    print(f"Train Loss: {train_loss:.3f} | Accuracy: {acc:.3f}")


# Initiate lists to hold training accuracies and losses through each epoch
test_losses = []
test_acc = []


# Test the model
def test(epoch):
    model.eval()  # set model to evaluation

    # Initiate variables
    n_correct = 0
    n_samples = 0
    running_loss = 0

    with torch.no_grad():
        for data, labels in test_loader:
            # data = data.reshape(-1, N_SEQUENCE, N_INPUTS).to(device)
            data = data.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)  # max returns (value ,index)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

    # Calculate losses and accuracies
    test_loss = running_loss / len(test_loader)
    acc = 100.0 * n_correct / n_samples
    test_losses.append(test_loss)
    test_acc.append(acc)

    print(f"Test Loss: {test_loss:.3f} | Accuracy: {acc:.3f}")


# timestamp created for estimating time left
train_start_time = time.time()

# Train and test the model
for epoch in range(N_EPOCHS):
    train(epoch)
    test(epoch)
    if (epoch + 1) % 10 == 0:
        percent_complete = (epoch + 1) / N_EPOCHS
        frac_left = (1 - percent_complete) / percent_complete
        time_left = (time.time() - train_start_time) * frac_left
        print(f"Epoch [{epoch + 1}/{N_EPOCHS}]")
        print(f"Estimated time left: {time_left:.0f}s")


plt.plot(train_acc, "-")
plt.plot(test_acc, "-")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train", "Test"])
plt.title("Train vs Test Accuracy")
plt.savefig("bimodal_class_accuracies.svg")
plt.show()

plt.plot(train_losses, "-")
plt.plot(test_losses, "-")
plt.xlabel("Epoch")
plt.ylabel("Losses")
plt.legend(["Train", "Test"])
plt.title("Train vs Test Losses")
plt.savefig("bimodal_class_losses.svg")
plt.show()

print("time elapsed: {:.2f}s".format(time.time() - start_time))


