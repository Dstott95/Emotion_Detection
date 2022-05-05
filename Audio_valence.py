# Modality: (01 = full-AV, 02 = video-only, 03 = audio-only)
# Vocal channel (01 = speech, 02 = song)
# 1 - Neutral - 0A, 0V
# 2 - Calm - 0A, 0V
# 3 - Happy - A, 1V
# 4 - Sad - 0A, -1V
# 5 - Angry - 1A, -1V
# 6 - Fearful - 1A, -1V
# 7 - Disgust - 0A, -1V
# 8 - Surprised - 1A, 0V
# Emotional intensity (01 = normal, 02 = strong)
# Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door")
# Repetition (01,02)
# Actor (01 to 24)

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
import os
import glob
start_time = time.time()


def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.figure(1)
    plt.matshow(df_confusion, cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)


class EmotionDataset(Dataset):
    def __init__(self, csv_file, sequence_length):
        data = pd.read_csv(csv_file)
        self.transform = transforms.Compose([transforms.ToTensor()])
        # raw_data = np.genfromtxt(csv_file, delimiter=',', dtype=np.float32)
        self.sequence_length = sequence_length
        self.width = len(data.columns)
        # self.width = len(raw_data[0,:])
        if len(data) % sequence_length != 0:
            # if dataset isn't exactly divisible by sequence length
            # then remove end values to make it divisible
            remainder = len(data) % N_SEQUENCE
            cut_off = len(data) - remainder
            data = data.iloc[:cut_off, :]
        '''if len(raw_data[:,0]) % sequence_length != 0:
            # if dataset isn't exactly divisible by sequence length
            # then remove end values to make it divisible
            remainder = len(raw_data[:,0]) % N_SEQUENCE
            cut_off = len(raw_data[:,0]) - remainder
            raw_data = raw_data[:cut_off,:]'''
        self.data = np.asarray(data)
        # self.data = raw_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        val_label = self.data[index, 0, 1]
        aro_label = self.data[index, 0, 2]
        features = self.data[index, :, 3:]
        labels = [val_label, aro_label]
        return np.asarray(features, dtype=np.float32), np.asarray(labels)

    def reshape(self):
        self.data = np.reshape(self.data, (-1, self.sequence_length, self.width))
        return self.data

class LSTM(nn.Module):

    def __init__(self, n_input, hidden_dim, n_layers, n_sequence, n_output):
        super(LSTM, self).__init__()
        self.n_input = n_input
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_sequence = n_sequence
        self.n_output = n_output
        self.lstm = nn.LSTM(input_size=n_input,
                            hidden_size=hidden_dim,
                            num_layers=n_layers,
                            batch_first=True,
                            dropout=0.2,
                            bidirectional=False)

        # self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(hidden_dim, n_output)

    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(device)
        lstm_out, xt = self.lstm(x, (h0, c0))
        out = self.fc(lstm_out[:, -1, :])
        out = torch.tanh(out)
        return out

    def init_hidden(self):
        return torch.zeros(1, self.hidden_dim)

if __name__ == '__main__':

    torch.manual_seed(1)
    device = torch.device('cuda')

    # Hyperparameters
    n_audio_features = 17
    n_facial_features = 35
    N_INPUTS = n_audio_features  # + n_facial_features
    N_SEQUENCE = 290  # 148800 samples, 1024 frame, 512 hop 148800/512 - 1 = 290
    N_OUTPUTS = 2
    N_EPOCHS = 1000
    N_LAYERS = 2
    BATCH_SIZE = 290
    LEARNING_RATE = 0.0001
    HIDDEN_DIM = 128

    model = LSTM(N_INPUTS, HIDDEN_DIM, N_LAYERS, N_SEQUENCE, N_OUTPUTS).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Load datasets
    train_dataset = EmotionDataset(r"C:\\Users\\darre\\PycharmProjects\\Emotion_Classifier\\audio\\train_data_v+a.csv", sequence_length=N_SEQUENCE)
    train_dataset.reshape()

    test_dataset = EmotionDataset(r"C:\\Users\\darre\\PycharmProjects\\Emotion_Classifier\\audio\\test_data_v+a.csv", sequence_length=N_SEQUENCE)
    test_dataset.reshape()

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               shuffle=True,
                                               batch_size=BATCH_SIZE)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              shuffle=True,
                                              batch_size=BATCH_SIZE)

    # Train the model
    n_total_steps = len(train_loader)
    train_start_time = time.time()
    for epoch in range(N_EPOCHS):
        for i, (data, labels) in enumerate(train_loader):
            # data = data.reshape(-1, N_SEQUENCE, N_INPUTS).to(device)
            data = data.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            percent_complete = (epoch+1)/N_EPOCHS
            frac_left = (1-percent_complete)/percent_complete
            print(f'Epoch [{epoch + 1}/{N_EPOCHS}], Loss: {loss.item():.6f}')
            print("Estimated time left: {:.0f}s".format((time.time() - train_start_time)*frac_left))

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for data, labels in train_loader:
            # data = data.reshape(-1, N_SEQUENCE, N_INPUTS).to(device)
            data = data.to(device)
            labels = labels.to(device)
            outputs = model(data)
            # max returns (value ,index)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy against training: {acc:.3f} %')

    # Test the model
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for data, labels in test_loader:
            data = data.reshape(-1, N_SEQUENCE, N_INPUTS).to(device)
            # data = data.to(device)
            labels = labels.to(device)
            outputs = model(data)
            # max returns (value ,index)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy against test: {acc:.3f} %')
        actual = labels.to(torch.device('cpu'))
        pred = predicted.to(torch.device('cpu'))
        con_mat = confusion_matrix(actual, pred, normalize=None)
        con_mat = pd.DataFrame(con_mat, columns=('Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised'))
        plot_confusion_matrix(con_mat)
        plt.show()

    print(con_mat)
    print("time elapsed: {:.2f}s".format(time.time() - start_time))
