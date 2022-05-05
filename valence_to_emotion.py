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

    def __init__(self, n_input, n_output):
        super(Linear, self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.fc = nn.Linear(n_input, n_output)

    def forward(self, x):
        x = x.to(device)
        out = self.fc(x)
        out = F.softmax(out, dim=0)
        return out

class EmotionDataset(Dataset):
    def __init__(self, csv_file, sequence_length):
        data = pd.read_csv(csv_file)
        self.sequence_length = sequence_length
        self.width = len(data.columns)
        self.data = np.asarray(data)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        # 1 is subtracted to bring labels into range 0-7 for PyTorch compatability
        # labels are stored as 1 to 8 in CSV file (in column index 0)
        label = int(self.data[index, 0, 0]-1)
        # column 0 is labels, column 1 onwards is features
        features = self.data[index, :, 3:] #
        return np.asarray(features,dtype=np.float32), label

    def reshape(self):
        # reshapes data into a 3D tensor required for LSTM
        # this splits the data into 'sequences' as required
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
                            dropout=0.3,
                            bidirectional=False)
        self.fc = nn.Linear(hidden_dim, n_output)

    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(device)
        lstm_out, xt = self.lstm(x, (h0, c0))
        out = self.fc(lstm_out[:,-1,:])
        return out

    def init_hidden(self):
        return torch.zeros(1, self.hidden_dim)

if __name__ == '__main__':
    torch.manual_seed(1)
    device = torch.device('cuda')

    # loads previously trained valence and arousal model
    # this model outputs two values: valence and arousal
    n_audio_features = 17
    n_facial_features = 35
    N_INPUTS = n_audio_features + n_facial_features
    N_SEQUENCE = 100  # all files normalised to 100 samples
    N_OUTPUTS = 2
    N_EPOCHS = 100
    N_LAYERS = 2
    BATCH_SIZE = 1
    LEARNING_RATE = 0.00006
    HIDDEN_DIM = 128
    val_model = LSTM(N_INPUTS, HIDDEN_DIM, N_LAYERS, N_SEQUENCE, N_OUTPUTS).to(device)
    val_model.load_state_dict(torch.load(r"C:\\Users\\darre\\PycharmProjects\\Emotion_Classifier\\bimodal_data\\valence\\state_dict.pt"))
    val_model.eval()

    # creates new model to map the valence and arousal outputs to discrete emotions
    n_inputs = 2  # equal to the outputs from val_model i.e. valence and arousal
    n_outputs = 8  # equal to the number of emotions to categorise
    model = Linear(n_inputs, n_outputs).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Load datasets
    train_dataset = EmotionDataset(r"C:\\Users\\darre\\PycharmProjects\\Emotion_Classifier\\bimodal_data\\valence\\train.csv", sequence_length=N_SEQUENCE)
    train_dataset.reshape()

    test_dataset = EmotionDataset(r"C:\\Users\\darre\\PycharmProjects\\Emotion_Classifier\\bimodal_data\\valence\\test.csv", sequence_length=N_SEQUENCE)
    test_dataset.reshape()

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               shuffle=True,
                                               batch_size=BATCH_SIZE)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              shuffle=True,
                                              batch_size=BATCH_SIZE)

    # Train the model to use valence and arousal output to classify emotion
    n_total_steps = len(train_loader)
    train_start_time = time.time()
    for epoch in range(N_EPOCHS):
        for i, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.to(device)

            # Forward pass
            val_output = val_model(data)
            outputs = model(val_output)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            percent_complete = (epoch + 1) / N_EPOCHS
            frac_left = (1 - percent_complete) / percent_complete
            time_left = (time.time() - train_start_time) * frac_left
            print(f'Epoch [{epoch + 1}/{N_EPOCHS}], Loss: {loss.item():.4f}')
            print(f"Estimated time left: {time_left:.0f}s")


    # Calculate the accuracy of the model against the training dataset
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for data, labels in train_loader:
            data = data.to(device)
            labels = labels.to(device)
            outputs = model(data)
            # max returns (value ,index)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy against training: {acc} %')

    # Test the model against test dataset
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
        print(f'Accuracy against test: {acc} %')
        actual = labels.to(torch.device('cpu'))
        pred = predicted.to(torch.device('cpu'))
        con_mat = confusion_matrix(actual, pred, normalize=None)
        con_mat = pd.DataFrame(con_mat,
                               columns=('Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised'))
        plot_confusion_matrix(con_mat)
        plt.show()

    print(con_mat)
    print("time elapsed: {:.2f}s".format(time.time() - start_time))


