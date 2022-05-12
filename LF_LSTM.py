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
        self.sequence_length = sequence_length
        self.width = len(data.columns)
        self.data = np.asarray(data)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        # 1 is subtracted to bring labels into range 0-7 for PyTorch compatability
        # labels are stored as 1 to 8 in CSV file (in column index 0)
        label = int(self.data[index, 0, 0]-1)
        # column 0 contains class labels, column 1 onwards contains features
        audio_features = self.data[index, :, 1:1+n_audio_features]
        facial_features = self.data[index, :, 1+n_audio_features:] #
        return np.asarray(audio_features,dtype=np.float32),\
               np.asarray(facial_features,dtype=np.float32),\
               label

    def reshape(self):
        # reshapes data into a 3D tensor required for LSTM
        # this splits the data into 'sequences' as required
        self.data = np.reshape(self.data, (-1, self.sequence_length, self.width))
        return self.data


class LSTM(nn.Module):

    def __init__(self, n_audio, n_facial, hidden_dim, n_layers, n_sequence, n_output):
        super(LSTM, self).__init__()
        self.n_audio = n_audio
        self.n_facial = n_facial
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_sequence = n_sequence
        self.n_output = n_output
        self.lstm_a = nn.LSTM(input_size=n_audio,
                            hidden_size=hidden_dim,
                            num_layers=n_layers,
                            batch_first=True,
                            dropout=0.3,
                            bidirectional=False)
        self.lstm_f = nn.LSTM(input_size=n_facial,
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              batch_first=True,
                              dropout=0.3,
                              bidirectional=False)

        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(hidden_dim, n_output)
        self.combine = nn.Linear(2*n_output, n_output)

    def forward(self, x_a, x_f):
        # initiate variables
        # a for audio, f for facial
        h0_a = torch.zeros(self.n_layers, x_a.size(0), self.hidden_dim).to(device)
        c0_a = torch.zeros(self.n_layers, x_a.size(0), self.hidden_dim).to(device)
        h0_f = torch.zeros(self.n_layers, x_f.size(0), self.hidden_dim).to(device)
        c0_f = torch.zeros(self.n_layers, x_f.size(0), self.hidden_dim).to(device)

        # LSTM layers
        lstm_out_a, xt = self.lstm_a(x_a, (h0_a, c0_a))
        lstm_out_f, xt = self.lstm_f(x_f, (h0_f, c0_f))

        # linear layers (modes still seperated)
        out_a = self.fc(lstm_out_a[:,-1,:])   # ensure shape is correct
        out_f = self.fc(lstm_out_f[:,-1,:])   # ensure shape is correct

        # Dropout layer
        out_a = self.drop(out_a)  # ensure shape is correct
        out_f = self.drop(out_f)

        # combine results
        out_c = torch.cat((out_a, out_f), 1)
        out = self.combine(out_c)
        return out

    def init_hidden(self):
        return torch.zeros(1, self.hidden_dim)

if __name__ == '__main__':
    torch.manual_seed(1)
    device = torch.device('cuda')

    # Hyperparameters
    n_audio_features = 17
    n_facial_features = 16
    N_SEQUENCE = 50  # all files normalised to 100 samples
    N_OUTPUTS = 8
    N_EPOCHS = 250
    N_LAYERS = 2
    BATCH_SIZE = 10000
    LEARNING_RATE = 0.0001
    HIDDEN_DIM = 256

    model = LSTM(n_audio_features, n_facial_features, HIDDEN_DIM, N_LAYERS, N_SEQUENCE, N_OUTPUTS).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Load datasets
    train_dataset = EmotionDataset(r"C:\\Users\\darre\\PycharmProjects\\Emotion_Classifier\\bimodal_data\\train.csv", sequence_length=N_SEQUENCE)
    train_dataset.reshape()

    test_dataset = EmotionDataset(r"C:\\Users\\darre\\PycharmProjects\\Emotion_Classifier\\bimodal_data\\test.csv", sequence_length=N_SEQUENCE)
    test_dataset.reshape()

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

        for i, sample in enumerate(train_loader):
            data_a, data_f, labels = sample[0].cuda(), sample[1].cuda(), sample[2].cuda()

            # Forward pass
            outputs = model(data_a, data_f)
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
        model.eval() # set model to evaluation

        # Initiate variables
        n_correct = 0
        n_samples = 0
        running_loss = 0

        with torch.no_grad():
            for i, sample in enumerate(test_loader):
                data_a, data_f, labels = sample[0].cuda(), sample[1].cuda(), sample[2].cuda()

                # Forward pass
                outputs = model(data_a, data_f)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)   # max returns (value ,index)
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

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for i, sample in enumerate(test_loader):
            data_a, data_f, labels = sample[0].cuda(), sample[1].cuda(), sample[2].cuda()

            # Forward pass
            outputs = model(data_a, data_f)
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
    #torch.save(model.state_dict(),
               #r"C:\\Users\\darre\\PycharmProjects\\Emotion_Classifier\\bimodal_data\\state_dict_.pt")
    # for name, param in model.named_parameters():
        # print(f"Name: {name}, Parameters: {param}")


