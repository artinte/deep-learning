import numpy
import pandas
import torch
import sklearn.model_selection
from matplotlib import pyplot


url = r"https://github.com/artinte/tiny-datasets/raw/refs/heads/develop/ecg.csv"

# Download the dataset
dataframe = pandas.read_csv(url)
raw_data = dataframe.values
dataframe.head()

print(raw_data.shape)

# The last element contains the labels
labels = raw_data[:, -1]
# The other data points are the electrocadriogram data
data = raw_data[:, 0:-1]

train_data, test_data, train_labels, test_labels = (
    sklearn.model_selection.train_test_split(
        data, labels, test_size=0.2, random_state=21
    )
)

train_data = torch.from_numpy(train_data)
test_data = torch.from_numpy(test_data)

# normalize the data to [0, 1]
min_val = torch.min(train_data)
max_val = torch.max(train_data)

train_data = (train_data - min_val) / (max_val - min_val)
test_data = (test_data - min_val) / (max_val - min_val)

train_data = train_data.to(torch.float32)
test_data = test_data.to(torch.float32)

train_labels = train_labels.astype(bool)
test_labels = test_labels.astype(bool)

normal_train_data = train_data[train_labels]
normal_test_data = test_data[test_labels]

anomalous_train_data = train_data[~train_labels]
anomalous_test_data = test_data[~test_labels]

train_labels = torch.tensor(train_labels)
test_labels = torch.tensor(test_labels)

train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False)

input_dims = 140

pyplot.grid()
pyplot.plot(numpy.arange(input_dims), normal_train_data[0])
pyplot.title("A Normal ECG")
pyplot.show()

pyplot.grid()
pyplot.plot(numpy.arange(input_dims), anomalous_train_data[0])
pyplot.title("An Anomalous ECG")
pyplot.show()


class AnomalyDetector(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 8),
            torch.nn.ReLU(),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 140),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

autoencoder = AnomalyDetector(input_dims).to(device)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
criterion = torch.nn.L1Loss()

train_history = []
val_history = []
for epoch in range(10):
    autoencoder.train()
    total_loss = 0
    for data, labels in train_loader:
        data = data.to(device)
        outputs = autoencoder.forward(data)
        loss = criterion(outputs, data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
    autoencoder.eval()
    val_loss = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            outputs = autoencoder.forward(data)
            loss = criterion(outputs, data)
            val_loss += loss.item()

    print(f"Epoch [{epoch+1}/10], train loss: {total_loss / len(train_loader):.4f}, val loss: {val_loss / len(test_loader):.4f}")


encoded_data = autoencoder.encoder(normal_test_data.to(device))
decoded_data = autoencoder.decoder(encoded_data).detach().cpu().numpy()

pyplot.plot(normal_test_data[0], 'b')
pyplot.plot(decoded_data[0], 'r')
pyplot.fill_between(numpy.arange(140), decoded_data[0], normal_test_data[0], color='lightcoral')
pyplot.legend(labels=['Input', 'Reconstruction', 'Error'])
pyplot.show()

encoded_data = autoencoder.encoder(anomalous_test_data.to(device))
decoded_data = autoencoder.decoder(encoded_data).detach().cpu().numpy()

pyplot.plot(anomalous_test_data[0], 'b')
pyplot.plot(decoded_data[0], 'r')
pyplot.fill_between(numpy.arange(140), decoded_data[0], anomalous_test_data[0], color='lightcoral')
pyplot.legend(labels=["Input", "Reconstruction", "Error"])
pyplot.show()
