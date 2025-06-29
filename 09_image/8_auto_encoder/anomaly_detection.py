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

pyplot.grid()
pyplot.plot(numpy.arange(140), normal_train_data[0])
pyplot.title("A Normal ECG")
pyplot.show()

pyplot.grid()
pyplot.plot(numpy.arange(140), anomalous_train_data[0])
pyplot.title("An Anomalous ECG")
pyplot.show()
