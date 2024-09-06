import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size = 513, n_class = 7, hid_1 = 256, hid_2 = 128):
        torch.manual_seed(28)
        super().__init__()
        self.input_size = input_size
        self.__type__ = 'mlp'
        self.__arch__ = 'mlp'
        self.__version__ = 'baseline'
        self.fc1 = nn.Linear(input_size,hid_1)
        self.fc2 = nn.Linear(hid_1,hid_2)
        self.fc3 = nn.Linear(hid_2, n_class)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.3)
        self.init_weights = False

    def forward(self,x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        x = nn.functional.log_softmax(x,-1)
        return x