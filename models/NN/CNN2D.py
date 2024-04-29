import torch
import torch.nn as nn
import torch.nn.functional as F


# Util function for computing the output size of a convolutional neuron
def compute_size(input_size, conv_layer):
    kernel_size = conv_layer.kernel_size[0]
    dilation = conv_layer.dilation[0]
    stride = conv_layer.stride[0]
    padding = conv_layer.padding[0]
    new_size = int((input_size + 2*padding - dilation*(kernel_size - 1) - 1)/stride + 1)
    return new_size

class CNN2D(nn.Module):
    def __init__(self, n_class = 7):
        torch.manual_seed(28)
        super().__init__()
        self.__type__ = 'cnn'
        self.__arch__ = 'cnn2d'
        self.__version__ = 'baseline'
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3, stride = 1)
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, stride = 1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride = 2)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(16*14*14,256)
        self.fc2 = nn.Linear(256, n_class)
        self.init_weights = False

    def forward(self,x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.flatten(1)
        x = self.dropout1(x)
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        x = F.log_softmax(x,-1)
        return x

