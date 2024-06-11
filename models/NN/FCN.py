import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_size(input_size, conv_layer):
    kernel_size = conv_layer.kernel_size[0]
    dilation = conv_layer.dilation[0]
    stride = conv_layer.stride[0]
    padding = conv_layer.padding[0]
    new_size = int((input_size + 2*padding - dilation*(kernel_size - 1) - 1)/stride + 1)
    return new_size

class 1DCNN(nn.Module):
    def __init__(self, input_size = 1024, filter = 64, kernel_size = 9, n_class = 7):
        torch.manual_seed(28)
        super().__init__()
        self.__type__ = 'cnn'
        self.__arch__ = 'cnn1d'
        self.__version__ = 'baseline'

        #Layer 1
        self.conv1 = nn.Conv1d(in_channels = 1, out_channels = filter, kernel_size = kernel_size, dilation = 3, stride = 2)
        layer1_shape = compute_size(input_size, self.conv1)
        self.norm1 = nn.BatchNorm1d(filter)

        #Layer 2
        self.conv2 = nn.Conv1d(in_channels = filter, out_channels = 2*filter, kernel_size = kernel_size, dilation = 3, stride = 2)
        layer2_shape = compute_size(layer1_shape, self.conv2)
        self.norm2 = nn.BatchNorm1d(2*filter)

        #Layer 3
        self.conv3 = nn.Conv1d(in_channels = 2*filter, out_channels = filter, kernel_size = kernel_size, dilation = 3, stride = 2)
        layer3_shape = compute_size(layer2_shape, self.conv3)
        self.norm3 = nn.BatchNorm1d(filter)

        #Pool
        self.pool = nn.AvgPool1d(kernel_size=3)

        #Activation
        self.actv = nn.ReLU()

        #fully-connected
        flatten_shape = self.conv3.out_channels*(layer3_shape//3)

        self.fc = nn.Linear(flatten_shape, n_class)
        self.dropout = nn.Dropout(0.5)
        
        self.init_weights = False

    def forward(self,x):
        x = self.actv(self.conv1(x))
        x = self.norm1(x)

        x = self.actv(self.conv2(x))
        x = self.norm2(x)

        x = self.actv(self.conv3(x))
        x = self.norm3(x)

        x = self.pool(x)
        x = x.flatten(1)

        x = self.dropout(x)
        x = self.fc(x)
        
        x = F.log_softmax(x,-1)
        return x    
