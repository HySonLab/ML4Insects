import torch
import torch.nn as nn
import torch.nn.functional as F


# Util function for computing the output size of a convolutional neuron
def compute_size(input_size, kernel_size, dilation = 1, stride = 1, padding = 'valid'):
    if padding == 'valid':
        pad_size = 0
    elif padding == 'same':
        pad_size = kernel_size//2
    new_size = int((input_size + 2*pad_size - dilation*(kernel_size - 1) - 1)/stride + 1)
    return new_size

class CNN2D(nn.Module):
    def __init__(self, n_class = 7):
        torch.manual_seed(28)
        super(CNN2D,self).__init__()
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
    
    
class CNN1D(nn.Module):
    def __init__(self, input_size = 1024, filter = 64, kernel_size = 9, n_class = 7):
        torch.manual_seed(28)
        super(CNN1D, self).__init__()
        self.__type__ = 'cnn'
        self.__arch__ = 'cnn1d'
        self.__version__ = 'baseline'
        self.conv1 = nn.Conv1d(in_channels = 1, out_channels = filter, kernel_size = kernel_size, dilation = 3, stride = 2)
        self.conv2 = nn.Conv1d(in_channels = filter, out_channels = filter, kernel_size = kernel_size - 2, dilation = 3, stride = 2)
        self.conv3 = nn.Conv1d(in_channels = filter, out_channels = filter, kernel_size = kernel_size - 4, dilation = 3, stride = 2)
        conv_shape1 = compute_size(input_size, kernel_size, dilation = 3, stride = 2)
        self.layernorm1 = nn.LayerNorm(normalized_shape=[filter,conv_shape1])
        conv_shape2 = compute_size(conv_shape1, kernel_size - 2, dilation = 3, stride = 2)
        self.layernorm2 = nn.LayerNorm(normalized_shape=[filter,conv_shape2])
        conv_shape3 = compute_size(conv_shape2, kernel_size - 4, dilation = 3, stride = 2)
        self.layernorm3 = nn.LayerNorm(normalized_shape=[filter,conv_shape3])
        self.pool = nn.MaxPool1d(kernel_size=3)
        flatten_shape = filter*(conv_shape3//3)
        self.fc = nn.Linear(flatten_shape, n_class)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.init_weights = False

    def forward(self,x):
        x = self.relu(self.conv1(x))
        x = self.layernorm1(x)
        x = self.relu(self.conv2(x))
        x = self.layernorm2(x)
        x = self.relu(self.conv3(x))
        x = self.layernorm3(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        x = F.log_softmax(x,-1)
        return x    

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, filter, kernel):
        super(ResNetBlock, self).__init__()
        self.__type__ = 'cnn'
        self.conv1 = nn.Conv1d(in_channels = in_channels, out_channels = filter, kernel_size = kernel, padding= 'same')
        self.conv2 = nn.Conv1d(in_channels = filter, out_channels = filter, kernel_size = kernel, padding= 'same')
        self.conv3 = nn.Conv1d(in_channels = filter, out_channels = filter, kernel_size = kernel, padding= 'same')
        self.norm = nn.LayerNorm(filter)
        self.relu = nn.ReLU()
        self.init_weights = False
    def forward(self,x):
        residual = self.conv1(x)
        x = self.relu(residual)
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.norm(x.transpose(1,2)).transpose(1,2)
        x = x + residual 
        return x
    
class ResNet(nn.Module):
    def __init__(self):
        torch.manual_seed(28)
        super(ResNet, self).__init__()
        self.__type__ = 'cnn'
        self.__arch__ = 'ResNet'
        self.__version__ = 'baseline'

        self.conv = nn.Conv1d(1,64, 15)
        self.pool0 = nn.MaxPool1d(kernel_size = 2)

        self.block1 = ResNetBlock(64, 64, 5)
        self.pool1 = nn.MaxPool1d(kernel_size = 2)
        self.block2 = ResNetBlock(64, 128, 5)
        self.pool2 = nn.MaxPool1d(kernel_size = 2)
        self.block3 = ResNetBlock(128, 128, 5)
        self.final_pool = nn.MaxPool1d(kernel_size = 2)

        self.fc = nn.Linear(128*63, 7)
        self.dropout = nn.Dropout(0.3)
        self.init_weights = True
        self.init_weight()

    def init_weight(self):
        blocks = [self.block1, self.block2, self.block3]
        for i in range(3):
            for layer in blocks[i].children():
                if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv1d):
                    layer.weight = nn.init.kaiming_normal_(layer.weight, nonlinearity= 'relu')
                    layer.bias = nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.BatchNorm1d):
                    layer.weight.data.fill_(1.0)
                    layer.bias.data.fill_(0.0)
        self.fc.weight = nn.init.kaiming_normal_(self.fc.weight, nonlinearity= 'relu')

    def forward(self,x):
        # First conv layer
        x = self.pool0(self.conv(x))

        # Three conv operations
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.block3(x)
        x = self.final_pool(x)
        
        # Flatten and output
        x = x.flatten(1)
        x = self.fc(self.dropout(x))
        x = F.log_softmax(x,-1)
        return x 