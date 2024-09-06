import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_size(input_size, conv_layer):
    kernel_size = conv_layer.kernel_size[0]
    dilation = conv_layer.dilation[0]
    stride = conv_layer.stride[0]
    padding = conv_layer.padding[0]
    new_size = int((input_size + 2*padding - dilation*(kernel_size - 1) - 1)//stride + 1)
    return new_size

class ResNetBlock(nn.Module):
    def __init__(self, input_size, in_channels, filter, kernel):
        super().__init__()
        self.__type__ = 'cnn'
        
        self.conv1 = nn.Conv1d(in_channels = in_channels, out_channels = filter, kernel_size = kernel, padding= kernel//2)
        out_size = compute_size(input_size, self.conv1)
        self.conv2 = nn.Conv1d(in_channels = filter, out_channels = filter, kernel_size = kernel, padding= kernel//2)
        out_size = compute_size(out_size, self.conv2)
        self.conv3 = nn.Conv1d(in_channels = filter, out_channels = filter, kernel_size = kernel, padding= kernel//2)
        out_size = compute_size(out_size, self.conv3)
        # self.norm = nn.BatchNorm1d(filter)
        
        self.norm = nn.LayerNorm([filter, input_size]) #=> More stable
        self.actv = nn.ReLU()
        self.init_weights = False
        self.out_size = out_size
    def forward(self,x):
        residual = self.conv1(x)
        x = self.actv(residual)
        x = self.actv(self.conv2(x))
        x = self.actv(self.conv3(x))
        x = x + residual
        x = self.norm(x)
        return x
    
class ResNet(nn.Module):
    def __init__(self, input_size, n_class = 7):
        torch.manual_seed(28)
        super().__init__()
        self.__type__ = 'cnn'
        self.__arch__ = 'ResNet'
        self.__version__ = 'baseline'

        self.conv = nn.Conv1d(1, 64, 15)
        self.pool0 = nn.MaxPool1d(kernel_size = 2)
        out_size = compute_size(input_size, self.conv) // 2
        self.block1 = ResNetBlock(out_size, 64, 64, 5)
        self.pool1 = nn.MaxPool1d(kernel_size = 2)
        out_size = out_size // 2
        self.block2 = ResNetBlock(252, 64, 128, 5)
        self.pool2 = nn.MaxPool1d(kernel_size = 2)
        out_size = out_size // 2
        self.block3 = ResNetBlock(126, 128, 128, 5)
        self.final_pool = nn.MaxPool1d(kernel_size = 2)
        out_size = out_size // 2
        self.fc = nn.Linear(128*out_size, n_class)
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
