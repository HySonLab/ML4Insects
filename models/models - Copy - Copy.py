import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt

'''
    Changelog 15.Feb.2024:
        + Add a new function compute_size()
        + Add a new attributes self.__arch__ to all architectures which specifies the architecture
        + Delete unused comments
'''

# Util function for computing the output size of a convolutional neuron
def compute_size(input_size, kernel_size, dilation = 1, stride = 1, padding = 'valid'):
    if padding == 'valid':
        pad_size = 0
    elif padding == 'same':
        pad_size = kernel_size//2
    new_size = int((input_size + 2*pad_size - dilation*(kernel_size - 1) - 1)/stride + 1)
    return new_size

class MLP(nn.Module):
    def __init__(self, input_size = 513, n_class = 7, hid_1 = 256, hid_2 = 128):
        super(MLP,self).__init__()
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
        x = F.log_softmax(x,-1)
        return x
        
class CNN2D(nn.Module):
    def __init__(self, n_class = 7):
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
        torch.manual_seed(28)
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
    

#================== UNDER CONSTRUCTION ============================================================
transition_matrix = torch.tensor([[1/2,1/2,0,0,0,0,0],
                        [1/7,1/7,1/7,1/7,1/7,1/7,1/7],
                        [0,1/3,1/3,1/3,0,0,0],
                        [0,1/3,1/3,1/3,0,0,0],
                        [0,1/2,0,0,1/2,0,0],
                        [0,1/2,0,0,0,1/2,0],
                        [0,1/2,0,0,0,0,1/2]], dtype = torch.float32, device = 'cuda')
    
class MultiScaleCNN(nn.Module):
    def __init__(self, input_size = 1080, filter = 64, kernel_size = 5, n_class = 7):
        torch.manual_seed(28)
        self.__type__ = 'cnn'
        self.__arch__ = 'multiscale_cnn'
        self.__version__ = 'baseline'
        super(MultiScaleCNN, self).__init__()    

        # Identity block
        self.conv1 = ConvBlock(input_size, filter, kernel_size, n_class)

        # Down sampling blocks
        self.ds1 = DownSamplingBlock(input_size, 2, filter, kernel_size, pool_size=3)
        self.ds2 = DownSamplingBlock(input_size, 3, filter, kernel_size, pool_size=3)
        self.ds3 = DownSamplingBlock(input_size, 5, filter, kernel_size, pool_size=3)
        self.ds4 = DownSamplingBlock(input_size, 8, filter, kernel_size, pool_size=3)
        # Moving average blocks
        self.avg1 = SmoothingBlock(input_size, 2, filter, kernel_size, pool_size=3, pool_stride= 3)
        self.avg2 = SmoothingBlock(input_size, 3, filter, kernel_size, pool_size=3, pool_stride= 3)
        self.avg3 = SmoothingBlock(input_size, 4, filter, kernel_size, pool_size=3, pool_stride= 3)       
        self.avg4 = SmoothingBlock(input_size, 5, filter, kernel_size, pool_size=3, pool_stride= 3)      

        # Full convolution stage
        self.conv2 = nn.Conv1d(filter, filter, kernel_size= 7, stride = 7)
        self.pool2 = nn.MaxPool1d(5, 5)
        self.norm = nn.BatchNorm1d(filter)
        self.relu = nn.ReLU()
        self.fc = nn.LazyLinear(7)
        self.dropout = nn.Dropout(0.3)
        
        self.init_weights = True

    def forward(self, x):
        xds1 = self.ds1(x)
        xds2 = self.ds2(x)
        xds3 = self.ds3(x)
        xds4 = self.ds4(x)

        xavg1 = self.avg1(x)
        xavg2 = self.avg2(x)
        xavg3 = self.avg3(x)
        xavg4 = self.avg4(x)
        
        xconv = self.conv1(x)
        #Concat
        x = torch.concat([xds1, xds2, xds3, xds4, xavg1, xavg2, xavg3, xavg4, xconv], dim = -1)
        
        x = self.relu(self.conv2(x))
        x = self.norm(x)
        x = self.pool2(x)
        x = x.flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        x = F.log_softmax(x, -1)
        return x

class DownSample(nn.Module):
    def __init__(self, scale):
        super(DownSample, self).__init__()
        self.scale = scale
    def forward(self, x):
        _, _, length = x.shape
        index = [i*self.scale for i in range(length//self.scale)]
        new = x[:,:,index]
        return new
    
class MovingAverage(nn.Module):
    def __init__(self, size):
        super(MovingAverage, self).__init__()
        self.size = size
    def forward(self, x):
        _, _, length = x.shape
        n = length//self.size
        mean_tensor = []
        for i in range(n):
            mean_tensor+= [torch.mean(x[:,:,i*self.size : (i+1)*self.size ], dim = -1).unsqueeze(-1)]*self.size 
        mean_tensor = torch.concat(mean_tensor, dim = -1)
        x = torch.add(x, torch.add(mean_tensor,-x))
        return x
    
class ConvBlock(nn.Module):
    def __init__(self, input_size, filter, kernel_size, pool_size):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels = 1, out_channels = filter, kernel_size = kernel_size)
        conv_shape = compute_size(input_size, kernel_size)
        self.layernorm1 = nn.LayerNorm(normalized_shape=[filter, conv_shape])
        self.pool = nn.MaxPool1d(pool_size)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.relu(self.conv(x))
        x = self.layernorm1(x)
        x = self.pool(x)
        return x  

class DownSamplingBlock(nn.Module): # Gonna use 3 blocks with scale 8, 5, 3
    def __init__(self, input_size, scale, filter, kernel_size, pool_size, pool_stride = 1):
        super(DownSamplingBlock, self).__init__()
        self.rescale = DownSample(scale)
        self.conv = nn.Conv1d(in_channels = 1, out_channels = filter, kernel_size = kernel_size)
        conv_shape = compute_size(input_size//scale, kernel_size)
        self.layernorm = nn.LayerNorm(normalized_shape=[filter, conv_shape])
        self.pool = nn.MaxPool1d(pool_size, stride = pool_stride)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.rescale(x)
        x = self.relu(self.conv(x))
        x = self.layernorm(x)
        x = self.pool(x)
        return x      

class SmoothingBlock(nn.Module): # Gonna use 3 blocks with scale 8, 5, 3
    def __init__(self, input_size, scale, filter, kernel_size, pool_size, pool_stride = 1):
        super(SmoothingBlock, self).__init__()
        self.smooth = MovingAverage(scale)
        self.conv = nn.Conv1d(in_channels = 1, out_channels = filter, kernel_size = kernel_size)
        conv_shape = compute_size(input_size, kernel_size)
        self.layernorm = nn.LayerNorm(normalized_shape=[filter, conv_shape])
        self.pool = nn.MaxPool1d(pool_size, stride = pool_stride)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.smooth(x)
        x = self.relu(self.conv(x))
        x = self.layernorm(x)
        x = self.pool(x)
        return x     
    

# class MLP_dwt(nn.Module):
#     def __init__(self, input_size = 267, n_class = 7, hid_1 = 256, hid_2 = 128):
#         super(MLP_dwt,self).__init__()
#         self.input_size = input_size
#         self.__type__ = 'mlp'
#         self.__version__ = 'single dropout'
#         self.fc1 = nn.Linear(input_size,hid_1)
#         self.fc2 = nn.Linear(hid_1,hid_2)
#         self.fc3 = nn.Linear(hid_2, n_class)
#         self.relu = nn.ReLU()
#         self.dropout1 = nn.Dropout(0.5)
#         self.dropout2 = nn.Dropout(0.3)
#         self.init_weights = False

#     def forward(self,x):
#         x = self.relu(self.fc1(x))
#         # x = self.dropout1(x)
#         x = self.relu(self.fc2(x))
#         x = self.dropout2(x)
#         x = self.fc3(x)
#         x = F.log_softmax(x,-1)
#         return x