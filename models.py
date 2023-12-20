import torch.nn as nn

class MLP(nn.Module):
    def __init__(self,input_size,n_classes,hid_1 = 128,hid_2 = 64):
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(input_size,hid_1)
        self.fc2 = nn.Linear(hid_1,hid_2)
        self.fc3 = nn.Linear(hid_2,n_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    def forward(self,x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class CNN2D(nn.Module):
    def __init__(self):
        super(CNN2D,self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride = 2)

        self.conv1 = nn.Conv2d(in_channels = 1,
                               out_channels = 16,
                               kernel_size = 3,
                               stride = 1)
        
        self.conv2 = nn.Conv2d(in_channels = 16,
                               out_channels = 16,
                               kernel_size = 3,
                               stride = 1)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(16*14*14,256)
        self.fc2 = nn.Linear(256,7)
        
    def forward(self,x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1,16*14*14)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D,self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels = 1,
                               out_channels = 16,
                               kernel_size = 5,
                               stride = 2)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride = 2)
        
        self.conv2 = nn.Conv1d(in_channels = 16,
                               out_channels = 16,
                               kernel_size = 5,
                               stride = 1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(in_channels = 16,
                               out_channels = 16,
                               kernel_size = 5,
                               stride = 1)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.5)
        self.batchnorm1 = nn.BatchNorm1d(16*60)
        self.fc1 = nn.Linear(16*60,256)
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256,7)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self,x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.pool3(self.relu(self.conv3(x)))
        x = self.batchnorm1(x.view(-1,16*60))
        x = self.batchnorm2(self.relu(self.fc1(x)))
        x = self.dropout(x)
        x = self.softmax(self.fc2(x))
        return x