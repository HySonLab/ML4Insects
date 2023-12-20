from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import numpy as np

class custom_dataset(Dataset):
    def __init__(self,X,y):
        self.X = X
        self.y = y 
    def __len__(self):
        return len(self.X)
    def __getitem__(self,i):
        return self.X[i], self.y[i]
    
def train_validation_test(X,y,r = 0.2, mode = 'mlp'):
    '''
        Input: 
            X: tuple (data,label) - Data for training and validation
            y: tuple (data,label) - Data for testing
            r: train/validation ratio
        Output:
            train, test, validation split 
    '''
    df,lab = X
    df_test,lab_test = y

    if mode == 'transformer':
        train_idx, val_idx = train_test_split(np.arange(0,len(X[0])),test_size = r)
        x_train, x_validation, y_train, y_validation = X[0][train_idx], X[0][val_idx], X[1][train_idx], X[1][val_idx]

    else:
        x_train, x_validation, y_train, y_validation = train_test_split(df,lab,test_size = r,stratify=lab)

    x_train = torch.from_numpy(x_train).float()
    x_validation = torch.from_numpy(x_validation).float()
    x_test = torch.from_numpy(df_test).float()

    y_train = torch.from_numpy(y_train).long()
    y_validation = torch.from_numpy(y_validation).long()
    y_test = torch.from_numpy(lab_test).long()

    if mode == 'mlp' or mode == 'transformer':
        return x_train, x_validation, x_test, y_train, y_validation, y_test
    
    elif mode == 'cnn2d' or mode == 'cnn1d':
        return x_train.unsqueeze(1), x_validation.unsqueeze(1), x_test.unsqueeze(1), y_train.unsqueeze(1), y_validation.unsqueeze(1), y_test.unsqueeze(1)

from torch.utils.data import DataLoader      

def get_loader(X,y,r=0.2,batch_size = 32,mode = 'mlp'):

    x_train, x_validation, x_test, y_train, y_validation, y_test = train_validation_test(X,y,r,mode)
    
    train_dataset = custom_dataset(x_train,y_train)
    validation_dataset = custom_dataset(x_validation,y_validation)
    test_dataset = custom_dataset(x_test,y_test)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    validation_loader = DataLoader(validation_dataset, batch_size = batch_size, shuffle = False)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

    print(f'Train, validate, test lengths: {x_train.shape[0],x_validation.shape[0],x_test.shape[0]}')
    print(f'Input shape:',next(iter(train_loader))[0].shape)

    return train_loader, validation_loader, test_loader
