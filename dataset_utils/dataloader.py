import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class custom_dataset(Dataset):
    def __init__(self,X,y):
        self.X = X
        self.y = y 
    def __len__(self):
        return len(self.X)
    def __getitem__(self,i):
        return self.X[i], self.y[i]
    
def train_validation_test_split(data, labels, r = [0.7, 0.1, 0.2], random_state = 28):

    ### Train/val/test split
    assert (len(r) == 3), 'Parameter r must be a list/tuple of length 3'
    r_train, r_val, r_test = r
    X, x_test, Y, y_test = train_test_split(data, labels, test_size = r_test, random_state = random_state, stratify = labels)
    x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size = r_val/r_train, random_state = random_state, stratify = Y)
    
    return x_train, x_val, x_test, y_train, y_val, y_test 

def to_tensor_split(split, device = 'cuda', unsqueeze = False):
    
    x_train, x_val, x_test, y_train, y_val, y_test = split
    ### To torch.tensor 
    x_train = torch.from_numpy(x_train).float().to(device)
    x_val = torch.from_numpy(x_val).float().to(device)
    x_test = torch.from_numpy(x_test).float().to(device)

    y_train = torch.from_numpy(y_train).long().to(device)
    y_val = torch.from_numpy(y_val).long().to(device)
    y_test = torch.from_numpy(y_test).long().to(device)

    ### Reshape
    if unsqueeze == True:
        return x_train.unsqueeze(1), x_val.unsqueeze(1), x_test.unsqueeze(1), y_train, y_val, y_test
    else:
        return x_train, x_val, x_test, y_train, y_val, y_test

def get_loaders(data, labels, r = [0.7, 0.1, 0.2], batch_size = 32, random_state = 28, device = 'cuda', unsqueeze = False):
    x_train, x_val, x_test, y_train, y_val, y_test = to_tensor_split( train_validation_test_split(data, labels, r = r, random_state = 28),
                                                                        device, unsqueeze )

    train_dataset = custom_dataset(x_train,y_train)
    val_dataset = custom_dataset(x_val,y_val)
    test_dataset = custom_dataset(x_test,y_test)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

    print(f'Train, validate, test set sizes: {x_train.shape[0],x_val.shape[0],x_test.shape[0]}')
    print(f'Input shape:',(tuple(next(iter(train_loader))[0].shape)))

    return train_loader, val_loader, test_loader

