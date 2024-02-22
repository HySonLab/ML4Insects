import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class custom_dataset(Dataset):
    def __init__(self,X,y):
        self.X = X
        self.y = y 
    def __len__(self):
        return len(self.X)
    def __getitem__(self,i):
        return self.X[i], self.y[i]
    
def train_validation_test(train, test, r = 0.1, model_type = 'mlp', random_state = 28):
    '''
        Input: 
            train: dictionary with keys (data,label) - Data for training and validation
            test: dictionary with keys (data,label) - Data for testing
            r: train/validation ratio or train, validation index split
        Output:
            train, test, validation split 
    '''
    df, lab = train['data'], train['label']
    df_test, lab_test = test['data'], test['label']

    if isinstance(r, float):
        x_train, x_validation, y_train, y_validation = train_test_split(df, lab, test_size = r, stratify=lab,random_state = random_state)
    elif len(r) == 2:
        train_idx = r[0]; val_idx = r[1]
        x_train = df[train_idx]; x_validation = df[val_idx]
        y_train = lab[train_idx]; y_validation = lab[val_idx]
    else: 
        raise RuntimeError('Param "r" should be a float or a tuple/list of indices.')
    x_train = torch.from_numpy(x_train).float()
    x_validation = torch.from_numpy(x_validation).float()
    x_test = torch.from_numpy(df_test).float()

    y_train = torch.from_numpy(y_train).long()
    y_validation = torch.from_numpy(y_validation).long()
    y_test = torch.from_numpy(lab_test).long()

    if model_type == 'mlp':
        return x_train, x_validation, x_test, y_train, y_validation, y_test
    
    elif model_type == 'cnn': #squeeze a channel dimension
        return x_train.unsqueeze(1), x_validation.unsqueeze(1), x_test.unsqueeze(1), y_train.unsqueeze(1), y_validation.unsqueeze(1), y_test.unsqueeze(1)
    
    elif model_type == 'transformer': #squeeze a channel dimension
        return x_train.unsqueeze(-1), x_validation.unsqueeze(-1), x_test.unsqueeze(-1), y_train.unsqueeze(-1), y_validation.unsqueeze(-1), y_test.unsqueeze(-1)
    
from torch.utils.data import DataLoader
def get_loader(X, y, r=0.1, batch_size = 32, random_state = 28, model_type = 'mlp'):
    '''
        Input: 
            X: dictionary with keys (data,label) - Data for training and validation
            y: dictionary with keys (data,label) - Data for testing
            r: train/validation ratio
        Output:
            train, test, validation DataLoaders 
    '''    
    x_train, x_validation, x_test, y_train, y_validation, y_test = train_validation_test(X,y,r,model_type,random_state=random_state)
    
    train_dataset = custom_dataset(x_train,y_train)
    validation_dataset = custom_dataset(x_validation,y_validation)
    test_dataset = custom_dataset(x_test,y_test)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    validation_loader = DataLoader(validation_dataset, batch_size = batch_size, shuffle = False)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

    print(f'Train, validate, test sizes: {x_train.shape[0],x_validation.shape[0],x_test.shape[0]}')
    print(f'Input shape:',(tuple(next(iter(train_loader))[0].shape)))

    return train_loader, validation_loader, test_loader

