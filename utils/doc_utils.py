import numpy as np 
import os 
wd = os.getcwd()

def get_filename(name):
    list_files = os.listdir(os.path.join(wd,'data',name))
    unique = []
    for name in list_files:
        name = name.split('.')[0]
        unique.append(name)
    unique = np.unique(unique)
    return unique
    
def get_train_test_filenames(train_ratio = None, n_train = None, n_test = None, name = None, random_state = 10):
    np.random.seed(random_state)
    if name is None:
        list_dir = [d for d in os.listdir('./data/') if not d.endswith('_ANA') if '.' not in d]
    else:
        if isinstance(name, list):
            list_dir = name
    splits = {}

    for name in list_dir: 
        recording_names = get_filename(name)
        np.random.shuffle(recording_names)
        if (train_ratio is not None):
            n = int(train_ratio*len(recording_names))
            train_name = recording_names[:n]
            test_name = recording_names[n:]
        elif (n_train is not None and n_test is not None):
            n = min(n_train,len(recording_names)-1)
            train_name = recording_names[:n]
            test_name = recording_names[n:n+n_test]       
        # print(n, name)     
        splits[name] = (train_name, test_name)
    return splits
