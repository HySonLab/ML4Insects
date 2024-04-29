from models.trainer import Trainer 
from models.classifier import Classifier

class EPGDataset(Dataset):
    def __init__(
                self, 
                data_path, 
                dataset_name,
                window_size: int = 1024, 
                hop_length: int = 1024, 
                method: str = 'raw',
                scale: bool = True,
                random_state: int = 28
                ):

        self.data_path = data_path
        self.dataset_name = dataset_name
        self.subdatasets = get_dataset_group(dataset_name)

        self.window_size = window_size
        self.hop_length = hop_length
        self.method = method 
        self.scale = scale 
    
        d, self.n_recordings = generate_train_test_data(dataset_name, window_size, hop_length, method)
        self.windows, self.labels = d['data'], d['label']
        self.waveforms, self.distributions = np.unique(self.labels, return_counts= True)
        n = len(self.waveforms)
        self.distributions = [round(self.distributions[i]/len(self.labels),2) for i in range(n)]
        self.label_map = {1: 0, 2: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6}
        
        self.print_dataset_info()

    def __len__(self):
        return self.len(self.recordings)

    def __getitem__(self, idx):
        return self.windows[idx], self.labels[idx]

    def print_dataset_info(self):
        print(f'Read {self.n_recordings} recordings')
        print(f'Signal processing method: {self.method} | Scale: {str(self.scale)}')
        n = len(self.waveforms)
        print('Class distribution (label:ratio): ' + ', '.join(f'{self.waveforms[i]}: {self.distributions[i]}' for i in range(n)))
        print(f'Labels map (from:to): {self.label_map}')

    def get_subdataset(self, name):
        filenames = os.listdir(f'{self.data_path}/{name}_ANA')
        subdataset = [n[:-4] for n in filenames]
        return subdataset
