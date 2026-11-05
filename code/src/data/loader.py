from pathlib import Path
from src.data.helpers import load_trec_data

class DataLoader:
    def __init__(self, path, dataset):
        '''Dataset loader'''
        self.path = Path(path)
        self.dataset = dataset

    def load(self):
        '''Load and standardize the selected dataset'''
        if self.dataset == 'trec':
            return load_trec_data(self.path)