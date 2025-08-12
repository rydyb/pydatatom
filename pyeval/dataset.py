import os
import gzip
import pickle

class Dataset:

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index: int):
        raise NotImplementedError


class GzipPickleDataset(Dataset):

    def __init__(self, path: str):
        self.files = [
            os.path.join(path, f) for f in os.listdir(path)
            if f.endswith(".gz")
        ]
        self.files.sort(key=os.path.getmtime)

    def __len__(self):
        return len(self.files)


    def __getitem__(self, index: int):
        path = self.files[index]
        with gzip.open(path, 'rb') as file:
                return pickle.load(file)
