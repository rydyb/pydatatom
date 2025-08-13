import os
import gzip
import pickle
import glob
from torch.utils.data import Dataset


class TransformDataset(Dataset):
    def __init__(self, dataset: Dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        data = self.dataset[index]
        if self.transform:
            data = self.transform(data)
        return data


class GlobDataset(Dataset):
    def __init__(self, glob_pattern: str, transform=None):
        self.files = glob.glob(glob_pattern)
        self.files.sort()
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int):
        path = self.files[index]
        if self.transform:
            path = self.transform(path)
        return path


class GzipPickleDataset(GlobDataset):
    def __init__(self, path: str, transform=None):
        glob_pattern = os.path.join(path, "*.gz")
        super().__init__(glob_pattern, transform)
        self.files.sort(key=os.path.getmtime)

    def __getitem__(self, index: int):
        path = self.files[index]
        with gzip.open(path, "rb") as file:
            data = pickle.load(file)
        if self.transform:
            data = self.transform(data)
        return data
