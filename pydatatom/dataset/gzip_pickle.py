import os
import gzip
import pickle
from .glob import GlobDataset


class GzipPickleDataset(GlobDataset):
    def __init__(self, path: str):
        glob_pattern = os.path.join(path, "*.gz")
        super().__init__(glob_pattern)
        self.files.sort(key=os.path.getmtime)

    def __getitem__(self, index: int):
        path = self.files[index]
        with gzip.open(path, "rb") as file:
            return pickle.load(file)
