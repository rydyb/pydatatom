import glob
from .dataset import Dataset


class GlobDataset(Dataset):
    def __init__(self, glob_pattern: str):
        self.files = glob.glob(glob_pattern)
        self.files.sort()

        if len(self.files) == 0:
            raise RuntimeError(f"no files found for pattern: {glob_pattern}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int):
        return self.files[index]
