from .dataset import Dataset


class Transform(Dataset):
    def __init__(self, dataset: Dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index: int):
        return self.transform(self.dataset[index])
