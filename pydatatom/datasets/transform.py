from .dataset import Dataset


class Transform(Dataset):
    def __init__(self, dataset: Dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index: int):
        data = self.dataset[index]
        data = self.transform(data)
        return data

    def __len__(self):
        return len(self.dataset)
