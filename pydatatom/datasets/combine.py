from .dataset import Dataset


class Combine(Dataset):
    def __init__(self, a: Dataset, b: Dataset, key: str):
        self.dataset_a = a
        self.dataset_b = b
        self.key = key
        assert len(self.dataset_a) == len(self.dataset_b)

    def __getitem__(self, index: int):
        data_a = self.dataset_a[index]
        data_b = self.dataset_b[index]
        data_a[self.key] = data_b
        return data_a

    def __len__(self):
        return len(self.dataset_a)
