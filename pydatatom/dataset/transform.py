from .dataset import Dataset


class Transform(Dataset):
    def __init__(self, dataset: Dataset, transform):
        self.dataset = dataset
        self.transforms = transform if isinstance(transform, list) else [transform]

    def __getitem__(self, index: int):
        data = self.dataset[index]
        for transform in self.transforms:
            data = transform(data)
        return data

    def __len__(self):
        return len(self.dataset)
