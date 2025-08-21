from abc import ABC, abstractmethod
from pydatatom.datasets import Dataset


class Step(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, context: dict, dataset: Dataset):
        pass

    @abstractmethod
    def transform(self, context: dict, dataset: Dataset):
        pass

    @abstractmethod
    def plot(self, context: dict):
        pass
