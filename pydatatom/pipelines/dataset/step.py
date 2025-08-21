from abc import ABC, abstractmethod
from pydatatom.data import Dataset


class Step(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, context, dataset: Dataset):
        pass

    @abstractmethod
    def transform(self, contexta, dataset: Dataset):
        pass
