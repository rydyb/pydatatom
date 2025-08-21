import numpy as np
from abc import ABC, abstractmethod


class CurveDetector(ABC):
    @abstractmethod
    def fit(self, x, y) -> np.array:
        raise NotImplementedError

    @abstractmethod
    def predict(self, x) -> np.array:
        raise NotImplementedError
