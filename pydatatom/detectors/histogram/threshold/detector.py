import numpy as np
from abc import ABC, abstractmethod


class ThresholdDetector(ABC):
    @abstractmethod
    def fit(self, x, y) -> np.array:
        raise NotImplementedError
