import numpy as np
from abc import ABC, abstractmethod


class BlobDetector(ABC):
    @abstractmethod
    def fit(self, image: np.array) -> np.array:
        raise NotImplementedError
