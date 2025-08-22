import pandas as pd
from abc import ABC, abstractmethod


class Measurement(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def fit(self, df: pd.DataFrame, x: str, y: str):
        pass

    @abstractmethod
    def plot(self, df: pd.DataFrame):
        pass
