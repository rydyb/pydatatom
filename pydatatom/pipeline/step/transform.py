from pydatatom.dataset import Transform

from .step import Step


class TransformStep(Step):
    def __init__(self, transform):
        self._transform = transform

    def fit(self, context: dict, dataset):
        pass

    def transform(self, context: dict, dataset):
        return Transform(dataset, self._transform)

    def plot(self, context: dict):
        print("nothing to plot")
