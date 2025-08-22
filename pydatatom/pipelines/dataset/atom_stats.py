import numpy as np

from pydatatom.datasets import Transform
from ..step import Step


class AtomStatsStep(Step):
    def fit(self, context: dict, dataset):
        pass

    def transform(self, context: dict, dataset):
        def probability(atoms):
            atoms = atoms.copy()
            atoms[0, :] |= atoms[1, :]
            num = atoms[1].sum(dtype=float)
            den = atoms[0].sum(dtype=float)
            return np.divide(num, den, out=np.zeros_like(num), where=den != 0)

        return Transform(dataset, probability)

    def plot(self, context: dict):
        print("nothing to plot")
