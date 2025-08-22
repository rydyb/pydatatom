import numpy as np

from pydatatom.dataset import Transform

from .step import Step


class AtomStatsStep(Step):
    def fit(self, context: dict, dataset):
        pass

    def transform(self, context: dict, dataset):
        def probability(atoms):
            atoms = atoms.copy()
            # assume that we failed to detect an atom in the first image if we have one atom in the second image
            atoms[0, :] |= atoms[1, :]

            # return nan if no atoms were detected in the first image as it is likely that something went wrong
            num = atoms[1].sum(dtype=float)
            den = atoms[0].sum(dtype=float)
            nan = np.full_like(num, np.nan, dtype=float)
            return np.divide(num, den, out=nan, where=den != 0)

        return Transform(dataset, probability)

    def plot(self, context: dict):
        print("nothing to plot")
