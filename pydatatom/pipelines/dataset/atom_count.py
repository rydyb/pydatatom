import numpy as np
from dataclasses import dataclass
from pydatatom.datasets import Dataset, Transform
from ..step import Step
from .atom_crop import AtomCropState


@dataclass
class AtomCountState(AtomCropState):
    pass


class AtomCountStep(Step):
    def __init__(self, num_bins: int = 50):
        self.num_bins = num_bins

    def fit(self, context: AtomCropState, dataset: Dataset):
        num_runs = len(dataset)
        num_atoms = context.atom_positions.shape[1]
        num_images = context.atom_positions.shape[0]

    def transform(self, context: AtomCropState, dataset: Dataset):
        reduce = lambda patches: patches.mean(axis=0)

        return Transform(dataset, reduce)

    def plot(self, context: AtomCropState):
        from matplotlib import pyplot as plt

        plt.figure()
        plt.title("Detected spots on mean image")
        plt.imshow(context.mean_image.mean(axis=0))
        plt.scatter(
            context.atom_positions[:, 1],
            context.atom_positions[:, 0],
            s=60,
            facecolors="none",
            edgecolors="r",
            linewidths=1.5,
        )
        plt.show()
