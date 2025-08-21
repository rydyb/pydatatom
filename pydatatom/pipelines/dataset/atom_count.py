import numpy as np
from dataclasses import dataclass
from pydatatom.datasets import Dataset, Transform


from ..step import Step
from .atom_crop import AtomCropState


@dataclass
class AtomCountState(AtomCropState):
    bins: np.ndarray | None = None
    counts: np.ndarray | None = None


class AtomCountStep(Step):
    def __init__(self, num_bins: int = 100):
        self.num_bins = num_bins

    def fit(self, context: AtomCropState, dataset: Dataset):
        natoms = context["atom_centers"].shape[0]
        image_mean = context["image_mean"]
        nimages = image_mean.shape[0]
        nbins = self.num_bins

        _, bins = np.histogram(image_mean, bins=nbins)
        counts = np.zeros((nimages, natoms, nbins), dtype=int)

        i, j = np.indices((nimages, natoms))

        for patch in dataset:
            mean = patch.mean(axis=(2, 3))
            idx = np.searchsorted(bins, mean, side="right") - 1
            idx = np.clip(idx, 0, nbins - 1)
            np.add.at(counts, (i, j, idx), 1)

        context["atom_bins"] = bins
        context["atom_counts"] = counts

    def transform(self, context: AtomCropState, dataset: Dataset):
        return dataset
        # reduce = lambda patches: patches.mean(axis=0)

        # return Transform(dataset, reduce)

    def plot(self, context: AtomCropState):
        from matplotlib import pyplot as plt

        nimages = context["atom_counts"].shape[0]
        natoms = context["atom_counts"].shape[1]
        bins = context["atom_bins"]
        counts = context["atom_counts"]

        fig, axes = plt.subplots(
            natoms, nimages, sharey=True, figsize=(nimages * 3, natoms * 3)
        )

        if natoms == 1 and nimages == 1:
            axes = [[axes]]
        elif natoms == 1:
            axes = [axes]
        elif nimages == 1:
            axes = [[ax] for ax in axes]

        for i in range(nimages):
            for j in range(natoms):
                ax = axes[j][i]

                ax.bar(
                    bins[:-1],
                    counts[i, j],
                    color="skyblue",
                    edgecolor="black",
                )

                ax.set_title(f"Image {i}, Spot {j}")
                if i == natoms - 1:
                    ax.set_xlabel("Spot sum")
                if j == 0:
                    ax.set_ylabel("Count")

        plt.tight_layout()
        plt.show()
