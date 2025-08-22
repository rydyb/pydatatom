import numpy as np
from dataclasses import dataclass

from pydatatom.dataset import Dataset, Transform
from pydatatom.detector import GaussianCurveDetector, GaussianThresholdDetector

from .step import Step
from .atom_crop import AtomCropState


@dataclass
class AtomCountState(AtomCropState):
    bins: np.ndarray | None = None
    counts: np.ndarray | None = None


def gaussian_mixture(x, **params):
    n = sum(1 for key in params.keys() if key.startswith("amp"))

    y = np.zeros_like(x)
    for i in range(n):
        amp = params[f"amp{i}"]
        mean = params[f"mean{i}"]
        width = params[f"width{i}"]
        y += amp * np.exp(-(((x - mean) / width) ** 2))

    return y + params["offset"]


class AtomCountStep(Step):
    def __init__(self, num_bins: int = 100):
        self.num_bins = num_bins

    def fit(self, context: AtomCropState, dataset: Dataset):
        natoms = context["atom_centers"].shape[0]
        image_mean = context["image_mean"]
        nimages = image_mean.shape[0]
        nbins = self.num_bins

        _, binedges = np.histogram(image_mean, bins=nbins)
        counts = np.zeros((nimages, natoms, nbins), dtype=int)

        i, j = np.indices((nimages, natoms))

        for patch in dataset:
            mean = patch.mean(axis=(2, 3))
            idx = np.searchsorted(binedges, mean, side="right") - 1
            idx = np.clip(idx, 0, nbins - 1)
            np.add.at(counts, (i, j, idx), 1)

        bincenters = (binedges[:-1] + binedges[1:]) / 2

        thresholds = np.zeros((nimages, natoms))

        fits = []
        for i in range(counts.shape[0]):
            fit = []
            for j in range(counts.shape[1]):
                curve = GaussianCurveDetector(n=2)
                curve.fit(bincenters, counts[i, j])

                threshold = GaussianThresholdDetector()
                threshold.fit(curve.amplitude, curve.mean, curve.width)

                fit.append(
                    {
                        "amp0": curve.amplitude[0],
                        "mean0": curve.mean[0],
                        "width0": curve.width[0],
                        "amp1": curve.amplitude[1],
                        "mean1": curve.mean[1],
                        "width1": curve.width[1],
                        "offset": curve.offset[0],
                        "threshold": threshold.threshold,
                    }
                )
                thresholds[i, j] = threshold.threshold
            fits.append(fit)

        context["atom_binedges"] = binedges
        context["atom_bincenters"] = bincenters
        context["atom_counts"] = counts
        context["atom_fits"] = fits
        context["atom_thresholds"] = thresholds

    def transform(self, context: AtomCropState, dataset: Dataset):
        thresholds = context["atom_thresholds"]
        return Transform(dataset, lambda x: x.mean(axis=(2, 3)) > thresholds)

    def plot(self, context: AtomCropState):
        from matplotlib import pyplot as plt

        nimages = context["atom_counts"].shape[0]
        natoms = context["atom_counts"].shape[1]
        bincenters = context["atom_bincenters"]
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
                fit = context["atom_fits"][i][j]

                x = bincenters
                y = gaussian_mixture(bincenters, **fit)

                ax = axes[j][i]
                ax.bar(
                    x,
                    counts[i, j],
                    color="skyblue",
                    edgecolor="black",
                )
                ax.plot(
                    x,
                    y,
                    color="darkblue",
                    linestyle="--",
                    linewidth=2,
                )
                ax.axvline(
                    fit["threshold"],
                    y.min(),
                    y.max(),
                    color="red",
                    linewidth=2,
                )
                ax.axvline(
                    fit["mean0"],
                    y.min(),
                    y.max(),
                    color="orange",
                    linewidth=1.5,
                    linestyle="dotted",
                )
                ax.axvline(
                    fit["mean1"],
                    y.min(),
                    y.max(),
                    color="orange",
                    linewidth=1.5,
                    linestyle="dotted",
                )

                ax.set_title(f"Image {i}, Spot {j}")
                if i == natoms - 1:
                    ax.set_xlabel("Spot sum")
                if j == 0:
                    ax.set_ylabel("Count")

        plt.tight_layout()
        plt.show()
