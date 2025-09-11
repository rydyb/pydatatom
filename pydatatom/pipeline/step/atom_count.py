import numpy as np
from dataclasses import dataclass
from lmfit import Model, Parameters

from pydatatom.dataset import Dataset, Transform
from pydatatom.detector import GaussianThresholdDetector

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


class GaussianCurveDetector:
    def __init__(self, n: int):
        self.n = n
        self.model = Model(gaussian_mixture)
        self.result = None

    def fit(self, x, y):
        params = Parameters()

        xmin = x.min()
        xrange = np.percentile(x, 50) - xmin
        xwidth = xrange / self.n

        for i in range(self.n):
            xstart = xmin + xwidth * i
            xend = xmin + xwidth * (i + 1)
            xbuf = xwidth * 0.1

            params.add(
                f"amp{i}", value=y.mean(), min=np.percentile(y, 0.1), max=y.max()
            )
            params.add(
                f"mean{i}",
                value=xstart + xwidth / 2,
                min=xstart + xbuf,
                max=xend - xbuf,
            )
            params.add(f"width{i}", value=x.std(), min=x.std() / 10, max=x.var())

        params.add("offset", value=y.min(), vary=False)

        self.result = self.model.fit(y, params, x=x, method="lbfgsb")

    def predict(self, x):
        if self.result is None:
            raise ValueError("Model has not been fitted yet")
        return self.result.eval(x=x)

    @property
    def amplitude(self):
        return np.array([self.result.params[f"amp{i}"].value for i in range(self.n)])

    @property
    def mean(self):
        return np.array([self.result.params[f"mean{i}"].value for i in range(self.n)])

    @property
    def width(self):
        return np.array([self.result.params[f"width{i}"].value for i in range(self.n)])

    @property
    def offset(self):
        return np.array([self.result.params["offset"].value])


class AtomCountStep(Step):
    def __init__(self, num_bins: int = 100):
        self.num_bins = num_bins

    def fit(self, context: dict, dataset: Dataset):
        natoms = context["atom_centers"].shape[0]
        image_mean = context["image_mean"]
        nimages = image_mean.shape[0]
        nbins = self.num_bins

        binedges = np.histogram_bin_edges(image_mean, bins=nbins)
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
                        "mean0_min": curve.result.params["mean0"].min,
                        "mean0_max": curve.result.params["mean0"].max,
                        "mean1_min": curve.result.params["mean1"].min,
                        "mean1_max": curve.result.params["mean1"].max,
                    }
                )
                thresholds[i, j] = threshold.threshold
            fits.append(fit)

        context["atom_binedges"] = binedges
        context["atom_bincenters"] = bincenters
        context["atom_counts"] = counts
        context["atom_fits"] = fits
        context["atom_thresholds"] = thresholds

    def transform(self, context: dict, dataset: Dataset):
        thresholds = context["atom_thresholds"]
        return Transform(dataset, lambda x: x.mean(axis=(2, 3)) > thresholds)

    def plot(self, context: dict):
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
                ax.fill_between(
                    [fit["mean0_min"], fit["mean0_max"]],
                    y.max(),
                    alpha=0.2,
                    color="orange",
                )
                ax.fill_between(
                    [fit["mean1_min"], fit["mean1_max"]],
                    y.max(),
                    alpha=0.2,
                    color="orange",
                )

                ax.set_title(f"Image {i}, Spot {j}")
                if i == natoms - 1:
                    ax.set_xlabel("Spot sum")
                if j == 0:
                    ax.set_ylabel("Count")

        plt.tight_layout()
        plt.show()
