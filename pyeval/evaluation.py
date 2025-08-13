import numpy as np
from .dataset import Dataset, TransformDataset
from .transform import MeanSpotCount
from .analysis.image.spot import TopNNMSSpotDetector

from .analysis.models import (
    GaussianMixture,
    DoubleGaussianMixtureOverlap,
    gaussian_mixture,
)


class Evaluation:
    def evaluate(self, dataset: Dataset):
        raise NotImplementedError


class MeanImageEvaluation(Evaluation):
    def __init__(self):
        self.mean_image = None

    def evaluate(self, dataset: Dataset):
        dataset = TransformDataset(dataset, lambda x: x["image"])

        for image in dataset:
            if self.mean_image is None:
                self.mean_image = image.copy().astype(np.float64)
            else:
                self.mean_image += image
        self.mean_image /= len(dataset)

        return dataset


class FixedSpotDetectionEvaluation(MeanImageEvaluation):
    def __init__(
        self,
        spot_num: int,
        spot_radius: int = 3,
    ):
        super().__init__()

        self.spot_num = spot_num
        self.spot_radius = spot_radius
        self.spot_counts = None

    def evaluate(self, dataset: Dataset):
        dataset = super().evaluate(dataset)

        self.spot_positions = TopNNMSSpotDetector(self.spot_num).detect(
            self.mean_image.mean(axis=0)
        )
        mean_spot_count = MeanSpotCount(self.spot_positions)

        dataset = TransformDataset(dataset, mean_spot_count)

        spot_counts = []
        for spot_count in dataset:
            spot_counts.append(spot_count)
        self.spot_counts = np.array(spot_counts)

        return dataset

    def plot_detected_spots(self):
        from matplotlib import pyplot as plt

        plt.figure()
        plt.title("Detected spots on mean image")
        plt.imshow(self.mean_image.mean(axis=0))
        plt.scatter(
            self.spot_positions[:, 1],
            self.spot_positions[:, 0],
            s=60,
            facecolors="none",
            edgecolors="r",
            linewidths=1.5,
        )
        plt.show()


class FixedSpotHistogramEvaluation(FixedSpotDetectionEvaluation):
    def __init__(self, spot_counts_bins: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.spot_counts_bins = spot_counts_bins

    def evaluate(self, dataset: Dataset):
        dataset = super().evaluate(dataset)

        n, m, l = self.spot_counts.shape

        self.spot_counts_histogram = np.zeros((m, l, self.spot_counts_bins), dtype=int)
        self.spot_counts_binedges = np.zeros((m, l, self.spot_counts_bins + 1))
        self.spot_counts_bincenters = np.zeros((m, l, self.spot_counts_bins))
        self.spot_counts_thresholds = np.zeros((m, l))

        self.spot_counts_gaussian2d_params = []

        for m_idx in range(m):
            gaussian2d_params = []

            for l_idx in range(l):
                counts, binedges = np.histogram(
                    self.spot_counts[:, m_idx, l_idx],
                    bins=self.spot_counts_bins,
                )
                self.spot_counts_histogram[m_idx, l_idx] = counts
                self.spot_counts_binedges[m_idx, l_idx] = binedges

                bincenters = (binedges[:-1] + binedges[1:]) / 2

                gaussian_model = GaussianMixture(n=2)
                gaussian_model.fit(bincenters, counts)

                overlap_optimizer = DoubleGaussianMixtureOverlap()
                overlap_optimizer.minimize(
                    gaussian_model.amplitude, gaussian_model.mean, gaussian_model.width
                )

                gaussian2d_params.append(
                    {
                        "amp0": gaussian_model.amplitude[0],
                        "mean0": gaussian_model.mean[0],
                        "width0": gaussian_model.width[0],
                        "amp1": gaussian_model.amplitude[1],
                        "mean1": gaussian_model.mean[1],
                        "width1": gaussian_model.width[1],
                        "offset": gaussian_model.offset[0],
                    }
                )

                self.spot_counts_thresholds[m_idx, l_idx] = overlap_optimizer.threshold
                self.spot_counts_bincenters[m_idx, l_idx] = bincenters

            self.spot_counts_gaussian2d_params.append(gaussian2d_params)

        self.spot_counts_binwidths = (
            self.spot_counts_binedges[:, :, 1:] - self.spot_counts_binedges[:, :, :-1]
        )

        self.spot_counts_bincenters = (
            self.spot_counts_binedges[:, :, 1:] + self.spot_counts_binedges[:, :, :-1]
        ) / 2

        self.spots = self.spot_counts > self.spot_counts_thresholds

    def plot_spot_sums_histograms(self):
        from matplotlib import pyplot as plt

        m, l, _ = self.spot_counts_histogram.shape

        fig, axes = plt.subplots(l, m, figsize=(m * 3, l * 3), sharey=True)

        if l == 1 and m == 1:
            axes = [[axes]]
        elif l == 1:
            axes = [axes]
        elif m == 1:
            axes = [[ax] for ax in axes]

        for l_idx in range(l):
            for m_idx in range(m):
                x = self.spot_counts_bincenters[m_idx, l_idx]
                y = self.spot_counts_histogram[m_idx, l_idx]

                p = self.spot_counts_gaussian2d_params[m_idx][l_idx]

                ax = axes[l_idx][m_idx]
                ax.bar(
                    x,
                    y,
                    width=self.spot_counts_binwidths[m_idx, l_idx],
                    color="skyblue",
                    edgecolor="black",
                )
                ax.plot(
                    x,
                    gaussian_mixture(
                        x,
                        **p,
                    ),
                    color="darkblue",
                    linestyle="--",
                    linewidth=2,
                )
                ax.axvline(
                    self.spot_counts_thresholds[m_idx, l_idx],
                    y.min(),
                    y.max(),
                    color="red",
                    linewidth=2,
                )
                ax.axvline(
                    p["mean0"],
                    y.min(),
                    y.max(),
                    color="orange",
                    linewidth=1.5,
                    linestyle="dotted",
                )
                ax.axvline(
                    p["mean1"],
                    y.min(),
                    y.max(),
                    color="orange",
                    linewidth=1.5,
                    linestyle="dotted",
                )
                ax.set_title(f"Image {m_idx}, Spot {l_idx}")
                if l_idx == l - 1:
                    ax.set_xlabel("Spot sum")
                if m_idx == 0:
                    ax.set_ylabel("Count")

        plt.tight_layout()
        plt.show()
