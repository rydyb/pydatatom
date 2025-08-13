import numpy as np
from .dataset import Dataset
from .aggregator import MeanAggregator, SpotAggregator
from .analysis.image.spot_detector import TopNNMSSpotDetector


class Evaluation:
    def __init__(self):
        self.build()

    def build(self):
        raise NotImplementedError

    def evaluate(self, dataset: Dataset):
        raise NotImplementedError


class MeanImageEvaluation(Evaluation):
    def build(self):
        self.mean_images = None
        self.mean_aggregator = MeanAggregator()

    def evaluate(self, dataset: Dataset):
        for item in dataset:
            self.mean_aggregator.update(item["image"])
        self.mean_images = self.mean_aggregator.result()


class FixedSpotDetectionEvaluation(MeanImageEvaluation):
    def __init__(
        self,
        spot_counts: int,
        spot_radius: int = 3,
    ):
        self.spot_counts = spot_counts
        self.spot_radius = spot_radius

        self.build()

    def build(self):
        super().build()

        self.spot_detector = TopNNMSSpotDetector(self.spot_counts)

    def evaluate(self, dataset: Dataset):
        super().evaluate(dataset)

        self.mean_image = self.mean_images.mean(axis=0)
        self.spots = self.spot_detector.detect(self.mean_image)

    def plot_detected_spots(self):
        from matplotlib import pyplot as plt

        plt.figure()
        plt.title("Detected spots on mean image")
        plt.imshow(self.mean_image)
        plt.scatter(
            self.spots[:, 1],
            self.spots[:, 0],
            s=60,
            facecolors="none",
            edgecolors="r",
            linewidths=1.5,
        )
        plt.show()


class FixedSpotSumEvaluation(FixedSpotDetectionEvaluation):
    def build(self):
        super().build()

        self.spot_aggregator = SpotAggregator()

    def evaluate(self, dataset: Dataset):
        super().evaluate(dataset)

        self.spot_aggregator.set_spots(self.spots)
        for item in dataset:
            self.spot_aggregator.update(item["image"])
        self.spot_sums = self.spot_aggregator.result()


class FixedSpotHistogramEvaluation(FixedSpotSumEvaluation):
    def __init__(self, spot_sums_bins: int, *args, **kwargs):
        self.spot_sums_bins = spot_sums_bins

        super().__init__(*args, **kwargs)

    def build(self):
        super().build()

    def evaluate(self, dataset: Dataset):
        super().evaluate(dataset)

        n, m, l = self.spot_sums.shape

        self.spot_sums_counts = np.zeros((m, l, self.spot_sums_bins), dtype=int)
        self.spot_sums_binedges = np.zeros((m, l, self.spot_sums_bins + 1))

        for m_idx in range(m):
            for l_idx in range(l):
                counts, binedges = np.histogram(
                    self.spot_sums[:, m_idx, l_idx], bins=self.spot_sums_bins
                )
                self.spot_sums_counts[m_idx, l_idx] = counts
                self.spot_sums_binedges[m_idx, l_idx] = binedges

        self.spot_sums_centers = (
            self.spot_sums_binedges[:, :, :-1] + self.spot_sums_binedges[:, :, 1:]
        ) / 2
        self.spot_sums_binwidths = (
            self.spot_sums_binedges[:, :, 1:] - self.spot_sums_binedges[:, :, :-1]
        )

    def plot_spot_sums_histograms(self):
        from matplotlib import pyplot as plt

        m, l, _ = self.spot_sums_counts.shape

        fig, axes = plt.subplots(l, m, figsize=(m * 3, l * 3))

        if l == 1 and m == 1:
            axes = [[axes]]
        elif l == 1:
            axes = [axes]
        elif m == 1:
            axes = [[ax] for ax in axes]

        for l_idx in range(l):
            for m_idx in range(m):
                ax = axes[l_idx][m_idx]
                ax.bar(
                    self.spot_sums_centers[m_idx, l_idx],
                    self.spot_sums_counts[m_idx, l_idx],
                    width=self.spot_sums_binwidths[m_idx, l_idx],
                )
                ax.set_title(f"Image {m_idx}, Spot {l_idx}")
                ax.set_xlabel("Spot sum")
                ax.set_ylabel("Count")

        plt.tight_layout()
        plt.show()
