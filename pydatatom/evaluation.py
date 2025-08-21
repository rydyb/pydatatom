import numpy as np
import pandas as pd
import lmfit
from .dataset import Dataset, TransformDataset
from .transform import MeanSpotCount
from .analysis.image.spot import TopNNMSSpotDetector

from .analysis.models import (
    GaussianMixture,
    DoubleGaussianMixtureOverlap,
    gaussian_mixture,
)


class FixedSpotHistogramEvaluation(FixedSpotDetectionEvaluation):
    def __init__(self, spot_counts_bins: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.spot_counts_bins = spot_counts_bins

    def evaluate(self, dataset: Dataset, plot=True):
        dataset = super().evaluate(dataset, plot=plot)

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

        self.spots = self.spot_counts > self.spot_counts_thresholds

        if plot:
            self.plot_spot_sums_histograms()

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


class FixedSpotSpectroscopyEvaluation(FixedSpotHistogramEvaluation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.df = None

    def evaluate(self, dataset, plot=False, drop_const=True):
        df = pd.DataFrame(
            list(
                TransformDataset(
                    dataset,
                    lambda x: {
                        key: value
                        for (key, value) in x.items()
                        if isinstance(value, float)
                    },
                )
            )
        )
        if drop_const:
            df = df.loc[:, df.nunique(dropna=False).ne(1)]  # drop constant parameters

        super().evaluate(dataset, plot=plot)

        spots = self.spots.copy()
        spots[:, 0, :] |= spots[:, 1, :]

        df["probability"] = spots[:, 1].sum(axis=-1) / spots[:, 0].sum(axis=-1)

        self.df = df

    def plot_resonance(
        self,
        param: str,
        show_residuals: bool = True,
        save_path: str = None,
        fit: bool = True,
    ):
        from matplotlib import pyplot as plt

        g = (
            self.df.groupby(param)["probability"]
            .agg(mean="mean", std="std", n="size")
            .reset_index()
            .sort_values(param)
        )

        x = g[param].to_numpy(float)
        y = g["mean"].to_numpy(float)
        std = g["std"].to_numpy(float)
        n = g["n"].to_numpy(float)
        sigma = std / np.sqrt(np.maximum(n, 1))
        pos = np.isfinite(sigma) & (sigma > 0)
        if not pos.any():
            sigma = None
        else:
            sigma = np.where(pos, sigma, np.median(sigma[pos]))

        if fit:

            def neg_lorentz(x, x0, gamma, A, offset):
                return offset - A * (gamma**2) / ((x - x0) ** 2 + gamma**2)

            model = lmfit.Model(neg_lorentz)
            params = model.make_params()

            x0_0 = x[np.argmin(y)]
            offset_0 = np.max(y)
            A_0 = max(offset_0 - np.min(y), 1e-9)
            gamma_0 = 0.1 * (x.max() - x.min()) if x.max() > x.min() else 1.0

            params["x0"].set(value=x0_0)
            params["gamma"].set(value=gamma_0, min=1e-12)
            params["A"].set(value=A_0, min=0.0)
            params["offset"].set(value=offset_0)

            result = model.fit(
                y, params, x=x, weights=1 / sigma if sigma is not None else None
            )

            popt = [
                result.params[name].value for name in ["x0", "gamma", "A", "offset"]
            ]
            perr = [
                result.params[name].stderr
                if result.params[name].stderr is not None
                else 0.0
                for name in ["x0", "gamma", "A", "offset"]
            ]
            x0, gamma, A, offset = popt

            xs = np.linspace(x.min(), x.max(), 1000)
            yf = neg_lorentz(xs, *popt)

            residuals = result.residual
            r_squared = result.rsquared
        if fit and show_residuals:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        else:
            fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.errorbar(x, y, yerr=std, fmt="o", capsize=3, label="mean ± std")

        if fit:
            ax1.plot(xs, yf, "-", label="fit")
            ax1.axvline(x0, color="orange", linestyle="--", label=f"x₀={x0:.3f}")

            textstr = f"R²={r_squared:.3f}\nx₀={x0:.3f}±{perr[0]:.3f}\nγ={gamma:.3f}±{perr[1]:.3f}"
            ax1.text(
                0.02,
                0.98,
                textstr,
                transform=ax1.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat"),
            )

        ax1.set_ylabel("Probability")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        if fit and show_residuals:
            ax2.plot(x, residuals, "o")
            ax2.axhline(0, color="red", linestyle="-")
            ax2.set_xlabel(param)
            ax2.set_ylabel("Residuals")
            ax2.grid(True, alpha=0.3)
        else:
            ax1.set_xlabel(param)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)

        plt.show()

        if fit:
            return popt
        else:
            return None
