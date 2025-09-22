import numpy as np
from math import erf
from lmfit import minimize, Model, Parameters
from lmfit.models import GaussianModel

gaussian = GaussianModel(prefix="noise_") + GaussianModel(prefix="signal_")


def fit_gaussian(x, y):
    params = gaussian.make_params()

    for m in gaussian.components:
        params[f"{m.prefix}height"].set(
            value=np.percentile(y, 90), min=np.percentile(y, 0.1), max=y.max()
        )
        params[f"{m.prefix}sigma"].set(
            value=0.1 * x.std(), min=0.02 * x.std(), max=x.std()
        )

    xmargin = 0.1 * x.std()
    xcenter = np.percentile(x, 30)

    xmin_noise = x.min()
    xmax_noise = xcenter - xmargin
    xmin_signal = xcenter + xmargin
    xmax_signal = np.percentile(x, 60)

    params["noise_center"].set(
        value=(xmin_noise + xmax_noise) / 2, min=xmin_noise, max=xmax_noise
    )
    params["signal_center"].set(
        value=(xmin_signal + xmax_signal) / 2, min=xmin_signal, max=xmax_signal
    )

    result = model.fit(y, params, x=x, method="lbfgsb")
    model.result = result
    model.params = params

    return model


def double_gaussian_overlap_infidelity(params, amp, mean, width):
    threshold = params["threshold"].value
    weight = amp * width

    return weight[0] * (erf((mean[0] - threshold) / width[0]) + 1.0) + weight[1] * (
        erf((threshold - mean[1]) / width[1]) + 1.0
    )


class Histogram:
    def __init__(self, nbins=100):
        self.nbins = 100
        self.counts = None
        self.binedges = None
        self.bincenters = None
        self.peak_model = None
        self.threshold = None

    def fit(self, patches):
        counts, binedges = np.histogram(patches, bins=self.nbins)
        bincenters = (binedges[:-1] + binedges[1:]) / 2

        self.counts = counts
        self.binedges = binedges
        self.bincenters = bincenters

        double_gaussian = fit_double_gaussian(bincenters, counts)
        self.peak_model = double_gaussian

        heights = np.array(
            [
                double_gaussian.result.params["noise_height"].value,
                double_gaussian.result.params["signal_height"].value,
            ]
        )
        centers = np.array(
            [
                double_gaussian.result.params["noise_center"].value,
                double_gaussian.result.params["signal_center"].value,
            ]
        )
        sigmas = np.array(
            [
                double_gaussian.result.params["noise_sigma"].value,
                double_gaussian.result.params["signal_sigma"].value,
            ]
        )

        params = Parameters()
        params.add(
            "threshold", value=centers.mean(), min=centers.min(), max=centers.max()
        )

        result = minimize(
            double_gaussian_overlap_infidelity,
            params,
            args=(heights, centers, sigmas),
            method="lbfgsb",
        )
        self.threshold = result.params["threshold"].value

        return self
