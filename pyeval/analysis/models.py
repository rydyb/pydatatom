import numpy as np
from math import erf
from lmfit import minimize, Model, Parameters


def gaussian_mixture(x, **params):
    n = sum(1 for key in params.keys() if key.startswith("amp"))

    y = np.zeros_like(x)
    for i in range(n):
        amp = params[f"amp{i}"]
        mean = params[f"mean{i}"]
        width = params[f"width{i}"]
        y += amp * np.exp(-(((x - mean) / width) ** 2))

    return y + params["offset"]


class GaussianMixture:
    def __init__(self, n: int):
        self.n = n
        self.model = Model(gaussian_mixture)
        self.result = None

    def fit(self, x, y):
        params = Parameters()

        xrange = x.max() - x.min()
        xwidth = xrange / self.n

        for i in range(self.n):
            xstart = x.min() + xwidth * i
            xend = x.min() + xwidth * (i + 1)

            params.add(
                f"amp{i}", value=y.mean(), min=np.percentile(y, 0.1), max=y.max()
            )
            params.add(
                f"mean{i}",
                value=xstart + xwidth / 2,
                min=xstart,
                max=xend,
            )
            params.add(f"width{i}", value=x.std(), min=x.std() / 10, max=xrange / 2)

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


def double_gaussian_mixture_overlap_infidelity(params, amp, mean, width):
    threshold = params["threshold"].value
    weight = amp * width

    return weight[0] * (erf((mean[0] - threshold) / width[0]) + 1.0) + weight[1] * (
        erf((threshold - mean[1]) / width[1]) + 1.0
    )


class DoubleGaussianMixtureOverlap:
    def __init__(self):
        self.result = None

    def minimize(self, amplitude, mean, width):
        params = Parameters()
        params.add("threshold", value=mean.mean(), min=mean.min(), max=mean.max())

        self.result = minimize(
            double_gaussian_mixture_overlap_infidelity,
            params,
            args=(amplitude, mean, width),
            method="lbfgsb",
        )

    @property
    def threshold(self):
        if self.result is None:
            raise ValueError("Model has not been fitted yet")
        return self.result.params["threshold"].value
