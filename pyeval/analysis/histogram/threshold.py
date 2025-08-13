from math import erf
from lmfit import minimize, Parameters
from ..models import Gaussian


def gaussian2d_overlap(params, mean0, mean1, width0, width1):
    threshold = params["threshold"].value

    return erf((mean0 - threshold) / width0) + erf((threshold - mean1) / width1)


class MinGaussian2dOverlapThreshold:
    def __init__(self):
        self.gaussian = Gaussian(2)

        self.params = Parameters()
        self.result = None

    def fit(self, x, y):
        self.gaussian.fit(x, y)

        mean0 = self.gaussian.result.params["mean0"]
        mean1 = self.gaussian.result.params["mean1"]
        width0 = self.gaussian.result.params["width0"]
        width1 = self.gaussian.result.params["width1"]

        self.params.add(
            "threshold", value=(mean0 + mean1) / 2, min=x.min(), max=x.max()
        )

        self.result = minimize(
            gaussian2d_overlap,
            self.params,
            args=(mean0, mean1, width0, width1),
            method="nelder",
        )

    @property
    def threshold(self):
        if self.result is None:
            raise ValueError("Model has not been fitted yet")
        return self.result.params["threshold"].value
