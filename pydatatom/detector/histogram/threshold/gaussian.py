import numpy as np
from math import erf
from lmfit import minimize, Parameters
from .detector import ThresholdDetector


def double_gaussian_mixture_overlap_infidelity(params, amp, mean, width):
    threshold = params["threshold"].value
    weight = amp * width

    return weight[0] * (erf((mean[0] - threshold) / width[0]) + 1.0) + weight[1] * (
        erf((threshold - mean[1]) / width[1]) + 1.0
    )


class GaussianThresholdDetector(ThresholdDetector):
    def __init__(self):
        self.result = None

    def fit(self, amplitude, mean, width):
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
