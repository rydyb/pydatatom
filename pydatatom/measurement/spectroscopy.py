import numpy as np
import pandas as pd
from lmfit.models import LorentzianModel, ConstantModel

from .measurement import Measurement


def fit_lorentzian_dip(x, y):
    model = LorentzianModel() + ConstantModel()

    g = guess_lorentzian_dip_params(x, y)

    params = model.make_params()
    for k, (val, vmin, vmax) in g.items():
        params[k].set(value=val, min=vmin, max=vmax)

    result = model.fit(
        y, params, x=x, method="differential_evolution", nan_policy="omit"
    )
    result = model.fit(
        y,
        result.params,
        x=x,
        method="least_squares",
        fit_kws={"loss": "soft_l1", "f_scale": 1.0},
        nan_policy="omit",
    )

    return result


def guess_lorentzian_dip_params(x, y):
    yptp = max(np.ptp(y), np.finfo(float).eps)

    xr = x.max() - x.min()
    dx = np.median(np.diff(np.sort(x)))

    c0 = np.percentile(y, 90)
    depth = max(c0 - y.min(), np.finfo(float).eps)

    sigma_min = max(dx, 1e-9)
    sigma_init = max(xr / 20.0, sigma_min)
    sigma_max = max(xr, sigma_init * 2)

    amplitude_init = -depth * np.pi * sigma_init
    amplitude_min = -2.0 * yptp * np.pi * sigma_max
    amplitude_max = -1e-16

    center_min = x.min()
    center_max = x.max()
    center_init = x[np.argmin(y)]

    offset_margin = 0.5 * yptp
    offset_init = c0
    offset_min = y.min() - offset_margin
    offset_max = y.max() + offset_margin

    return {
        "amplitude": (amplitude_init, amplitude_min, amplitude_max),
        "center": (center_init, center_min, center_max),
        "sigma": (sigma_init, sigma_min, sigma_max),
        "c": (offset_init, offset_min, offset_max),
    }


def aggregate(x: np.array, y: np.array):
    edges = np.histogram_bin_edges(x, bins="fd")
    nbins = len(edges) - 1

    idx = np.digitize(x, edges, right=True) - 1
    idx[idx == nbins] = nbins - 1

    valid = (idx >= 0) & (idx < nbins)
    idx = idx[valid]
    y = y[valid]

    count = np.bincount(idx, minlength=nbins)
    sum_y = np.bincount(idx, weights=y, minlength=nbins)
    sum_ysq = np.bincount(idx, weights=y * y, minlength=nbins)

    mean = np.divide(sum_y, count, out=np.full(nbins, np.nan), where=count > 0)
    var_num = sum_ysq - count * mean**2
    var = np.divide(var_num, count - 1, out=np.full(nbins, np.nan), where=count > 1)
    std = np.sqrt(np.maximum(var, 0.0))

    left = edges[:-1]
    right = edges[1:]
    x_rep = 0.5 * (left + right)

    return {
        "x": x_rep,
        "y": mean,
        "yerr": std,
        "n": count,
        "left": left,
        "right": right,
        "edges": edges,
    }


class Spectroscopy(Measurement):
    def __init__(self):
        self._model = None
        self._data = None

    @property
    def fitresult(self):
        if self._model is None:
            raise ValueError("no fit result available")
        return self._model.params

    @property
    def data(self):
        if self._data is None:
            raise ValueError("no data available")
        return self._data

    def fit(self, x: np.array, y: np.array):
        self._data = aggregate(x, y)
        self._model = fit_lorentzian_dip(self._data["x"], self._data["y"])

    def predict(self, x: np.array) -> np.array:
        return self._model.eval(x=x)

    def plot(self):
        from matplotlib import pyplot as plt

        xdata = self._data["x"]
        ydata = self._data["y"]
        ydataerr = self._data["yerr"]

        xfit = np.linspace(xdata.min(), xdata.max(), 100)
        yfit = self.predict(x=xfit)

        result = self._model.params
        center = result["center"]
        width = result["sigma"]
        residual = self._model.residual

        fig, (ax, axr) = plt.subplots(
            2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]}, figsize=(6, 5)
        )

        # data + fit
        ax.errorbar(xdata, ydata, yerr=ydataerr, fmt="o", capsize=3, label="mean ± std")
        ax.plot(xfit, yfit, color="orange", label="fit")
        ax.set_ylabel("Probability")
        ax.set_title(
            f"μ = {center.value:.6g} ± {center.stderr:.2g}, "
            f"σ = {width.value:.6g} ± {width.stderr:.2g}"
        )
        ax.legend()

        # residuals
        axr.axhline(0, lw=1, color="black")
        axr.plot(xdata, residual, "o", ms=4)
        axr.set_xlabel("Scan parameter")
        axr.set_ylabel("Residuals")

        fig.tight_layout()
        plt.show()

    def to_pandas(self):
        return pd.DataFrame(self._data)
