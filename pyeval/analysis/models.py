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


# Here, but in the codebase in general, I would appreciate type hints. What do you expect x,y to be?
def shortest_mass_interval(x, y, mass=0.95):
    x = np.asarray(x)
    y = np.asarray(y)

    assert x.ndim == y.ndim == 1 and x.size == y.size, (
        "x and y must be 1D and same length"
    )
    assert 0 < mass <= 1, f"mass must be in (0, 1], got {mass}"

    # keep only nonnegative weights
    nonnegative_mask = y >= 0
    x, y = x[nonnegative_mask], y[nonnegative_mask]

    if x.size == 0 or y.sum() == 0:
        return (np.nan, np.nan), (None, None), nonnegative_mask  # nothing to do

    # sort by x
    order = np.argsort(x)
    x, y = x[order], y[order]

    target = mass * y.sum()

    # two-pointer sliding window to find minimal-width window with weight >= target
    left = right = 0
    w = 0.0
    best = (np.inf, 0, 0)  # (width, i_best, j_best)

    while left < len(x):
        while right < len(x) and w < target:
            w += y[right]
            right += 1  # window is [left, right-1]
        if w >= target:
            width = x[right - 1] - x[left]
            if width < best[0]:
                best = (width, left, right - 1)
        w -= y[left]
        left += 1

    _, left_best, right_best = best
    x_left, x_right = x[left_best], x[right_best]

    # Build a mask for the original input (optional, handy for slicing)
    # Start by mapping back to original indices
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(order.size)
    inside_sorted = (np.arange(x.size) >= left_best) & (np.arange(x.size) <= right_best)

    mask_original = np.zeros(nonnegative_mask.size, dtype=bool)
    mask_original[nonnegative_mask] = inside_sorted[inv_order[nonnegative_mask]]
    return x_right - x_left


class GaussianMixture:
    def __init__(self, n: int):
        self.n = n
        self.model = Model(gaussian_mixture)
        self.result = None

    def fit(self, x, y):
        params = Parameters()

        # xrange = x.max() - x.min()
        xrange = shortest_mass_interval(x, y, mass=0.95)
        xwidth = xrange / self.n

        for i in range(self.n):
            xstart = x.min() + xwidth * i
            xend = x.min() + xwidth * (i + 1)
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


def double_gaussian_mixture_overlap_infidelity(params, amp, mean, width):
    threshold = params["threshold"].value
    weight = amp * width

    # Would add a comment and or more descriptive variable names. I want to be able to check that this is correct. (Where?)
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
