import numpy as np
from lmfit import Model, Parameters
from .detector import CurveDetector


def gaussian_mixture(x, **params):
    n = sum(1 for key in params.keys() if key.startswith("amp"))

    y = np.zeros_like(x)
    for i in range(n):
        amp = params[f"amp{i}"]
        mean = params[f"mean{i}"]
        width = params[f"width{i}"]
        y += amp * np.exp(-(((x - mean) / width) ** 2))

    return y + params["offset"]


def shortest_mass_interval(x, y, mass=0.95):
    x = np.asarray(x)
    y = np.asarray(y)

    assert x.ndim == y.ndim == 1 and x.size == y.size, (
        "x and y must be 1D and same length"
    )

    # keep only nonnegative weights
    m = y >= 0
    x, y = x[m], y[m]
    if x.size == 0 or y.sum() == 0:
        return (np.nan, np.nan), (None, None), m  # nothing to do
    # sort by x
    order = np.argsort(x)
    x, y = x[order], y[order]

    target = mass * y.sum()

    # two-pointer sliding window to find minimal-width window with weight >= target
    i = j = 0
    w = 0.0
    best = (np.inf, 0, 0)  # (width, i_best, j_best)

    while i < len(x):
        while j < len(x) and w < target:
            w += y[j]
            j += 1  # window is [i, j-1]
        if w >= target:
            width = x[j - 1] - x[i]
            if width < best[0]:
                best = (width, i, j - 1)
        w -= y[i]
        i += 1

    _, i_best, j_best = best
    x_left, x_right = x[i_best], x[j_best]

    # Build a mask for the original input (optional, handy for slicing)
    # Start by mapping back to original indices
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(order.size)
    inside_sorted = (np.arange(x.size) >= i_best) & (np.arange(x.size) <= j_best)

    mask_original = np.zeros(m.size, dtype=bool)
    mask_original[m] = inside_sorted[inv_order[m]]
    return x_right - x_left


class GaussianCurveDetector(CurveDetector):
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
