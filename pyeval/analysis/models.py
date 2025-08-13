import numpy as np
from lmfit import Model, Parameters


def gaussian(x, **params):
    n = sum(1 for key in params.keys() if key.startswith("amp"))

    y = np.zeros_like(x)
    for i in range(n):
        amp = params[f"amp{i}"]
        mean = params[f"mean{i}"]
        width = params[f"width{i}"]
        y += amp * np.exp(-((x - mean) ** 2) / width**2)

    return y + params["offset"]


class Gaussian:
    def __init__(self, n: int):
        self.n = n
        self.model = Model(gaussian)
        self.params = Parameters()
        self.result = None

    def fit(self, x, y):
        for i in range(self.n):
            self.params.add(f"amp{i}", value=y.max(), min=y.min())
            self.params.add(
                f"mean{i}",
                value=x.min() + (x.max() - x.min()) * i / (self.n - 1),
                min=x.min(),
                max=x.max(),
            )
            self.params.add(f"width{i}", value=x.std(), min=0.01)

        self.params.add("offset", value=y.min())

        self.result = self.model.fit(y, self.params, x=x)

    @property
    def means(self):
        return np.array([self.result.params[f"mean{i}"].value for i in range(self.n)])

    @property
    def widths(self):
        return np.array([self.result.params[f"width{i}"].value for i in range(self.n)])

    def predict(self, x):
        if self.result is None:
            raise ValueError("Model has not been fitted yet")
        return self.result.eval(x=x)
