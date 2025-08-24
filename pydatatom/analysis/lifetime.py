import numpy as np
from lmfit.models import Model, ExpressionModel
from matplotlib.axes import Axes


def fit_lifetime(x: np.array, y: np.array):
    model = ExpressionModel("amplitude * (1 - exp(-x/decay))", independent_vars=["x"])

    params = model.make_params()
    params["amplitude"].set(value=y.max(), min=0, max=y.max())
    params["decay"].set(value=np.diff(x).mean(), min=0, max=x.max())

    return model.fit(y, params, x=x)


def plot_lifetime(
    x: np.array,
    y: np.array,
    yerr: np.array,
    model: Model,
    ax: Axes,
    axr: Axes | None = None,
    label: str = "",
):
    xfit = np.linspace(np.nanmin(x), np.nanmax(x), 200)
    yfit = model.eval(x=xfit)
    decay = model.params["decay"]

    eb = ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=3, label="_hide_")
    color = eb[0].get_color()

    ax.plot(
        xfit,
        yfit,
        label=f"τ={decay.value:.6g}±{(decay.stderr or np.nan):.2g}" + f"({label})"
        if label
        else "",
        color=color,
    )

    if axr is not None:
        axr.axhline(0, lw=1, color="black")
        axr.plot(x, model.residual, "o", ms=4, color=color)
