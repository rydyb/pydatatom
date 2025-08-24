import numpy as np
from lmfit.models import Model, LorentzianModel, ConstantModel
from matplotlib.axes import Axes


def _guess_lorentzian_dip_params(x: np.array, y: np.array):
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


def fit_resonance(x: np.array, y: np.array):
    model = LorentzianModel() + ConstantModel()

    params = model.make_params()
    for k, (val, vmin, vmax) in _guess_lorentzian_dip_params(x, y).items():
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


def plot_resonance(
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

    params = model.params
    center = params["center"]
    sigma = params["sigma"]

    eb = ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=3, label="_hide_")
    color = eb[0].get_color()

    ax.plot(
        xfit,
        yfit,
        color=color,
        label=f"μ={center.value:.6g}±{(center.stderr or np.nan):.2g}, σ={sigma.value:.6g}±{(sigma.stderr or np.nan):.2g}"
        + (f"{label})" if label else ""),
    )

    if axr is not None:
        axr.axhline(0, lw=1, color="black")
        axr.plot(x, model.residual, "o", ms=4, color=color)
