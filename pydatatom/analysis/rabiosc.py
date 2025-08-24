import numpy as np
from lmfit.models import ExpressionModel


def fit_rabiosc(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    span = max(xmax - xmin, 1e-9)

    offset0 = float(np.nanmedian(y))
    amp0 = float(0.5 * (np.nanpercentile(y, 95) - np.nanpercentile(y, 5)))
    amp0 = max(abs(amp0), 1e-6)

    order = np.argsort(x)
    xs, ys = x[order], y[order] - offset0
    s = np.sign(ys)
    s[s == 0] = 1
    nzc = int(np.sum(s[1:] * s[:-1] < 0))
    f0 = max(nzc / (2.0 * span), 1.0 / (5.0 * span))
    dx = np.diff(xs) if xs.size > 1 else np.array([span])
    dx_med = float(np.nanmedian(dx)) if dx.size else span
    f_max_nyq = 0.45 / max(dx_med, 1e-12)
    f_min = max(1.0 / (5.0 * span), f0 / 5.0)
    f_max = min(f_max_nyq, f0 * 5.0) if np.isfinite(f0) and f0 > 0 else f_max_nyq
    if f_min >= f_max:
        f_min = max(1.0 / (5.0 * span), 1e-9)
        f_max = f_max_nyq

    omega0 = 2.0 * np.pi * np.clip(f0, f_min, f_max)
    tau0 = max(span / 2.0, 2.0 * dx_med)

    model = ExpressionModel(
        "offset + amplitude * cos(omega * x + phase) * exp(-x / tau)",
        independent_vars=["x"],
    )
    params = model.make_params(
        offset=offset0, amplitude=amp0, omega=omega0, phase=0.0, tau=tau0
    )

    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
    ptp = max(y_max - y_min, 1e-6)
    params["offset"].set(min=y_min - ptp, max=y_max + ptp)
    params["amplitude"].set(min=1e-9, max=3.0 * ptp)
    params["omega"].set(min=2.0 * np.pi * f_min, max=2.0 * np.pi * f_max)
    params["phase"].set(min=-np.pi, max=np.pi)
    params["tau"].set(min=max(2.0 * dx_med, 1e-6), max=10.0 * span)

    return model.fit(y, params, x=x)


def plot_rabiosc(x, y, yerr, model, axis, axisres=None, label=None):
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    xfit = np.linspace(np.nanmin(x), np.nanmax(x), 400)
    yfit = model.eval(x=xfit)

    p = model.params
    omega = p["omega"]
    tau = p["tau"]
    fval = omega.value / (2 * np.pi)
    ferr = (omega.stderr or np.nan) / (2 * np.pi)

    eb = axis.errorbar(x, y, yerr=yerr, fmt="o", capsize=3, label="_hide_")
    color = eb[0].get_color()
    axis.plot(
        xfit,
        yfit,
        color=color,
        label=f"{label or 'series'} | f={fval:.6g}±{ferr:.2g}, τ={tau.value:.6g}±{(tau.stderr or np.nan):.2g}",
    )

    if axisres is not None:
        axisres.axhline(0, lw=1, color="black")
        axisres.plot(x, model.residual, "o", ms=4, color=color)
