import numpy as np
from math import lgamma
from sklearn.mixture import GaussianMixture

def _em_two_poisson(x, max_iter=200, tol=1e-6):
    x = np.asarray(x, dtype=np.float64)
    qlo, qhi = np.quantile(x, [0.2, 0.8])
    lam0 = max(1e-6, float(qlo))
    lam1 = max(lam0 + 1e-6, float(qhi))
    pi1 = 0.5
    logfact = np.vectorize(lambda y: lgamma(y + 1.0))(x)
    for _ in range(max_iter):
        logp0 = np.log1p(-pi1) + x * np.log(lam0 + 1e-12) - lam0 - logfact
        logp1 = np.log(pi1 + 1e-12) + x * np.log(lam1 + 1e-12) - lam1 - logfact
        m = np.maximum(logp0, logp1)
        w0 = np.exp(logp0 - m)
        w1 = np.exp(logp1 - m)
        r1 = w1 / (w0 + w1)
        sum_r1 = r1.sum()
        sum_r0 = x.size - sum_r1
        if sum_r1 <= 1e-9 or sum_r1 >= x.size - 1e-9:
            break
        pi1_new = sum_r1 / x.size
        lam1_new = (r1 * x).sum() / max(sum_r1, 1e-12)
        lam0_new = ((1 - r1) * x).sum() / max(sum_r0, 1e-12)
        if lam0_new > lam1_new:
            lam0_new, lam1_new = lam1_new, lam0_new
            pi1_new = 1.0 - pi1_new
        delta = max(
            abs(pi1_new - pi1),
            abs(lam0_new - lam0) / (lam0 + 1e-9),
            abs(lam1_new - lam1) / (lam1 + 1e-9),
        )
        pi1, lam0, lam1 = pi1_new, lam0_new, lam1_new
        if delta < tol:
            break
    pi1 = float(np.clip(pi1, 1e-6, 1 - 1e-6))
    lam0 = float(max(lam0, 1e-9))
    lam1 = float(max(lam1, lam0 + 1e-9))
    return pi1, lam0, lam1

def poisson_earth_mover(counts: np.ndarray):
    x = np.asarray(counts, dtype=np.float64)
    if x.size < 5 or np.allclose(x, x[0]):
        t = int(max(0, np.ceil(x.mean())))
        return t, (0.5, float(x.mean()), float(x.mean()))
    pi1, lam0, lam1 = _em_two_poisson(x)
    den = np.log(lam1) - np.log(lam0)
    if abs(den) < 1e-12:
        tau = int(np.ceil(0.5 * (lam0 + lam1)))
    else:
        x_star = ((lam1 - lam0) + np.log((1 - pi1) / pi1)) / den
        tau = int(max(0, np.ceil(x_star)))
    return tau, (pi1, lam0, lam1)


def gaussian_mixture_fit(counts: np.ndarray, random_state: int = 0):
    x = np.asarray(counts, dtype=float)
    if x.size < 5 or np.allclose(x, x[0]):
        t = int(max(0, np.ceil(np.median(x))))
        return t, {"pi": np.array([0.5, 0.5]), "mu": np.array([x.mean(), x.mean()]), "var": np.array([1.0, 1.0])}

    y = 2.0 * np.sqrt(x + 3.0/8.0)
    Y = y.reshape(-1, 1)

    gmm = GaussianMixture(n_components=2, covariance_type="full", n_init=5, random_state=random_state)
    gmm.fit(Y)

    mu = gmm.means_.ravel()
    var = gmm.covariances_.reshape(2)  # (2,1,1) -> (2,)
    pi = gmm.weights_.ravel()

    order = np.argsort(mu)
    mu0, mu1 = mu[order]
    s0, s1 = np.sqrt(var[order])
    v0, v1 = var[order]
    w0, w1 = pi[order]

    a = 0.5*(1.0/v1 - 1.0/v0)
    b = (mu0/v0 - mu1/v1)
    c = (-mu0**2/(2.0*v0) + mu1**2/(2.0*v1) + np.log(w0) - np.log(w1) - 0.5*np.log(v0) + 0.5*np.log(v1))

    if abs(a) < 1e-12:
        roots = np.array([-c/b])
    else:
        disc = b*b - 4*a*c
        if disc < 0:
            roots = np.array([(mu0+mu1)/2.0])
        else:
            sq = np.sqrt(disc)
            roots = np.array([(-b - sq)/(2*a), (-b + sq)/(2*a)])

    y_thr = None
    for r in roots:
        if min(mu0, mu1) <= r <= max(mu0, mu1):
            y_thr = r
            break
    if y_thr is None:
        y_thr = roots[np.argmin(np.abs(roots - 0.5*(mu0+mu1)))]

    x_thr = (y_thr/2.0)**2 - 3.0/8.0
    tau = int(max(0, np.ceil(x_thr)))

    return tau, {"pi": np.array([w0, w1]), "mu": np.array([mu0, mu1]), "var": np.array([v0, v1]), "gmm": gmm}
