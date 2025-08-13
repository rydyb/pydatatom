import numpy as np
from math import lgamma

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
