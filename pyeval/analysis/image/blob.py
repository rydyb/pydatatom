import numpy as np
from scipy.ndimage import gaussian_filter

def topn_nms(image: np.ndarray,
                        spots: int,
                        spot_size: float = 1.2,
                        spot_distance: int = 12,
                        window_size: int = 6,
                        k_threshold: float = 3.0,
                        plot=False):
    # use a difference of Gaussian spatial filter to enhance the contrast
    band = gaussian_filter(image, spot_size) - gaussian_filter(image, spot_size * 3)

    # calculate a robust threshold using median absolute deviation
    med = np.median(band)
    mad = np.median(np.abs(band - med)) + 1e-12
    thr = med + k_threshold * 1.4826 * mad

    # suppress every pixel below the threshold
    score = band.copy()
    score[score < thr] = -np.inf

    H, W = score.shape
    rr, cc = np.ogrid[:H, :W]

    # iteratively find the global maximum and suppress pixels beyond spot_distance
    peaks = []
    for _ in range(spots):
        idx = np.argmax(score)
        val = score.flat[idx]
        if not np.isfinite(val):
            raise ValueError(f"only found {len(peaks)} peaks above threshold (need {spots}).")
        r, c = divmod(idx, W)
        peaks.append((r, c))
        mask = (rr - r) ** 2 + (cc - c) ** 2 <= (spot_distance ** 2)
        score[mask] = -np.inf

    bg_sub = np.clip(image - np.median(image), 0, None)

    # calculate center of mass inside for each of the previously found windows
    cents = []
    for r, c in peaks:
        r0, r1 = max(0, r - window_size), min(H, r + window_size + 1)
        c0, c1 = max(0, c - window_size), min(W, c + window_size + 1)
        patch = bg_sub[r0:r1, c0:c1]
        if patch.sum() <= 0:
            cents.append((float(r), float(c)))
            continue
        pr, pc = np.mgrid[r0:r1, c0:c1]
        s = patch.sum()
        cents.append((float((pr * patch).sum() / s), float((pc * patch).sum() / s)))

    # sort centroids by row and column for comparison
    cents = np.array(cents)
    order = np.lexsort((cents[:, 1], cents[:, 0]))
    cents = cents[order]

    return cents
