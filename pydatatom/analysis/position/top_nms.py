import numpy as np
from scipy.ndimage import gaussian_filter


def _find_peaks(score, blob_num, blob_distance):
    H, W = score.shape
    rr, cc = np.ogrid[:H, :W]
    peaks = []

    for _ in range(blob_num):
        idx = np.argmax(score)
        val = score.flat[idx]
        if not np.isfinite(val):
            raise ValueError(
                f"only found {len(peaks)} peaks above threshold (need {blob_num})."
            )
        r, c = divmod(idx, W)
        peaks.append((r, c))
        mask = (rr - r) ** 2 + (cc - c) ** 2 <= (blob_distance**2)
        score[mask] = -np.inf

    return peaks


def _refine_centroids(image, peaks, window_size):
    bg_sub = np.clip(image - np.median(image), 0, None)
    H, W = image.shape
    centroids = []

    for r, c in peaks:
        r0, r1 = max(0, r - window_size), min(H, r + window_size + 1)
        c0, c1 = max(0, c - window_size), min(W, c + window_size + 1)
        patch = bg_sub[r0:r1, c0:c1]

        if patch.sum() <= 0:
            centroids.append((float(r), float(c)))
            continue

        pr, pc = np.mgrid[r0:r1, c0:c1]
        s = patch.sum()
        centroids.append((float((pr * patch).sum() / s), float((pc * patch).sum() / s)))

    return centroids


def _sort_centroids(self, centroids):
    centroids = np.array(centroids)
    order = np.lexsort((centroids[:, 1], centroids[:, 0]))
    return centroids[order]


def top_nms(
    image: np.array,
    blob_num: int,
    blob_size=1.2,
    blob_distance=12,
    window_size=6,
    k_threshold=3.0,
):
    # differential of Gaussian "bandpass" filter
    band = gaussian_filter(image, blob_size) - gaussian_filter(image, blob_size * 3)

    # threshold via median + k * MAD
    med = np.median(band)
    mad = np.median(np.abs(band - med)) + np.eps
    threshold = med + k_threshold * 1.4826 * mad

    score = band.copy()
    score[score < threshold] = -np.inf

    peaks = _find_peaks(score)
    centroids = _refine_centroids(image, peaks)
    return _sort_centroids(centroids)
