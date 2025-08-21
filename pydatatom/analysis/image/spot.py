import numpy as np
from scipy.ndimage import gaussian_filter


class SpotDetector:
    def detect(self, image):
        raise NotImplementedError


class TopNNMSSpotDetector(SpotDetector):
    def __init__(
        self,
        spot_count,
        spot_size=1.2,
        spot_distance=12,
        window_size=6,
        k_threshold=3.0,
    ):
        self.spot_count = spot_count
        self.spot_size = spot_size
        self.spot_distance = spot_distance
        self.window_size = window_size
        self.k_threshold = k_threshold

    def detect(self, image):
        band = self._apply_dog_filter(image)
        threshold = self._calculate_threshold(band)
        score = self._apply_threshold(band, threshold)
        peaks = self._find_peaks(score)
        centroids = self._refine_centroids(image, peaks)
        return self._sort_centroids(centroids)

    def _apply_dog_filter(self, image):
        return gaussian_filter(image, self.spot_size) - gaussian_filter(
            image, self.spot_size * 3
        )

    def _calculate_threshold(self, band):
        med = np.median(band)
        mad = np.median(np.abs(band - med)) + 1e-12
        return med + self.k_threshold * 1.4826 * mad

    def _apply_threshold(self, band, threshold):
        score = band.copy()
        score[score < threshold] = -np.inf
        return score

    def _find_peaks(self, score):
        H, W = score.shape
        rr, cc = np.ogrid[:H, :W]
        peaks = []

        for _ in range(self.spot_count):
            idx = np.argmax(score)
            val = score.flat[idx]
            if not np.isfinite(val):
                raise ValueError(
                    f"only found {len(peaks)} peaks above threshold (need {self.spot_count})."
                )
            r, c = divmod(idx, W)
            peaks.append((r, c))
            mask = (rr - r) ** 2 + (cc - c) ** 2 <= (self.spot_distance**2)
            score[mask] = -np.inf

        return peaks

    def _refine_centroids(self, image, peaks):
        bg_sub = np.clip(image - np.median(image), 0, None)
        H, W = image.shape
        centroids = []

        for r, c in peaks:
            r0, r1 = max(0, r - self.window_size), min(H, r + self.window_size + 1)
            c0, c1 = max(0, c - self.window_size), min(W, c + self.window_size + 1)
            patch = bg_sub[r0:r1, c0:c1]

            if patch.sum() <= 0:
                centroids.append((float(r), float(c)))
                continue

            pr, pc = np.mgrid[r0:r1, c0:c1]
            s = patch.sum()
            centroids.append(
                (float((pr * patch).sum() / s), float((pc * patch).sum() / s))
            )

        return centroids

    def _sort_centroids(self, centroids):
        centroids = np.array(centroids)
        order = np.lexsort((centroids[:, 1], centroids[:, 0]))
        return centroids[order]
