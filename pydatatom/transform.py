import numpy as np


class MeanSpotCount:
    def __init__(self, spot_positions: np.ndarray, spot_radius: int = 3):
        self.spot_positions = spot_positions
        self.spot_radius = spot_radius

    def __call__(self, image: np.array):
        m, h, w = image.shape

        counts = np.zeros((m, len(self.spot_positions)), dtype=np.float64)

        for image_idx in range(m):
            for spot_idx, (i, j) in enumerate(self.spot_positions):
                # define region around spot with bounds checking
                i_min = max(0, int(i) - self.spot_radius)
                i_max = min(h, int(i) + self.spot_radius + 1)
                j_min = max(0, int(j) - self.spot_radius)
                j_max = min(w, int(j) + self.spot_radius + 1)

                # sum pixels in the region around the spot
                counts[image_idx, spot_idx] = image[
                    image_idx, i_min:i_max, j_min:j_max
                ].mean()

        return counts
