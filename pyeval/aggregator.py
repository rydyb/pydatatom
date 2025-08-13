import numpy as np


class Aggregator:
    def update(self, value: np.ndarray):
        raise NotImplementedError

    def result(self):
        raise NotImplementedError


class SumAggregator(Aggregator):
    def __init__(self, dtype=np.float64):
        self.value = None
        self.dtype = dtype

    def update(self, value: np.ndarray):
        if self.value is None:
            self.value = value.copy().astype(self.dtype)
        else:
            self.value += value

    def result(self):
        return self.value


class MeanAggregator(SumAggregator):
    def __init__(self, dtype=np.float64):
        super().__init__(dtype)
        self.count = 0

    def update(self, value: np.ndarray):
        super().update(value)
        self.count += 1

    def result(self):
        return self.value / self.count


class SpotAggregator(Aggregator):
    def __init__(
        self, spots: np.ndarray = None, spot_radius: int = 3, dtype=np.float64
    ):
        self.dtype = dtype
        self.spots = spots
        self.spot_sums = []
        self.spot_radius = spot_radius
        self.num_spots = len(self.spots) if self.spots is not None else 0

    def set_spots(self, spots: np.ndarray):
        self.spots = spots
        self.num_spots = len(self.spots)

    def update(self, images: np.ndarray):
        if self.spots is None:
            raise ValueError(
                "Spots must be set before calling update. Use set_spots() method."
            )

        m, h, w = images.shape

        item_spot_sums = np.zeros((m, self.num_spots))
        for img_idx in range(m):
            image = images[img_idx]
            for spot_idx, (r, c) in enumerate(self.spots):
                # define region around spot with bounds checking
                r_min = max(0, int(r) - self.spot_radius)
                r_max = min(h, int(r) + self.spot_radius + 1)
                c_min = max(0, int(c) - self.spot_radius)
                c_max = min(w, int(c) + self.spot_radius + 1)

                # sum pixels in the region around the spot
                region_sum = image[r_min:r_max, c_min:c_max].mean()
                item_spot_sums[img_idx, spot_idx] = region_sum
        self.spot_sums.append(item_spot_sums)

    def result(self):
        return np.array(self.spot_sums)
