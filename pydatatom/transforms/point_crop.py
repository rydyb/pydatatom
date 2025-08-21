import numpy as np


class PointCrop:
    def __init__(self, points: np.ndarray, size: int):
        self.points = points
        self.size = size

    def __call__(self, image: np.array):
        if image.ndim == 2:
            image = image[None, ...]
        m, h, w = image.shape

        patches = []
        for y0, x0 in self.points:
            y1, y2 = y0 - self.h // 2, y0 + self.h // 2
            x1, x2 = x0 - self.w // 2, x0 + self.w // 2
            patches.append(image[:, y1:y2, x1:x2])
        if not patches:
            raise ValueError("no patches cropped")

        return np.stack(patches, axis=0)
