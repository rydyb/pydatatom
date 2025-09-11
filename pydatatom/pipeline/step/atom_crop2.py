import numpy as np
from dataclasses import dataclass
from numpy.lib.stride_tricks import sliding_window_view

from pydatatom.dataset import Dataset, Transform

from ..step import Step
from .image_mean import ImageMeanState


@dataclass
class AtomCrop2State(ImageMeanState):
    atom_centers: np.ndarray | None = None


class AtomCrop2Step(Step):
    def __init__(self, atom_centers: np.array, atom_size: int = 5):
        self.atom_centers = atom_centers.round().astype(int)
        self.atom_size = atom_size

    def fit(self, context: dict, dataset: Dataset):
        context["atom_centers"] = self.atom_centers

    def transform(self, context: dict, dataset: Dataset):
        patch_size = self.atom_size

        cols = context["atom_centers"][:, 0]
        rows = context["atom_centers"][:, 1]

        h, w = context["image_mean"].shape[1:]

        if np.any(rows < self.atom_size // 2) or np.any(
            rows >= h - self.atom_size // 2
        ):
            raise ValueError("atom centers too close to image border")
        if np.any(cols < self.atom_size // 2) or np.any(
            cols >= w - self.atom_size // 2
        ):
            raise ValueError("atom centers too close to image border")

        def point_crop(image: np.ndarray):
            window = sliding_window_view(image, (patch_size, patch_size), axis=(1, 2))
            return window[:, rows, cols]

        return Transform(dataset, point_crop)

    def plot(self, context: dict):
        pass
