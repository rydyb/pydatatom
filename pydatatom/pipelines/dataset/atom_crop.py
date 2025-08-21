import numpy as np
from dataclasses import dataclass
from pydatatom.datasets import Dataset, Transform
from pydatatom.transforms import PointCrop
from .step import Step
from .mean_image import MeanImageState


@dataclass
class AtomCropState(MeanImageState):
    atom_positions: np.ndarray


class AtomCropStep(Step):
    def __init__(self, size: int = 3, method: str = "topn_nms", **kwargs):
        if method == "topn_nms":
            pass
        else:
            raise ValueError(f"Unknown method: {method}")

    def fit(self, context: MeanImageState, dataset: Dataset):
        mean_image = context.mean_image

        context.atom_positions = self.detector.fit(mean_image)

    def transform(self, context: AtomCropState, dataset: Dataset):
        atom_positions = context.atom_positions

        return Transform(dataset, PointCrop(atom_positions, self.size))

    def plot(self, context: AtomCropState):
        from matplotlib import pyplot as plt

        plt.figure()
        plt.title("Detected spots on mean image")
        plt.imshow(context.mean_image.mean(axis=0))
        plt.scatter(
            context.atom_positions[:, 1],
            context.atom_positions[:, 0],
            s=60,
            facecolors="none",
            edgecolors="r",
            linewidths=1.5,
        )
        plt.show()
