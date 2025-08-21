import numpy as np
from dataclasses import dataclass
from pydatatom.datasets import Dataset, Transform
from pydatatom.transforms import PointCrop
from pydatatom.detectors import TopNNMSBlobDetector
from ..step import Step
from .mean_image import MeanImageState


@dataclass
class AtomCropState(MeanImageState):
    atom_positions: np.ndarray | None = None


class AtomCropStep(Step):
    def __init__(self, atom_num: int, atom_size: int = 3):
        self.atom_num = atom_num
        self.atom_size = atom_size
        self.detector = TopNNMSBlobDetector(blob_num=atom_num, blob_size=atom_size)

    def fit(self, context: MeanImageState, dataset: Dataset):
        mean_image = context["mean_image"].mean(axis=0)
        context["atom_positions"] = self.detector.fit(mean_image)

    def transform(self, context: AtomCropState, dataset: Dataset):
        atom_positions = context["atom_positions"].round().astype(int)
        atom_size = self.atom_size

        return Transform(dataset, PointCrop(atom_positions, atom_size))

    def plot(self, context: AtomCropState):
        from matplotlib import pyplot as plt

        mean_image = context["mean_image"]
        atom_positions = context["atom_positions"]

        plt.figure()
        plt.title("Detected spots on mean image")
        plt.imshow(mean_image.mean(axis=0))
        plt.scatter(
            atom_positions[:, 1],
            atom_positions[:, 0],
            s=60,
            facecolors="none",
            edgecolors="r",
            linewidths=1.5,
        )
        plt.show()
