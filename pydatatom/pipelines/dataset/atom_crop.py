import numpy as np
from dataclasses import dataclass
from pydatatom.datasets import Dataset, Transform
from pydatatom.transforms import PointCrop
from pydatatom.detectors import TopNNMSBlobDetector
from ..step import Step
from .image_mean import ImageMeanState


@dataclass
class AtomCropState(ImageMeanState):
    atom_centers: np.ndarray | None = None


class AtomCropStep(Step):
    def __init__(self, atom_num: int, atom_size: int = 3):
        self.detector = TopNNMSBlobDetector(blob_num=atom_num, blob_size=atom_size)

    @property
    def atom_num(self):
        return self.detector.blob_num

    @property
    def atom_size(self):
        return self.detector.blob_size

    def fit(self, context: ImageMeanState, dataset: Dataset):
        image_mean = context["image_mean"]
        context["atom_centers"] = self.detector.fit(image_mean.mean(axis=0))

    def transform(self, context: AtomCropState, dataset: Dataset):
        atoms = context["atom_centers"].round().astype(int)
        return Transform(dataset, PointCrop(atoms, self.atom_size))

    def plot(self, context: AtomCropState):
        from matplotlib import pyplot as plt

        plt.figure()
        plt.title("Detected spots on mean image")
        plt.imshow(context["image_mean"].mean(axis=0))
        plt.scatter(
            context["atom_centers"][:, 1],
            context["atom_centers"][:, 0],
            s=60,
            facecolors="none",
            edgecolors="r",
            linewidths=1.5,
        )
        plt.show()
