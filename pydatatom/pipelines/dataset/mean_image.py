import numpy as np
from dataclasses import dataclass
from pydatatom.datasets import Dataset, Transform
from pydatatom.transforms import PickKey
from .step import Step


@dataclass
class MeanImageState:
    mean_image: np.ndarray | None = None
    mean_count: int = 0


class MeanImageStep(Step):
    def fit(self, context: MeanImageState, dataset: Dataset):
        mean = None
        count = 0

        for image in dataset:
            if mean is None:
                mean = image.copy().astype(np.float64)
            else:
                mean += image
                count += 1

        context.mean_image = mean / count
        context.mean_count = count

    def transform(self, context: MeanImageState, dataset: Dataset):
        return dataset

    def plot(self, context: MeanImageState):
        from matplotlib import pyplot as plt

        plt.figure()
        plt.imshow(context.mean_image)
        plt.title(f"MeanImage of {context.mean_count} images")
        plt.show()
