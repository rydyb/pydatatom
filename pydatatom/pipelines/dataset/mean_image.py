import numpy as np
from dataclasses import dataclass
from pydatatom.datasets import Dataset
from ..step import Step


@dataclass
class MeanImageState:
    mean_image: np.ndarray | None = None
    mean_count: int = 0


class MeanImageStep(Step):
    def fit(self, context: MeanImageState, dataset: Dataset):
        sum = None
        count = 0

        for image in dataset:
            if sum is None:
                sum = image.copy().astype(np.float64)
            else:
                sum += image
                count += 1

        context["mean_image"] = sum / count
        context["mean_count"] = count

    def transform(self, context: MeanImageState, dataset: Dataset):
        return dataset

    def plot(self, context: MeanImageState):
        from matplotlib import pyplot as plt

        image = context["mean_image"]
        count = context["mean_count"]
        n = image.shape[0]

        fig, axis = plt.subplots(n, 1)
        for i in range(n):
            axis[i].imshow(image[i])
            axis[i].set_title(f"Mean of {count} images ({i + 1} of {n})")
        plt.show()
