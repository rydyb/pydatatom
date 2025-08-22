import numpy as np
from dataclasses import dataclass

from pydatatom.dataset import Dataset

from .step import Step


@dataclass
class ImageMeanState:
    image_mean: np.ndarray | None = None
    image_count: int = 0


class ImageMeanStep(Step):
    def fit(self, context: ImageMeanState, dataset: Dataset):
        sum = None
        count = 0

        for image in dataset:
            if sum is None:
                sum = image.copy().astype(np.float64)
            else:
                sum += image
            count += 1

        context["image_mean"] = sum / count
        context["image_count"] = count

    def transform(self, context: ImageMeanState, dataset: Dataset):
        return dataset

    def plot(self, context: ImageMeanState):
        from matplotlib import pyplot as plt

        mean = context["image_mean"]
        count = context["image_count"]
        nimages = mean.shape[0]

        fig, axis = plt.subplots(nimages, 1)
        for i in range(nimages):
            axis[i].imshow(mean[i])
            axis[i].set_title(f"Mean of {count} images ({i + 1} of {nimages})")
        plt.show()
