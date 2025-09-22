import numpy as np
import torch
from tqdm.auto import tqdm
from pydatatom.dataset import Dataset


def count(
    dataset: Dataset,
    subset_size: int = None,
    batch_size=32,
    num_workers=16,
    shuffle=False,
):
    counts = None

    if subset_size is not None and subset_size < len(dataset):
        indices = torch.randperm(len(dataset))[:subset_size]
        dataset = torch.utils.data.Subset(dataset, indices)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle
    )

    for i, patches in enumerate(tqdm(dataloader)):
        patches = patches.numpy().mean(axis=(-1, -2))

        if counts is None:
            counts = np.zeros((len(dataset), *patches.shape))

        counts[batch_size * i : batch_size * (i + 1), :, :] = patches

    return counts
