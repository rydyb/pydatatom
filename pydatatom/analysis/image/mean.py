import torch
from tqdm.auto import tqdm
from pydatatom.dataset import Dataset


def mean(
    dataset: Dataset,
    subset_size: int = None,
    batch_size=32,
    num_workers=16,
    shuffle=False,
):
    mean_image = None

    if subset_size is not None and subset_size < len(dataset):
        indices = torch.randperm(len(dataset))[:subset_size]
        dataset = torch.utils.data.Subset(dataset, indices)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle
    )

    for images in tqdm(dataloader):
        # sum over the batch dimension
        image = images.numpy().sum(axis=1)

        if mean_image is None:
            mean_image = image.copy()
        else:
            mean_image += image

    return mean_image / len(dataset)
