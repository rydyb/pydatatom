import pandas as pd
from torch.utils.data import Dataset as TorchDataset


class Dataset(TorchDataset):
    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame(list(self))
