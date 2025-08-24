import pandas as pd

from pydatatom.dataset import Dataset, Transform
from pydatatom.transform import PickType

from .step import Step


class Pipeline:
    def __init__(self, name: str, steps: list[Step] = []):
        self.name = name
        self.steps = []
        self.context = {}

        for step in steps:
            self.add_step(step)

    def add_step(self, step: Step, index: int | None = None):
        if index is None:
            self.steps.append(step)
        else:
            self.steps.insert(index, step)

    def fit(self, dataset: Dataset):
        for step in self.steps:
            step.fit(self.context, dataset)
            dataset = step.transform(self.context, dataset)

    def transform(self, dataset: Dataset):
        for step in self.steps:
            dataset = step.transform(self.context, dataset)

        return dataset

    def plot_step(self, step_index: int):
        step = self.steps[step_index]
        step.plot(self.context)

    def dataframe(
        self, raw_dataset: Dataset, drop_nan: bool = True, drop_nunique: bool = True
    ) -> pd.DataFrame:
        tran_dataset = self.transform(raw_dataset)

        tran_df = Transform(
            tran_dataset, [lambda x: {self.name: x.tolist()}]
        ).to_pandas()
        meta_df = Transform(
            raw_dataset,
            [
                PickType(float),
            ],
        ).to_pandas()

        df = pd.concat([meta_df, tran_df], axis=1)
        if drop_nan:
            df = df.dropna()
        if drop_nunique:
            df = df.loc[:, df.nunique(dropna=True).ne(1)].copy()

        return df
