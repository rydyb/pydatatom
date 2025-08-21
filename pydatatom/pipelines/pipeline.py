from .step import Step


class Pipeline:
    def __init__(self, steps: list[Step] = []):
        self.steps = []
        self.context = {}

        for step in steps:
            self.add_step(step)

    def add_step(self, step: Step, index: int | None = None):
        if index is None:
            self.steps.append(step)
        else:
            self.steps.insert(index, step)

    def run(self, dataset):
        for step in self.steps:
            step.fit(self.context, dataset)
            dataset = step.transform(self.context, dataset)

        return dataset

    def plot_step(self, step_index: int):
        step = self.steps[step_index]
        step.plot(self.context)
