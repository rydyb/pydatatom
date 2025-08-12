# pyeval

Package for fast, simple and modular evaluation of atom images.

## Install

Create a virtual environment:
```shell
python -m venv .venv
```

To activate the virtual environment:
```shell
source .venv/bin/activate
```

Install the dependencies (if not done already):
```shell
pip install .
```

## Usage

Inside the python shell or a notebook:
```python
from pyeval import GzipPickleDataset, LossCorrectedEvaluator


dataset = GzipPickleDataset('path/to/dataset')

evaluator = LossCorrectedEvaluator(num_spots=16)
evaluator.evaluate(dataset)

df = evaluator.to_df()

plt.figure()
plt.plot(df['aom_frequency'], df['atoms'])
plt.show()
```
