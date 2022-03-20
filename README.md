# transformer
Implementing a transformer from scratch in Jax, Flax and Torch.

## Installation

### Data

We use the [WMT'14 English-German data](https://nlp.stanford.edu/projects/nmt/) to train and test the transformer models. To download, you can run the `get_data.sh` script.

### Code

[Poetry](https://github.com/python-poetry/poetry) is used for dependency management.

Run `poetry install` to install dependencies and create the virtual environment.

All Python commands should then be prefixed by `poetry run`, e.g. `poetry run python test.py`
