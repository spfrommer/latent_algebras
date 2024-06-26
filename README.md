# Transport of Algebraic Structure to Latent Embeddings

This codebase accompanies [our paper](https://arxiv.org/abs/2405.16763) ([project website](https://spfrommer.github.io/latent_algebras/)):

**Transport of Algebraic Structure to Latent Embeddings**\
Samuel Pfrommer, Brendon G. Anderson, Somayeh Sojoudi\
2024 International Conference on Machine Learning (ICML, **Spotlight**).

<p align="center">
  <img src="./readme_fig.png" width="850"/>
</p>

## Setup and execution
To set up the environment, make a [virtualenv](https://virtualenv.pypa.io/en/latest/) and run `bash setup.sh`.

Make sure to log in to a [Weights & Biases](https://wandb.ai/site) account (free for academic users).

All scripts to reproduce the results should then be run from within `latalg` with the virtual environment activated.
```console
bash experiments/gen_data.sh
bash experiments/train_latent_model.sh
bash experiments/train_oeprator_modules.sh
bash experiments/test.sh
```

Output figures will lie in the `test_out` directory.

## Key implementations
The various combinations of candidate operations are defined in `main/algebra.py`, as are Algorithms 2 and 3 for generating terms. The parameterization of the induced latent algebra is defined in `main/operator_module.py`. The computation of metrics is defined in `_step_variables` in `main/operator_module.py`. The `data` directory contains code for generating the dataset of INRs as well as training an inr2vec encoder-decoder architecture over this dataset.
