# Numpyro Template Project
Numpyro deep probabilistic programming made easi**er**.

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

* [Numpyro Template Project](#numpyro-template-project)
	* [Requirements](#requirements)
	* [Features](#features)
	* [Folder Structure](#folder-structure)
	* [Experiment Tracking](#experiment-tracking)
	* [License](#license)
	* [Acknowledgements](#acknowledgements)

<!-- /code_chunk_output -->

## Requirements
* Python >= 3.5 (3.6 recommended)
* numpyro >= 0.18.0
* tqdm (Optional for `test.py`)
* tensorboard >= 1.14 (see [Tensorboard Visualization](#tensorboard-visualization))

## Features
* Clear folder structure which is suitable for configurable probabilistic programming projects.
* `.yaml` config file support for convenient parameter tuning.
* Customizable command line options for more convenient parameter tuning.
* Checkpoint saving and resuming.
* Abstract base classes for faster development:
  * `Trainer` handles training process logging and more.
  * `DataModule` handles data shuffling and validation data splitting.
  * `ParaMonad` handles checkpoint saving/resuming, updating of mutable parameters, and JAX RNG keys.

## Folder Structure
  ```
  numpyro_template/
  │
  ├── train.py - main script to start training
  ├── test.py - evaluation of trained model
  │
  ├── configs/ - Hydra configuration files for , models, guides, parametric monads, and trainers
  │   ├── data/ - data-module configurations
  │   ├── guide/ - variational guide program or sampler configurations
  │   ├── logger/ - experiment-tracking configurations (Tensorboard, Weights & Biases)
  │   ├── model/ - generative model program configurations
  │   ├── monad/ - state monad configurations for RNG keys, mutable parameters and optimizer states, etc.
  |   └── trainer/ - trainer class configurations
  │
  ├── data/ - default directory for storing input data
  │
  ├── notebooks/ - Jupyter notebooks showing off results
  │   └── vae.ipynb - an example with a trained Variational Autoencoder
  |
  ├── src/ - core source code in Python
  │   ├── data/ - data modules and the core `DataModule` class
  │   ├── logger/ - logger source
  │   ├── model/ - model source code in Numpyro
  │   ├── trainer/ - source code to `ParaMonad`, `Trainer`, and their subclasses
  |   └── utils/ - small utility functions
      ├── util.py
      └── ...  
  ```

## Experiment Tracking
Both `src/train.py` and `src/test.py` log to whichever writers the `logger`
config group selects, and each writer receives the same scalars, histograms and
image batches:

```bash
python src/train.py                     # Tensorboard (the default)
python src/train.py logger=wandb        # Weights & Biases
python src/train.py logger=many_loggers # both at once
python src/train.py '~logger'           # no experiment tracking
```

Weights & Biases access details live in `configs/logger/wandb.yaml` and are
overridable from the command line like any other option, e.g.
`logger=wandb logger.wandb.entity=my-team logger.wandb.project=vae logger.wandb.mode=offline`.
They default to the standard `WANDB_*` environment variables, which `rootutils`
loads from a `.env` file in the project root:

```bash
WANDB_API_KEY=...            # keep the key here, *not* in the config
WANDB_ENTITY=my-team
WANDB_PROJECT=numpyro-template
WANDB_BASE_URL=...           # only for a self-hosted W&B server
WANDB_MODE=online            # "offline" logs locally for a later `wandb sync`
```

Since W&B run steps must increase monotonically while training and validation
each count steps within their own epoch, per-step scalars are plotted against
the `train/step` and `valid/step` metrics, and epoch-level metrics
(`train/…`, `val/…`) against `epoch`.

## License
This project is licensed under the MIT License. See  LICENSE for more details

## Acknowledgements
This project is inspired by the project [Tensorflow-Project-Template](https://github.com/MrGemy95/Tensorflow-Project-Template) by [Mahmoud Gemy](https://github.com/MrGemy95)
