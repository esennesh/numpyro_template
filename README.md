# Numpyro Template Project
Numpyro deep probabilistic programming made easi**er**.

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

* [Numpyro Template Project](#numpyro-template-project)
	* [Requirements](#requirements)
	* [Features](#features)
	* [Folder Structure](#folder-structure)
	* [Usage](#usage)
		* [Config file format](#config-file-format)
		* [Using config files](#using-config-files)
		* [Resuming from checkpoints](#resuming-from-checkpoints)
    * [Using Multiple GPU](#using-multiple-gpu)
	* [Customization](#customization)
		* [Custom CLI options](#custom-cli-options)
		* [Data Loader](#data-loader)
		* [Trainer](#trainer)
		* [Model](#model)
		* [Loss](#loss)
		* [metrics](#metrics)
		* [Additional logging](#additional-logging)
		* [Validation data](#validation-data)
		* [Checkpoints](#checkpoints)
    * [Tensorboard Visualization](#tensorboard-visualization)
	* [Contribution](#contribution)
	* [TODOs](#todos)
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

## License
This project is licensed under the MIT License. See  LICENSE for more details

## Acknowledgements
This project is inspired by the project [Tensorflow-Project-Template](https://github.com/MrGemy95/Tensorflow-Project-Template) by [Mahmoud Gemy](https://github.com/MrGemy95)
