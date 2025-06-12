import argparse
import collections
import hydra
import logging
from numpyro import optim
from omegaconf import DictConfig
import os
import rootutils
from typing import Any, Callable, Dict, List, Optional, Tuple

from .data.datamodule import DataModule
from .trainer import ParaMonad, Trainer
from .utils import extras, get_metric_value, task_wrapper

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

log = logging.LoggerAdapter(logger=logging.getLogger(__name__))
@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: DataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating generative model <{cfg.model._target_}>")
    model: Callable = hydra.utils.instantiate(cfg.model)
    log.info(f"Instantiating guide inference program <{cfg.guide._target_}>")
    guide: Callable = hydra.utils.instantiate(cfg.guide)

    log.info(f"Instantiating trainable module <{cfg.monad._target_}>")
    monad: ParaMonad = hydra.utils.instantiate(cfg.monad,
                                               data_shape=datamodule.shape,
                                               guide=guide, model=model)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: BaseTrainer = hydra.utils.instantiate(cfg.trainer, logger=log)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "monad": monad,
        "trainer": trainer,
    }
    # log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.train(monad, datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.train_metrics.result()

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = str(trainer.checkpoint_dir + '/model_best')
        if ckpt_path == "" or not os.path.exists(ckpt_path):
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        test_metrics = trainer.test(monad, datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")
    else:
        test_metrics = {}

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict

@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
